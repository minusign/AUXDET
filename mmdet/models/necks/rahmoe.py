# -*- coding: utf-8 -*-
"""
RAHMoE: Response-Aware Hierarchical Mixture of Experts.

Deployed after decoder fusion stages (d3 and d2), this module applies
per-pixel dense soft routing over 1 shared expert and 4 response-state
experts to handle heterogeneous local response states.
"""

# This file is adapted from PTBD-Net's ``model/rahmoe.py``.
# Original copyright: (c) 2026 Gong-Yi.
# PTBD-Net is distributed under the MIT License. AuxDet adapts the module
# interface for use as an MMDetection neck component.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ECA2d(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        y = self.pool(x).flatten(2).transpose(1, 2)
        y = torch.sigmoid(self.conv(y))
        return x * y.transpose(1, 2).unsqueeze(-1)


# ---------------------------------------------------------------------------
#  Experts
# ---------------------------------------------------------------------------

class SharedDecoderExpert(nn.Module):
    """Always-active shared expert for stable general decoding."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class LowResponseTargetExpert(nn.Module):
    """Enhances weak targets via local contrast gating.

    x − AvgPool7(x) highlights local protrusions even when absolute
    response is low, preventing dim targets from being ignored.
    """
    def __init__(self, dim):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.contrast_gate = nn.Sequential(
            nn.Conv2d(1, dim, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        response = x.abs().mean(dim=1, keepdim=True)
        contrast = response - F.avg_pool2d(response, 7, 1, 3)
        return self.local(x) * (1.0 + self.contrast_gate(contrast))


class CompactTargetRefineExpert(nn.Module):
    """Refines compact candidate responses with dual 3×3 / 5×5 receptive fields."""
    def __init__(self, dim):
        super().__init__()
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )
        self.eca = ECA2d(dim)

    def forward(self, x):
        return self.eca(self.fuse(torch.cat([self.dw3(x), self.dw5(x)], dim=1)))


class HighResponseAmbiguityExpert(nn.Module):
    """Contextual recalibration for strong-but-ambiguous responses.

    Large-kernel (7×7) and dilated (3×3, d=2) context are fused to
    decide whether a high response should keep its own feature or
    rely more on surrounding context.
    """
    def __init__(self, dim):
        super().__init__()
        self.large = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        c1, c2 = self.large(x), self.dilated(x)
        g = self.gate(torch.cat([c1, c2], dim=1))
        return g * x + (1.0 - g) * 0.5 * (c1 + c2)


class StructuredClutterExpert(nn.Module):
    """Suppresses directional structural clutter via 1×7 / 7×1 strip convolutions.

    Cloud edges, horizons, and building contours typically exhibit
    strong directionality that this expert can recognise.
    """
    def __init__(self, dim):
        super().__init__()
        self.h_conv = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3),
                                groups=dim, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0),
                                groups=dim, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.fuse(torch.cat([self.h_conv(x), self.v_conv(x)], dim=1))
        return s * self.gate(s)


# ---------------------------------------------------------------------------
#  Router
# ---------------------------------------------------------------------------

class DenseResponseStateRouter(nn.Module):
    """Per-pixel dense soft router.

    Constructs 6 explicit response-state cues (mean response, local contrast,
    multi-scale context, directional structure) and fuses them with a
    compressed feature descriptor to produce per-expert gates via Softmax.

    AuxDet's optional metadata adaptation supports a logit-bias mode and a
    FiLM mode. Both use one final visual-router logits layer and one softmax;
    disabled mode creates no extra parameters. route_diagnostics only retains
    detached scalar/four-element statistics.
    """
    NUM_CUES = 6

    def __init__(self, dim, num_experts=4, hidden=32, temperature=1.5,
                 metadata_conditioned_router=False, meta_feature_dim=128,
                 meta_router_hidden=32, meta_router_scale_init=1.0,
                 metadata_conditioning_type=None, film_strength=0.5,
                 film_hidden=64, route_diagnostics=False):
        super().__init__()
        self.num_experts = num_experts
        if metadata_conditioning_type is None:
            metadata_conditioning_type = (
                'logit_bias' if metadata_conditioned_router else 'visual')
        if metadata_conditioning_type not in ('visual', 'logit_bias', 'film'):
            raise ValueError('metadata_conditioning_type must be visual, '
                             'logit_bias, or film.')
        self.metadata_conditioning_type = metadata_conditioning_type
        self.metadata_conditioned_router = metadata_conditioning_type != 'visual'
        self.meta_feature_dim = meta_feature_dim
        self.film_strength = float(film_strength)
        self.route_diagnostics = route_diagnostics
        self.route_stats = {}

        self.feat_reduce = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            LayerNorm2d(hidden), nn.GELU(),
        )
        self.router = nn.Sequential(
            nn.Conv2d(hidden + self.NUM_CUES, hidden, 3, padding=1, bias=False),
            LayerNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, num_experts, 1, bias=True),
        )
        self.register_buffer("temperature", torch.tensor(float(temperature)))

        if self.metadata_conditioning_type == 'logit_bias':
            if meta_feature_dim <= 0 or meta_router_hidden <= 0:
                raise ValueError('Metadata router dimensions must be positive.')
            # AuxDet metadata-conditioning adaptation of PTBD-Net RAHMoE.
            # The existing visual router and all experts remain unchanged.
            self.metadata_mlp = nn.Sequential(
                nn.LayerNorm(meta_feature_dim),
                nn.Linear(meta_feature_dim, meta_router_hidden), nn.GELU(),
                nn.Linear(meta_router_hidden, num_experts),
            )
            nn.init.zeros_(self.metadata_mlp[-1].weight)
            nn.init.zeros_(self.metadata_mlp[-1].bias)
            self.meta_scale = nn.Parameter(
                torch.tensor(float(meta_router_scale_init)))
        elif self.metadata_conditioning_type == 'film':
            if meta_feature_dim <= 0 or film_hidden <= 0:
                raise ValueError('FiLM dimensions must be positive.')
            # The visual hidden tensor is the output of router[0:3] and has
            # exactly ``hidden`` channels. The last layer remains router[3].
            self.film_mlp = nn.Sequential(
                nn.LayerNorm(meta_feature_dim),
                nn.Linear(meta_feature_dim, film_hidden), nn.GELU(),
                nn.Linear(film_hidden, hidden * 2),
            )
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)

    @torch.no_grad()
    def set_temperature(self, value):
        self.temperature.fill_(max(0.3, float(value)))

    def build_cues(self, x):
        feat = self.feat_reduce(x)

        r = x.abs().mean(dim=1, keepdim=True)                          # mean response
        c = r - F.avg_pool2d(r, 7, 1, 3)                               # local contrast
        s3 = F.avg_pool2d(r, 3, 1, 1)                                  # 3×3 context
        s5 = F.avg_pool2d(r, 5, 1, 2)                                  # 5×5 context
        s7 = F.avg_pool2d(r, 7, 1, 3)                                  # 7×7 context
        gh = torch.abs(r - F.avg_pool2d(r, (1, 7), 1, (0, 3)))        # horizontal structure
        gv = torch.abs(r - F.avg_pool2d(r, (7, 1), 1, (3, 0)))        # vertical structure

        return torch.cat([feat, r, c, s3, s5, s7, gh + gv], dim=1)

    def forward(self, x, meta_feature=None):
        prompt = self.build_cues(x)
        # Keep the existing Sequential module and state keys. h is the actual
        # hidden feature immediately before the original logits convolution.
        h = self.router[0](prompt)
        h = self.router[1](h)
        h = self.router[2](h)
        visual_hidden = h
        meta_output = None
        if self.metadata_conditioning_type == 'film':
            meta_output = self._validate_meta(x, meta_feature)
            film_params = self.film_mlp(meta_output)
            gamma, beta = film_params.chunk(2, dim=1)
            gamma = gamma.to(h.dtype)[:, :, None, None]
            beta = beta.to(h.dtype)[:, :, None, None]
            h = (h * (1.0 + self.film_strength * gamma.tanh())
                 + self.film_strength * beta.tanh())
        visual_logits = self.router[3](h)
        logits = visual_logits
        meta_logits = None
        if self.metadata_conditioning_type == 'logit_bias':
            meta_output = self._validate_meta(x, meta_feature)
            meta_logits = self.metadata_mlp(meta_output)
            # Preserve the visual path's dtype under AMP and apply exactly
            # one softmax to the combined logits. The cast preserves gradients.
            meta_bias = (self.meta_scale * meta_logits).to(visual_logits.dtype)
            logits = visual_logits + meta_bias[:, :, None, None]
        gates = torch.softmax(logits / self.temperature.clamp_min(0.3), dim=1)
        self.route_stats = {}
        if self.route_diagnostics:
            with torch.no_grad():
                visual = visual_logits.detach().float()
                probs = gates.detach().float()
                self.route_stats = {
                    'visual_logits_mean': visual.mean(),
                    'visual_logits_std': visual.std(unbiased=False),
                    # Keep this statistic on the unconditioned output of
                    # router[0:3], so it is comparable across router modes.
                    'visual_hidden_mean': visual_hidden.detach().float().mean(),
                    'visual_hidden_std': visual_hidden.detach().float().std(unbiased=False),
                    'expert_usage': probs.mean(dim=(0, 2, 3)),
                    'entropy': -(probs * probs.clamp_min(1e-8).log()
                                 ).sum(dim=1).mean(),
                }
                if meta_logits is not None:
                    meta = meta_logits.detach().float()
                    centered = meta - meta.mean(dim=1, keepdim=True)
                    self.route_stats.update(
                        meta_logits_mean=meta.mean(),
                        meta_logits_std=meta.std(unbiased=False),
                        meta_logits_centered_std=centered.std(unbiased=False),
                        meta_scale=self.meta_scale.detach().float().clone())
                elif self.metadata_conditioning_type == 'film':
                    params = self.film_mlp(meta_output).detach().float()
                    gamma, beta = params.chunk(2, dim=1)
                    self.route_stats.update(
                        film_gamma_mean=gamma.mean(),
                        film_gamma_std=gamma.std(unbiased=False),
                        film_beta_mean=beta.mean(),
                        film_beta_std=beta.std(unbiased=False),
                        film_strength=visual_hidden.new_tensor(self.film_strength))
        return gates, logits

    def _validate_meta(self, x, meta_feature):
        if (meta_feature is None or meta_feature.ndim != 2
                or meta_feature.shape != (x.shape[0], self.meta_feature_dim)):
            raise ValueError('meta_feature must have shape '
                             '[B, meta_feature_dim] when metadata routing '
                             'is enabled.')
        if meta_feature.device != x.device:
            raise ValueError('meta_feature and visual features must share '
                             'the same device.')
        return meta_feature


# ---------------------------------------------------------------------------
#  RAHMoE wrapper
# ---------------------------------------------------------------------------

class RAHMoE(nn.Module):
    """Response-Aware Hierarchical Mixture of Experts.

    Deployed at decoder stages d3 (64×64) and d2 (128×128).
    Output:  F_out = F + γ_shared · Shared(F) + γ_routed · Σ_e π_e ⊙ Expert_e(F)

    where π_e ∈ [0, 1]^{B×1×H×W} are per-pixel soft router weights
    (dense routing — all experts always participate).

    meta_feature is required for the ``logit_bias`` and ``film`` router
    modes. The optional AuxDet adaptation conditions only the router; experts
    and the four original auxiliary loss formulas remain the PTBD-Net
    implementation.
    """
    def __init__(self, dim, num_experts=4, router_hidden=32, temperature=1.5,
                 gamma_shared_init=0.1, gamma_routed_init=0.1,
                 balance_weight=1e-3, diversity_weight=1e-4,
                 entropy_weight=1e-4, zloss_weight=1e-5,
                 entropy_target=1.05, compute_aux_loss=True,
                 metadata_conditioned_router=False, meta_feature_dim=128,
                 meta_router_hidden=32, meta_router_scale_init=1.0,
                 metadata_conditioning_type=None, film_strength=0.5,
                 film_hidden=64,
                 route_diagnostics=False):
        super().__init__()
        if num_experts != 4:
            raise ValueError('RAHMoE currently requires num_experts=4.')
        self.num_experts = num_experts
        self.compute_aux_loss = compute_aux_loss
        if metadata_conditioning_type is None:
            metadata_conditioning_type = (
                'logit_bias' if metadata_conditioned_router else 'visual')
        if metadata_conditioning_type not in ('visual', 'logit_bias', 'film'):
            raise ValueError('metadata_conditioning_type must be visual, '
                             'logit_bias, or film.')
        self.metadata_conditioning_type = metadata_conditioning_type
        self.metadata_conditioned_router = metadata_conditioning_type != 'visual'

        self.shared = SharedDecoderExpert(dim)
        self.experts = nn.ModuleList([
            LowResponseTargetExpert(dim),
            CompactTargetRefineExpert(dim),
            HighResponseAmbiguityExpert(dim),
            StructuredClutterExpert(dim),
        ])
        self.router = DenseResponseStateRouter(
            dim=dim, num_experts=num_experts,
            hidden=router_hidden, temperature=temperature,
            metadata_conditioned_router=metadata_conditioned_router,
            meta_feature_dim=meta_feature_dim,
            meta_router_hidden=meta_router_hidden,
            meta_router_scale_init=meta_router_scale_init,
            metadata_conditioning_type=metadata_conditioning_type,
            film_strength=film_strength,
            film_hidden=film_hidden,
            route_diagnostics=route_diagnostics,
        )

        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.gamma_shared = nn.Parameter(torch.tensor(float(gamma_shared_init)))
        self.gamma_routed = nn.Parameter(torch.tensor(float(gamma_routed_init)))

        self.balance_weight   = balance_weight
        self.diversity_weight = diversity_weight
        self.entropy_weight   = entropy_weight
        self.zloss_weight     = zloss_weight
        self.entropy_target   = entropy_target

        self.aux_loss  = None
        self.aux_loss_logs = {}
        self.route_logs = {}

    def forward(self, x, return_vis=False, meta_feature=None):
        shared = self.shared(x)

        if self.metadata_conditioning_type != 'visual':
            gates, logits = self.router(x, meta_feature=meta_feature)
        else:
            gates, logits = self.router(x)                              # [B, E, H, W]
        expert_outputs = [e(x) for e in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)               # [B, E, C, H, W]

        weighted = gates[:, :, None] * expert_stack
        routed = weighted.sum(dim=1)
        routed = self.out_proj(routed)

        out = x + self.gamma_shared * shared + self.gamma_routed * routed

        # The neck is also used by validation and inference through the same
        # feature-extraction path.  Auxiliary routing regularization is a
        # training-only loss and must not be computed in eval mode.
        self.aux_loss_logs = {}
        if self.compute_aux_loss and self.training:
            self.aux_loss = self._compute_aux(gates, logits, expert_stack)
        else:
            self.aux_loss = None
        # Opt-in compact detached statistics; never cache full-resolution
        # gates on the module. return_vis still exposes them explicitly.
        self.route_logs = dict(self.router.route_stats)

        if not return_vis:
            return out

        vis = {
            "input_feature": x.detach(),
            "response_map": x.detach().abs().mean(dim=1, keepdim=True),
            "router_logits": logits.detach(),
            "route_weights": gates.detach(),
            "expert_outputs": expert_stack.detach(),
            "weighted_outputs": weighted.detach(),
            "shared_output": shared.detach(),
            "routed_sum": weighted.sum(dim=1).detach(),
            "routed_projection": routed.detach(),
            "final_output": out.detach(),
        }
        return out, vis

    def _compute_aux(self, gates, logits, experts):
        # Keep the original formulas while evaluating their reductions in
        # float32.  This avoids half-precision underflow/overflow in log,
        # normalization and logsumexp without detaching the graph.
        gates = gates.float()
        logits = logits.float()
        experts = experts.float()
        E = self.num_experts

        #  balance:  prevent collapse to a single expert
        usage = gates.mean(dim=(0, 2, 3))
        balance = (usage - torch.full_like(usage, 1.0 / E)).pow(2).sum()

        #  entropy target:  prevent gates from becoming too uniform or too peaked
        entropy = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()
        entropy_loss = (entropy - self.entropy_target).pow(2)

        #  diversity:  encourage experts to produce different outputs
        pooled = F.normalize(experts.mean(dim=(-1, -2)), dim=-1)        # [B, E, C]
        sim = torch.matmul(pooled, pooled.transpose(-1, -2))            # [B, E, E]
        eye = torch.eye(E, device=sim.device, dtype=torch.bool)
        diversity = sim[:, ~eye].pow(2).mean()

        #  z-loss:  stabilise router logit magnitude
        zloss = torch.logsumexp(logits, dim=1).pow(2).mean()

        # Detached scalars for diagnosis only; the returned loss below keeps
        # its original graph and coefficients. Never retain full activations.
        self.aux_loss_logs = {
            'balance': balance.detach(),
            'diversity': diversity.detach(),
            'entropy': entropy_loss.detach(),
            'zloss': zloss.detach(),
        }

        return (self.balance_weight * balance
                + self.diversity_weight * diversity
                + self.entropy_weight * entropy_loss
                + self.zloss_weight * zloss)

    @torch.no_grad()
    def set_temperature(self, value):
        self.router.set_temperature(value)

