_base_ = ['./auxdet_r50_fpn_1x_voc.py']

model = dict(
    neck=dict(
        enable_moe=True,
        moe_cfg=dict(
            num_experts=4,
            expert_hidden_ratio=0.25,
            temperature=1.0,
            residual_scale_init=0.01)))
