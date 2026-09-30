_base_ = ['./auxdet_r50_lfp_p2_after_edge_1x_voc.py']

# Meta-LFP condition using the matching P2 auxiliary fused feature.
model = dict(
    neck=dict(
        sfs_cfg=None,
        sfs_fusions=(),
        lfp_gate_cfg=dict(
            level=0,
            mode='visual_aux',
            hidden_dim=64,
            gate_init=0.1,
        ),
    )
)

work_dir = 'work_dirs/auxdet_r50_lfp_p2_after_edge_visual_aux_gate_1x_voc'
