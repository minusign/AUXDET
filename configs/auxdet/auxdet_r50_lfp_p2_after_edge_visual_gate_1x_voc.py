_base_ = ['./auxdet_r50_lfp_p2_after_edge_1x_voc.py']

# Meta-LFP visual-only condition. The zero auxiliary branch keeps the gate
# input dimension and parameter count equal to visual_aux_gate.
model = dict(
    neck=dict(
        sfs_cfg=None,
        sfs_fusions=(),
        lfp_gate_cfg=dict(
            level=0,
            mode='visual',
            hidden_dim=64,
            gate_init=0.1,
        ),
    )
)

work_dir = 'work_dirs/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc'
