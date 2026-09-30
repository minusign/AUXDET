_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# Visual-only scale selection.  The metadata argument is absent by design.
model = dict(
    neck=dict(
        lfp_cfg=None,
        lfp_levels=(),
        sfs_cfg=None,
        sfs_fusions=(),
        lfp_gate_cfg=None,
        mlc_cfg=dict(
            level=0,
            channels=256,
            dilations=(1, 2, 3),
            mode='visual',
            metadata_dim=96,
            hidden_dim=64,
            lambda_init=0.1,
        )))

work_dir = './work_dirs/auxdet_r50_mlc_p2_visual_1x_voc'
