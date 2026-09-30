_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# ALCNet-style MLC scale max pooling on the P2 lateral branch.  LFP, SFS and
# Meta-LFP are explicitly disabled so this is a clean module-transfer baseline.
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
            mode='original',
            metadata_dim=96,
            hidden_dim=64,
            lambda_init=0.1,
        )))

work_dir = './work_dirs/auxdet_r50_mlc_p2_original_1x_voc'
