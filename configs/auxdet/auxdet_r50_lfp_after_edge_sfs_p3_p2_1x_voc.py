_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# B4: LFP on the P2 lateral after edge fusion, then SFS for P3 -> P2.
model = dict(
    neck=dict(
        lfp_cfg=dict(
            wave='haar',
            mode='zero',
            with_gauss=True,
            gauss_gate=0.5,
        ),
        lfp_levels=(0,),
        lfp_position='after_edge',
        sfs_cfg=dict(
            n_heads=8,
            n_points=4,
        ),
        sfs_fusions=(0,),
    )
)
