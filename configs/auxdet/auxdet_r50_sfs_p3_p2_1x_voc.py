_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# B3: SFS-only replacement of the P3 -> P2 top-down fusion.
model = dict(
    neck=dict(
        sfs_cfg=dict(
            n_heads=8,
            n_points=4,
        ),
        sfs_fusions=(0,),
    )
)
