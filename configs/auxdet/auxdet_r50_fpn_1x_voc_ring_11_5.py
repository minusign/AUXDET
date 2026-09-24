_base_ = ['./auxdet_r50_fpn_1x_voc.py']

model = dict(
    neck=dict(
        vmcr_enabled=True,
        vmcr_levels=(0, 1),
        vmcr_outer_kernel=11,
        vmcr_inner_kernel=5,
    ))
