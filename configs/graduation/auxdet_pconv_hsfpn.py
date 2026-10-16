_base_ = ['./auxdet_base.py']

model = dict(
    backbone=dict(
        type='ResNetPConv',
        pconv_kernel=3,
        pconv_stages=(1, 2)),
    neck=dict(
        type='AuxFPNFusion',
        fusion_type='hs',
        hs_ratio=(0.25, 0.25)))

