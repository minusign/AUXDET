_base_ = ['./auxdet_base.py']

model = dict(
    backbone=dict(
        type='ResNetPConv',
        pconv_kernel=3,
        pconv_stages=(1, 2)))

