_base_ = ['./auxdet_base.py']

model = dict(
    neck=dict(
        type='AuxFPNFusion',
        fusion_type='hs',
        hs_ratio=(0.25, 0.25)))

