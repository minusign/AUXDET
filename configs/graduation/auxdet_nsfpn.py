_base_ = ['./auxdet_base.py']

model = dict(
    neck=dict(
        type='AuxFPNFusion',
        fusion_type='ns'))

