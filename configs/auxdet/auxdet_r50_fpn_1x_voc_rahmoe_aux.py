_base_ = ['./auxdet_r50_fpn_1x_voc_rahmoe.py']

model = dict(
    neck=dict(
        rahmoe_cfg=dict(
            compute_aux_loss=True,
        ),
    ))
