_base_ = ['./auxdet_r50_fpn_1x_voc_rahmoe_aux.py']

model = dict(
    neck=dict(
        rahmoe_position='p2_bottleneck',
        rahmoe_level=0,
        rahmoe_bottleneck_dim=64,
        p2_residual_scale_init=1.0,
    ))

log_processor = dict(num_digits=8)
