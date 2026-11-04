_base_ = ['./auxdet_r50_fpn_1x_voc.py']

model = dict(
    neck=dict(
        enable_rahmoe=True,
        rahmoe_level=1,
        rahmoe_cfg=dict(
            num_experts=4,
            router_hidden=32,
            temperature=1.5,
            gamma_shared_init=0.1,
            gamma_routed_init=0.1,
            compute_aux_loss=False,
        ),
    ))
