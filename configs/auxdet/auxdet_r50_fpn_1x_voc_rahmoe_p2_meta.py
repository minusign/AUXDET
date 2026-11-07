_base_ = ['./auxdet_r50_fpn_1x_voc_rahmoe_p2.py']

model = dict(
    neck=dict(
        heatmap_enabled=False,
        heatmap_gate_enabled=False,
        rahmoe_position='p2_bottleneck',
        rahmoe_level=0,
        rahmoe_bottleneck_dim=64,
        rahmoe_cfg=dict(
            metadata_conditioned_router=True,
            meta_feature_dim=128,
            meta_router_hidden=32,
            meta_router_scale_init=1.0,
            num_experts=4,
            temperature=1.5,
            compute_aux_loss=True,
            route_diagnostics=False,
        ),
    ))

train_cfg = dict(max_epochs=12, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[8, 11], gamma=0.1),
]
randomness = dict(seed=42)
load_from = None
resume = False
work_dir = './work_dirs/auxdet_rahmoe_p2_meta_seed42_12e'
log_processor = dict(num_digits=8)
