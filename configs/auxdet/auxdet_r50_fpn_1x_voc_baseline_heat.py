_base_ = ['./auxdet_r50_fpn_1x_voc.py']

model = dict(
    neck=dict(
        enable_rahmoe=False,
        heatmap_enabled=True,
        heatmap_gate_enabled=False,
        heatmap_hidden_channels=64,
        heatmap_loss_weight=0.05,
        heatmap_alpha=2.0,
        heatmap_gamma=4.0,
        heatmap_min_radius=1,
        heatmap_max_radius=16,
        heatmap_min_overlap=0.7,
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
work_dir = './work_dirs/auxdet_baseline_heat_seed42_12e'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2,
                    save_best='pascal_voc/AP50', rule='greater'))
log_processor = dict(num_digits=8)
