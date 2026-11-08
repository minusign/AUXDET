"""P2 visual RAH-MoE plus the same Quality-Aware StandardRoIHead."""

custom_imports = dict(
    imports=['mmdet.models.roi_heads.quality_aware_standard_roi_head'],
    allow_failed_imports=False)

_base_ = ['./auxdet_r50_fpn_1x_voc_rahmoe_p2.py']

model = dict(
    roi_head=dict(
        type='QualityAwareStandardRoIHead',
        quality_enabled=True,
        quality_score_fusion_enabled=True,
        quality_loss_weight=0.05,
        quality_alpha=0.5,
        quality_hidden_dim=64,
        quality_in_channels=256,
        quality_iou_legacy_coordinate=True),
    neck=dict(
        heatmap_enabled=False,
        heatmap_gate_enabled=False,
        rahmoe_cfg=dict(
            metadata_conditioning_type='visual')))

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
work_dir = './work_dirs/auxdet_rahmoe_p2_quality_seed42_12e'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2,
                    save_best='pascal_voc/AP50', rule='greater'))
log_processor = dict(num_digits=8)
