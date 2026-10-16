"""Common server-facing base for the six AuxDet experiments."""

import os
import os.path as osp

_base_ = ['../auxdet/auxdet_r50_fpn_1x_voc.py']

_default_root = osp.abspath(
    osp.join('{{fileDirname}}', '..', '..', 'data_root')
)
data_root = os.environ.get('AUXDET_DATA_ROOT') or _default_root

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))

# Match the original single-GPU, batch-size-4 effective LR explicitly.
optim_wrapper = dict(optimizer=dict(lr=0.005))
auto_scale_lr = dict(enable=False)

# VOCMetric emits ``pascal_voc/AP50`` at IoU 0.5. Keep one explicit metric
# for checkpoint selection; the base model/evaluator remain inherited.
val_evaluator = dict(type='VOCMetric', metric='mAP', iou_thrs=[0.5], eval_mode='area')
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(interval=1, save_best='pascal_voc/AP50', rule='greater'))
