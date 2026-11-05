_base_ = ['./auxdet_r50_fpn_1x_voc_rahmoe_aux.py']

model = dict(neck=dict(rahmoe_position='mid_topdown'))

log_processor = dict(num_digits=8)
