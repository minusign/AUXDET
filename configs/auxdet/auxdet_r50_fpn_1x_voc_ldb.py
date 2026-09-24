_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# Pixel-detail bypass before the original M2DM loop; VMCR is disabled.
model = dict(neck=dict(
    vmcr_enabled=False,
    ldb_enabled=True,
    ldb_mode='pixel'))

randomness = dict(seed=42)
