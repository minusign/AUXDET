_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# Equal-duration continuation of the original AuxDet path.
model = dict(neck=dict(vmcr_enabled=False, ldb_enabled=False))

randomness = dict(seed=42)
