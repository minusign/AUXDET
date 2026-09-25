_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# B1: AuxDet + LFP on the P2/C2 lateral only.
model = dict(
    neck=dict(
        lfp_cfg=dict(
            wave='haar',
            mode='zero',
            with_gauss=True,
            gauss_gate=0.5,
        ),
        lfp_levels=(0,),
    )
)
