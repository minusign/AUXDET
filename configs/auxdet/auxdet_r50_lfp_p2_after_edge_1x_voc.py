_base_ = ['./auxdet_r50_fpn_1x_voc.py']

# B2: AuxDet + LFP on P2 after metadata modulation and EdgeConvSep fusion.
model = dict(
    neck=dict(
        lfp_cfg=dict(
            wave='haar',
            mode='zero',
            with_gauss=True,
            gauss_gate=0.5,
        ),
        lfp_levels=(0,),
        lfp_position='after_edge',
    )
)
