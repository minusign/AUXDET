# Meta-LFP 最小版本说明

## 实现范围

Meta-LFP 保留现有 LFP、M2DM、LEEM 和检测头，只在 B2 的 `after_edge` 位置增加一个逐样本残差门控。当前配置只作用于 P2 侧向特征 `laterals[0]`。

当前真实特征流为：

```text
backbone lateral features
  -> M2DM auxiliary modulation
  -> LEEM/EdgeConvSep
  -> outer residual edge fusion
  -> LFP(F)
  -> Meta-LFP residual gate
  -> top-down FPN fusion
```

M2DM 每个浅层循环都会生成一次 `all_aux_fea`。当前实现中 `MetaFeatureProcessorWithSem(channel_outs=out_channels//2)` 的最终输出维度为 `[B, 128]`；它融合了视角、波段、尺寸和视觉语义补偿信息，所以本文称它为“辅助融合特征”，不称为纯元数据编码。

门控使用：

```text
F_lfp = LFP(F)
F_out = F + g * (F_lfp - F)
```

其中 `g` 是 `[B,1,1,1]`。`visual` 模式把辅助融合特征置零后送入相同的 MLP；`visual_aux` 模式使用当前 P2 循环的 `all_aux_fea`。两者的 MLP 都是 `384 -> 64 -> 1`，参数量都是 **24,705**：

```text
(256 + 128) * 64 + 64 + 64 + 1 = 24,705
```

最后一层权重为零，bias 为 `logit(0.1)`，所以 `init_weights` 后的初始门控为 0.1。它偏向原始 `F`，不等价于 B2 的直接 `LFP(F)` 初始化。

## 配置

- [visual gate 配置](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/configs/auxdet/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc.py)
- [visual_aux gate 配置](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/configs/auxdet/auxdet_r50_lfp_p2_after_edge_visual_aux_gate_1x_voc.py)
- [门控实现](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/aux_fpn.py)

两份配置都继承 B2，保留相同数据、增强、训练日程、优化器和学习率；显式关闭 SFS，并使用独立 `work_dir`。B0、B1、B2、B3、B4 配置没有新增门控参数和行为变化。

## 运行命令

在服务器的 AuxDet 根目录运行，替换实际 checkpoint 和工作目录：

```bash
python tools/train.py \
  configs/auxdet/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc.py \
  --work-dir work_dirs/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc

python tools/test.py \
  configs/auxdet/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc.py \
  work_dirs/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc/best.pth \
  --work-dir work_dirs/auxdet_r50_lfp_p2_after_edge_visual_gate_1x_voc \
  --out results/ablation/cache/visual_gate.pkl
```

将文件名中的 `visual_gate` 替换为 `visual_aux_gate` 即可运行联合条件版本。当前实现没有默认 `load_from` 或 `resume`，新实验按 B2 的原有初始化方式开始训练。

加载旧 B2 checkpoint 时，新增 `lfp_gates.0.mlp.*` 键缺失是预期现象；应使用 MMEngine 的非严格加载方式并确认日志只报告这些新增键。B0/B1/B2/B3/B4 的旧键不应出现缺失或 unexpected keys。不要把 B2 的旧权重称为 Meta-LFP 已训练权重。

## 统计与实验解释

最近一次 forward 后，可以读取：

```python
stats = model.neck.get_lfp_gate_stats()
# {'0': {'mean': ..., 'std': ..., 'min': ..., 'max': ...}}
```

统计值已经 `detach`，不会保留计算图，也不会在 forward 中打印。

建议至少比较四组：

1. 原始 AuxDet（B0）；
2. 现有 B2（after-edge 后直接输出 LFP）；
3. Meta-LFP `visual`；
4. Meta-LFP `visual_aux`。

本次只有模块和配置验证，没有训练结果，不能据此声称 Meta-LFP 提升了 AP50、Recall 或其他指标。`all_aux_fea` 本身包含视觉补偿信息，因此 `visual` 与 `visual_aux` 的差异不能单独证明收益来自纯元数据。后续做元数据扰动时，只扰动新增门控的条件路径，保留 M2DM、LEEM 的正确输入和视觉补偿信息。

