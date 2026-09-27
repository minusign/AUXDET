# AuxDet + NS-FPN 功能检查

后续更新（2026-09-27）：下文记录的是初次审计。其指出的 cfg/levels 配套校验和依赖声明现已补齐；原始 JSON 及其源码哈希保留作为历史证据。新增工具与复查结果见[消融实验工具报告](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/docs/zh_cn/auxdet_ablation_tools.md)。

检查日期：2026-09-27。检查对象为当前工作区中的 `lfp.py`、`sfs.py`、`aux_fpn.py` 和 B0–B4 配置。

## 结论

当前 B1–B4 配置中的 LFP 和 SFS 已实际接入 AuxFPN。CPU FP32 数值检查、反向传播检查和 AuxFPN 训练/推理模式检查均通过，未发现当前配置存在小波子带错位、新增参数断梯度或 SFS 输出不连续的问题。

这次检查确认了模块计算机制及其在 neck 中的接入。CUDA 算子、完整检测器的 loss/predict、混合精度和实际 checkpoint 在本轮没有执行验证。已有评测 CSV 也未显示加入模块后整体检测性能超过基线。

另有两项工程问题：新增依赖未纳入项目环境声明；模块配置与层级开关缺少一致性检查。当前 B1–B4 的参数填写完整，不触发后一问题。

## 1. 当前实际结构

| 配置 | LFP 位置 | SFS 位置 | 实测调用顺序（P2 分支） |
| --- | --- | --- | --- |
| B0 | 无 | 无 | 动态调制 → 边缘增强 |
| B1 | P2，调制前 | 无 | LFP → 动态调制 → 边缘增强 |
| B2 | P2，边缘增强后 | 无 | 动态调制 → 边缘增强 → LFP |
| B3 | 无 | P3→P2 | 动态调制 → 边缘增强 → SFS |
| B4 | P2，边缘增强后 | P3→P2 | 动态调制 → 边缘增强 → LFP → SFS |

这是 NS-FPN 的局部适配。P4→P3、P5→P4 仍使用原有上采样相加。B4 的 SFS 使用 P2 作为 query、已融合高层信息的 P3 作为 key/value。

`AuxFPN` 最终返回 P2/P3 两个输出，与当前 RPN 的 `[4, 8]` anchor strides、RoI extractor 的 `[4, 8]` feature strides 匹配。配置中的 `num_outs=4` 与最终返回两个输出是本项目原有约定，未将其判为新增模块错误。

`TwoStageDetector.loss()`、`predict()` 和 `_forward()` 均经由 `extract_feat()` 调用带元数据的 neck；因此源码中不存在仅训练使用 LFP/SFS、推理绕过模块的分支。

相关代码：[接入位置](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/aux_fpn.py:246)、[检测器调用](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/detectors/two_stage.py:115)、[检测头配置](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/configs/auxdet/auxdet_r50_fpn_1x_voc.py:44)。

## 2. 验证方法和环境

- 环境：macOS ARM64，Python 3.12.14、PyTorch 2.2.2、MMCV-lite 2.1.0、MMEngine 0.10.7、pytorch-wavelets 1.3.0、PyWavelets 1.10.0、NumPy 1.26.4。
- 固定随机种子：20260927；CPU FP32。
- 临时 Python 环境：`/private/tmp/auxdet-nsfpn-audit-env`。
- LFP 直接使用项目实现和真实的 `pytorch_wavelets` 运算。
- AuxFPN 使用项目中的真实 metadata processor、DMLP、edge convolution，以及 MMCV 的 ConvModule 和 MMEngine 的 BaseModule。
- CPU 检查绕过 MMDetection 汇总导入中的无关扩展。SFS 的底层 CUDA Function 在测试进程内被明确替换为 MMCV 自带的 `multi_scale_deformable_attn_pytorch`。模型源文件未被改写。
- SFS 对照检查直接读取同级 `NS-FPN` 参考仓库中的类定义，将适配版参数逐项映射到原版，并在相同底层参考算子上比较输出及梯度。另对 attention weights 和 offset residual 施加非零初始化，避免只比较默认均匀注意力情形。

这项替换只能验证 Python 层算法、形状和 autograd 链路，不能作为 CUDA 扩展验证。算子接口及参考实现见 [MMCV 2.1.0 官方源码](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/multi_scale_deform_attn.py)。

完整数值、依赖版本和被检查模型/配置的 SHA-256 保存在 [reference_checks.json](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/nsfpn_audit/reference_checks.json)。

## 3. 功能检查结果

### LFP

| 检查 | 实测结果 |
| --- | --- |
| 多通道 DWT→IDWT，`[2,5,31,29]` | 最大误差 `4.76837158203125e-07` |
| DWT→IDWT，`[1,256,32,32]` | 最大误差 `9.5367431640625e-07` |
| 极小奇数尺寸 DWT→IDWT，`[2,7,1,3]` | 最大误差 `1.1920928955078125e-07` |
| 实际 P2 尺寸 DWT→IDWT，`[1,256,256,256]` | 最大误差 `9.5367431640625e-07` |
| LFP，开启/关闭 Gaussian，`[2,256,31,29]` | 输出尺寸正确、数值有限、输入与全部可训练参数梯度非零 |
| 完整 LFP，`[1,256,256,256]` | 前向/反向通过；attention 和 sigma 均有非零梯度 |
| 偶数尺寸的低频 LL 保留 | 最大误差不超过 `7.16e-07` |
| 固定随机输入上的高频能量 | 两个测试条件下均低于输入高频能量 |

小波正逆变换中的 orientation/channel 维度互为逆操作；奇数尺寸在重建后裁剪到原尺寸。此前记录的子带顺序错误在当前实现中没有复现。

低频引导、高频门控、可学习 Gaussian 和 IDWT 均参与计算。随机特征上的高频衰减证明该计算链路在起作用；它不能判断被削弱的高频究竟属于背景还是小目标，也不证明真实数据上的误检会降低。

相关代码：[小波变换](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/lfp.py:33)、[净化计算](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/lfp.py:149)。

### SFS

以下 query/key 组合均通过前向、反向、有限数值和连续内存检查：

- `[2,256,32,32]` / `[2,256,16,16]`。
- `[2,256,31,29]` / `[2,256,16,15]`。
- `[1,256,256,256]` / `[1,256,128,128]`。

query、key 及 SFS 的全部 21 个可训练参数张量都有非零、有限梯度，包括 `shared_offsets_residual`、attention weights、value/output projection。

在本次参数映射对照样例中，适配版相对原版的输出、query 梯度、key 梯度和 offset residual 梯度最大绝对误差均为 **0.0**。这支持当前实现保留了原版螺旋采样计算语义。测试覆盖的是明确列出的有限样例，不代表所有设备和精度下的逐位一致。

相关代码：[采样与输出](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/sfs.py:177)。

### B0–B4 AuxFPN

- 加载真实继承后的配置，使用 backbone 输入通道 `[256,512,1024,2048]`、空间尺寸 `[32,16,8,4]`。
- batch=2、训练模式：五组配置均通过 forward/backward。
- 输出均为 `[2,256,32,32]` 和 `[2,256,16,16]`，数值有限、内存连续。
- B1/B2 的两个新增参数张量、B3 的 21 个、B4 的 23 个，均获得非零、有限梯度。
- 通过 forward hooks 验证 LFP/SFS 确实被调用，并核对了插入顺序。
- batch=1、eval 模式：五组配置均通过前向检查。

这部分验证范围是 neck。没有把合成特征测试计为完整检测器训练或真实数据评测。

## 4. 发现的问题

### P2：LFP 依赖未进入项目环境声明

`lfp.py` 需要 `pytorch_wavelets`，其运行也依赖 `PyWavelets`；项目的 runtime requirements 和现有安装说明均未声明这两项。按原安装步骤准备的新环境无法直接构建 B1、B2、B4。

本次临时 Python 3.12 环境中还复现了 `pytorch-wavelets==1.3.0` 导入 `pkg_resources` 失败；安装 `setuptools==80.9.0` 后解决。现有 Python 3.9 服务器环境不应因此被判定为已损坏，但迁移环境时需要记录这个兼容条件。

建议增加 NS-FPN 可选依赖清单或补充安装说明，至少列明 `pytorch-wavelets`、`PyWavelets`，并记录已验证的版本组合。出处：[依赖清单](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/requirements/runtime.txt:1)、[LFP 导入](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/lfp.py:24)。

### P2：模块配置与层级开关没有一致性校验

已用实际 AuxFPN 复现：

| 不完整配置 | 当前行为 |
| --- | --- |
| `lfp_levels=(0,)`，但 `lfp_cfg=None` | 模块字典为空，前向成功，LFP 被静默跳过 |
| 只写 `lfp_cfg={}`，保留空的 `lfp_levels` | 模块未创建，前向成功 |
| `sfs_fusions=(0,)`，但 `sfs_cfg=None` | 构造成功，前向抛出 `KeyError: '0'` |
| 只写 `sfs_cfg={}`，保留空的 `sfs_fusions` | 模块未创建，前向成功 |

其中静默跳过尤其容易污染消融实验：配置名声称启用了模块，实际仍运行基线分支。建议在构造阶段验证每组 cfg/levels 是否成对有效，并对不完整组合报出清楚的 `ValueError`。

现有 B1–B4 均同时提供了所需 cfg 和非空 levels/fusions，本次检查没有发现这些配置被静默忽略。

出处：[模块创建条件](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/aux_fpn.py:221)、[LFP 跳过条件](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/aux_fpn.py:252)、[SFS 字典访问](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/mmdet/models/necks/aux_fpn.py:306)。复现记录：[additional_checks.json](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/nsfpn_audit/additional_checks.json)。

## 5. 已有检测效果证据

以下数值直接读取工作区已有 CSV 的 `scope=overall` 行，样本数均为 2000，GT 数均为 4884。本轮没有重新训练或重新运行这些 checkpoint。

| 配置/来源 | AP50 | Recall |
| --- | --- | --- |
| [B0](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/final_analysis/B0_domain_metrics.csv) | 0.772 | 0.872 |
| [B1](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/final_analysis/B1_domain_metrics.csv) | 0.766 | 0.866 |
| [B2](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/final_analysis/B2_domain_metrics.csv) | 0.768 | 0.862 |
| [B3](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/final_analysis/B3_domain_metrics.csv) | 0.761 | 0.858 |
| [B4](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/final_analysis/B4_domain_metrics.csv) | 0.763 | 0.862 |

当前 CSV 中，B4 相比 B0 的 AP50 低 0.9 个百分点，Recall 低 1.0 个百分点。FP 从 35831 降至 34363，同时 FN 从 625 升至 674；单独看 FP 数下降不足以证明整体收益。这些检测框计数也不能直接当作 NS-FPN 分割论文中的像素级虚警率 Fa。

工作区旧对话记录中的部分指标与当前 CSV 不完全一致，本报告以当前 CSV 为准。CSV 未携带完整运行命令和 checkpoint 哈希，最终论文用表仍需要与对应评测日志核对。

## 6. 复现入口

自检脚本：[verify_nsfpn.py](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/analysis_tools/verify_nsfpn.py)。脚本的两种后端是显式选择，不会自动把 CUDA 失败伪装成 CPU 通过。

本机已执行：

```bash
MPLCONFIGDIR=/private/tmp/auxdet-nsfpn-mpl \
/private/tmp/auxdet-nsfpn-audit-env/bin/python -B \
  '/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/analysis_tools/verify_nsfpn.py' \
  --backend reference --large \
  --json '/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/nsfpn_audit/reference_checks.json'
```

脚本还提供服务器验收入口。将该脚本同步到已有服务器项目后，可在原 `auxdet118` 环境运行：

```bash
conda run -n auxdet118 python \
  /home/ices/cjh/AuxDet/tools/analysis_tools/verify_nsfpn.py \
  --backend cuda --large --full-model \
  --json /home/ices/cjh/AuxDet/results/nsfpn_audit/cuda_checks.json
```

`--full-model` 使用 B4、合成图像和合成 GT，检查完整检测器的 loss、backward、predict；关闭 backbone 的预训练下载。此入口在本轮仅完成源码和语法检查，尚未在 CUDA 环境执行。通过它也只说明执行链路可运行，检测精度仍由真实数据评测决定。
