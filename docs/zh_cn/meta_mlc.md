# AuxDet Meta-MLC：实现、验证与实验用法

## 1. 本轮范围与仓库状态

正式项目是 `/Users/aaash/Documents/2026夏/AUXDET/AUXDET`。父目录还包含只读参考 `ALCNet-reference/`、`NS-FPN/`。

2026-09-30 实际检查到的分支是 `feature/meta-lfp`，起点提交是 `a95371c145e38e9a673d47dd94e2f4b564747118`。这与用户预期的 `feature/meta-mlc` 不同。代码直接编辑在当前工作区，未创建、切换分支或 worktree，未执行任何 Git 写操作。提交前请由用户确认目标分支和起点。

本轮新增 P2 的 MLC 残差分支、三种尺度融合模式、配置、测试和导出工具。现有 Meta-LFP、B0–B4、检测头、损失和历史实验输出保留。没有执行完整训练、下载数据集或权重。

## 2. 论文与官方参考核查

论文：[Attentional Local Contrast Networks for Infrared Small Target Detection](https://arxiv.org/abs/2012.08573)，核对第 III 节。

官方仓库：[YimianDai/open-alcnet](https://github.com/YimianDai/open-alcnet)。本地只读参考提交：

```text
5f71be58e2b703505005a75edbdf780647540fee
```

引用文件是父目录中的 `ALCNet-reference/model/contrast.py`，重点为 `circ_shift`、`cal_pcm`、`MPCMNet`、`CalMPCM.hybrid_forward`、`MPCMResNetFPN.cal_mpcm` 的尺度融合分支；另外阅读了 `model/segmentation.py` 和 `README.md`。没有安装 MXNet，也没有复制参考仓库或其 `.git`。

### 原算法对应关系

对每个通道、位置、邻域距离 d，四组相反方向为：

```text
(-d,-d), (-d,0), (-d,d), (0,-d)
```

方向响应：

```text
P_q(F) = (F - Shift_q(F)) * (F - Shift_-q(F))
D_d(F) = min_q P_q(F)
MLC(F) = max_d D_d(F)
```

- **方向差分**：中心与八邻域点逐元素相减。
- **方向聚合**：四个相反方向差分乘积取 **min**；保留有符号响应，没有额外 ReLU、abs 或方向门控。
- **尺度聚合**：原版模式逐通道、逐位置取不同 d 的 **max**。
- **官方边界**：通过切片和拼接进行循环平移，图像相反边缘会互相作为邻域。

论文公式 (1)/(2) 和官方 `cal_pcm` 使用方向 min；公式 (4) 印成方向 max，两者存在矛盾。本实现采用公式 (1)/(2) 和可执行官方代码。公式 (5) 的尺度最大池化与官方实现一致。不能把方向 min 与尺度 max 交换。

参考仓库有多种候选距离：例如 `MPCMNet` / `CalMPCM.hybrid_forward` 使用 9、13、17，`MPCMResNetFPN` 的同层尺度 max 分支使用 13、17。论文 DLC-FPN 实验使用 d=13。这些距离与官方下采样方案有关，不能直接认为它们是 AuxDet P2 的最优距离。

### AuxDet 的明确适配

本次三份配置统一采用 d=(1,2,3)，P2 stride=4，对应输入图像上约 4、8、12 像素的中心到邻域距离；最大跨度约 24 像素。这是适配 P2 的第一轮候选，尚未通过训练选优。

三份配置统一将循环边界改为 **replicate**：邻域坐标超出特征图时钳制到本图边界。这避免左边缘与右边缘、顶部与底部相互比较；batch 维从不移动。边缘 DLC 响应可能因重复边界值而减弱，应在实验中关注边缘目标。

因此 `original` 配置准确含义是“ALCNet DLC 方向算法与原版 MLC 尺度 max 的 AuxDet 迁移对照”，包含共同的 P2 距离、边界和残差封装适配。它不是完整 ALCNet 网络的原样复现。测试验证官方函数在无边界区域与本移植一致，另外独立核查适配后的边界。

## 3. 当前 AuxFPN 特征流

```text
backbone C2/C3/C4/C5
  → lateral conv
  → M2DM（P2/P3）：正确元数据 + 当前层视觉语义补偿
  → LEEM（P2/P3）
  → LEEM 外层残差融合
  → 可选 Meta-MLC（仅 laterals[0]，即 P2）
  → FPN 自顶向下融合
  → fpn conv
  → P2/P3 检测头
```

MLC 固定插入在 LEEM 外层残差之后、FPN 自顶向下之前，保持 `[B,256,H,W]`。当前模型最终只返回 P2/P3 给检测头。

三份新配置全部明确关闭 `lfp_cfg`、`lfp_levels`、`sfs_cfg`、`sfs_fusions`、`lfp_gate_cfg`。构建测试检查相应 ModuleDict 为空，相关模块没有被创建或执行。若同时请求 MLC 与 LFP/SFS/Meta-LFP，会在构造时明确报错。第一轮仅支持 `level=0` 且 `start_level=0`。

### 纯元数据真实来源

`MetaFeatureProcessorWithSem(channel_outs=128)` 已经为 M2DM 编码：

| 分支 | 原始输入 | 当前输出维度 |
|---|---|---:|
| view | Air / Space / Land 三维 one-hot | 32 |
| band_type | LWIR / NIR / SWIR 三维 one-hot | 32 |
| image size | width/2048、height/2048、width/height，再取 sin/cos | 32 |
| 视觉语义补偿 | 当前层下采样特征与深层特征之差，经 GAP | 32 |

新的兼容接口 `return_metadata_features=True` 返回已经计算的 `view_feat`、`band_feat`、`size_feat`，拼接成 **z_m=[B,96]**。它位于 `aux_fusion_mlp` 与语义补偿融合之前，不包含 `sem_feat`。默认接口仍返回 **all_aux_fea=[B,128]**，原参数名称和路径不变。

只在 P2 的 `visual_metadata` 模式请求额外返回值；不重新执行编码器。M2DM 仍按原有 P2/P3 两次循环编码，新模块不增加带 BatchNorm 编码器的调用或统计更新。

**all_aux_fea 是辅助融合特征，包含视觉补偿，不能称为纯元数据。** Meta-LFP 的 visual_aux 使用它；本次 Meta-MLC 使用另行取出的纯 view/band/size 中间编码。

缺失或未知 view/band_type 采用全零 one-hot，之后仍通过原来的可学习 Linear/BN，编码结果不必为零。尺寸优先使用已有 width/height，缺失时使用 ori_shape，随后使用 img_shape；没有可用的正数有限尺寸时明确报错。没有 GT 尺度、位置或其他推理时不可获得的信息。

编码器沿用原来的 BatchNorm1d；真实 AuxFPN 的 batch=1 验证在 eval 模式完成。训练时每个设备上的 batch=1 仍可能触发原有 BN 限制，本次不改变该算法。

## 4. 三种融合模式

### original

```text
C = max_k D_k(F)
```

逐通道 max 的胜出尺度可能不同，因此没有 `[B,K,H,W]` 标量 softmax 权重。统计中 `fusion='max'`，`mean`、`entropy` 为 null，不展示均匀权重占位值。

### visual

```text
V = SiLU(Conv1x1(AvgPool3x3_replicate(F)))  # [B,64,H,W]
A = VisualLogit(V)                        # [B,3,H,W]
alpha = softmax(A, dim=scale)
C = sum_k alpha_k * D_k(F)
```

仅新 MLC 尺度选择分支不依赖元数据；原 M2DM、LEEM 仍使用正确元数据。

### visual_metadata（主配置）

采用一个纯元数据线性投影生成 FiLM 条件与尺度偏置：

```text
gamma, beta, bias = Linear(z_m)       # 64 + 64 + 3 = 131 channels
Interaction = VisualLogit_without_bias(
                  V * tanh(gamma) + tanh(beta))
A = VisualLogit(V) + bias + Interaction
alpha = softmax(A, dim=scale)
C = sum_k alpha_k * D_k(F)
```

gamma、beta 按图像广播到空间位置；交互项乘上局部 V，所以同一张图内的 logits/alpha 可以随位置改变。没有新增方向门控、动态卷积、频域模块、BatchNorm 或 Dropout。

### 共同残差封装与稳定性

三种模式均使用：

```text
R = Conv1x1(GroupNorm(C))
lambda = softplus(lambda_raw)
Y = F + lambda * R
```

`lambda_init=0.1`，`lambda_raw=log(expm1(0.1))`，训练中范围为 `(0,+∞)`，不强行限制为 `[0,1]`。小正值使初期修正较弱，同时投影和条件分支都能获得梯度。归一化放在投影之前，避免直接将平方量级的差分乘积加入主特征。

FP16/BF16 DLC 差分乘积升到 FP32，softmax 使用 FP32；归一化之后再转回输入特征 dtype 进入投影，返回值兼容主特征。没有主路径 detach 或原地修改。随机初始输入及放大 1000 倍的测试均无 NaN/Inf；这不保证任意训练参数或无限幅值输入都稳定。

元数据条件投影的权重、bias 初始为零，所以正式初始化后 `visual_metadata` 的新增元数据条件暂时不起作用；相同视觉分支权重时它等于 visual 的尺度权重。条件投影本身首步有梯度，z_m 在这条新增条件路径的首步梯度为零，后续条件投影更新后可影响权重。测试用显式非零权重验证条件效果及梯度，这些测试权重没有进入正式配置。

当前 AuxFPN `init_cfg` 只对 Conv2d 做 Xavier 初始化，不覆盖条件 Linear 的零初始化和 lambda。真实 `AuxFPN.init_weights()` 已测试。Runner 后续加载 checkpoint 会覆盖匹配的已训练键；本模块没有在 forward 中重置参数。

## 5. 配置与参数量

新配置都继承 `configs/auxdet/auxdet_r50_fpn_1x_voc.py`，不默认指定 load_from 或 resume；保留原始 AuxDet 的 torchvision ResNet50 基础初始化。

| 配置文件（configs/auxdet/） | 尺度融合 | work_dir |
|---|---|---|
| auxdet_r50_mlc_p2_original_1x_voc.py | 尺度 max | work_dirs/auxdet_r50_mlc_p2_original_1x_voc |
| auxdet_r50_mlc_p2_visual_1x_voc.py | 视觉 softmax | work_dirs/auxdet_r50_mlc_p2_visual_1x_voc |
| auxdet_r50_mlc_p2_visual_metadata_1x_voc.py | 视觉 + 纯元数据 FiLM/偏置 softmax | work_dirs/auxdet_r50_mlc_p2_visual_metadata_1x_voc |

测试比较三份配置与基线的训练/验证/测试数据加载配置、优化器、日程和 train_cfg，均相同。hidden_dim=64、dilations=(1,2,3)、lambda_init=0.1 相同。三份配置的模块封装和位置完全共享。

实际统计口径：**sum(p.numel() for p in module.parameters())**，包含可学习参数，不包含 buffer/激活/计算量。当前 B0 neck 为 6,686,965 参数。

| 模式 | 共同残差封装 | 视觉描述/尺度头 | 纯元数据条件投影 | 新增合计 | neck 总参数 |
|---|---:|---:|---:|---:|---:|
| original | 66,049 | 0 | 0 | 66,049 | 6,753,014 |
| visual | 66,049 | 16,643 | 0 | 82,692 | 6,769,657 |
| visual_metadata | 66,049 | 16,643 | 12,707 | 95,399 | 6,782,364 |

共同封装：256×256 无 bias 投影=65,536；GN weight/bias=512；lambda_raw=1。

视觉分支：256→64 的 1×1 Conv+bias=16,448；64→3 的尺度头=195。

条件分支：96→131 Linear+bias=12,707。每图条件投影约 12,576 次乘加；另有空间 FiLM 与共享尺度头的交互计算。DLC/归一化/投影的主要开销由三种模式共享。

visual 没有创建无用的元数据分支；visual_metadata 所有条件参数参与计算和训练。不能宣称二者参数量完全相同。已有元数据编码器是基线已经具备的成本，本表不将它再次算作新增成本。完整检测器总参数尚未在本地构建；服务器可用 `--backend native --full-model` 报告该口径。

## 6. 服务器训练与测试

在服务器激活原 AuxDet 环境，进入真正的内层仓库。下面路径 `/path/to/AUXDET` 请换成服务器的实际 Git 仓库根目录，使用已有数据配置。

```bash
export AUXDET_REPO=/path/to/AUXDET
cd "$AUXDET_REPO"

# 先做真实环境的构建/小尺寸 neck 检查；不训练、不下载权重。
python tools/analysis_tools/verify_meta_mlc.py --backend native --full-model \
  --json results/meta_mlc_server_validation.json

# 三个独立实验，使用相同随机种子与原配置日程。
python tools/train.py configs/auxdet/auxdet_r50_mlc_p2_original_1x_voc.py \
  --cfg-options randomness.seed=20260930
python tools/train.py configs/auxdet/auxdet_r50_mlc_p2_visual_1x_voc.py \
  --cfg-options randomness.seed=20260930
python tools/train.py configs/auxdet/auxdet_r50_mlc_p2_visual_metadata_1x_voc.py \
  --cfg-options randomness.seed=20260930
```

多 GPU 使用项目已有入口，例如：

```bash
bash tools/dist_train.sh configs/auxdet/auxdet_r50_mlc_p2_visual_metadata_1x_voc.py 2 \
  --cfg-options randomness.seed=20260930
```

保持与原始 AuxDet 相同的 GPU 数、每 GPU batch、学习率及数据划分，不擅自打开 auto-scale-lr。已有 work_dir 上重复运行会新增/修改训练文件，正式新轮实验请指定新的 `--work-dir`。

测试每个实验自身训练的 checkpoint：

```bash
python tools/test.py configs/auxdet/auxdet_r50_mlc_p2_original_1x_voc.py \
  work_dirs/auxdet_r50_mlc_p2_original_1x_voc/epoch_12.pth \
  --work-dir work_dirs/auxdet_r50_mlc_p2_original_1x_voc_eval_run1 \
  --out results/mlc_original_run1_predictions.pkl
python tools/test.py configs/auxdet/auxdet_r50_mlc_p2_visual_1x_voc.py \
  work_dirs/auxdet_r50_mlc_p2_visual_1x_voc/epoch_12.pth \
  --work-dir work_dirs/auxdet_r50_mlc_p2_visual_1x_voc_eval_run1 \
  --out results/mlc_visual_run1_predictions.pkl
python tools/test.py configs/auxdet/auxdet_r50_mlc_p2_visual_metadata_1x_voc.py \
  work_dirs/auxdet_r50_mlc_p2_visual_metadata_1x_voc/epoch_12.pth \
  --work-dir work_dirs/auxdet_r50_mlc_p2_visual_metadata_1x_voc_eval_run1 \
  --out results/mlc_visual_metadata_run1_predictions.pkl
```

`epoch_12.pth` 是 1x 日程的示例名称，请以实际生成的 checkpoint 为准。测试不会在本地代为启动。

### Checkpoint 初始化与继续训练

- 默认三份配置与原始 AuxDet 使用相同的 backbone 预训练初始化；新 MLC 参数正常随机/零初始化。不默认加载 Meta-LFP 训练权重。
- 若研究设计要求从同一份 AuxDet B0 权重开始，应对所有实验统一指定同一 checkpoint，使用 `--cfg-options load_from=/path/to/b0.pth`，并在记录中说明这改变了从 backbone 初始化的实验方案。
- 旧 B0 checkpoint 加载到新配置时，新增 `neck.mlc_modules.0.*` 键缺失是预期。共享原参数键没有改名，测试验证只有新分支键缺失。
- 不同融合模式的 MLC checkpoint 不应互相作为公平初始化；模式特有键的缺失/多余需要单独检查。
- 同配置恢复训练使用项目原 `--resume`；恢复的是该实验已经训练的条件参数和 lambda，不应重新置零。

## 7. 尺度统计与解释接口

默认关闭统计，训练期间不打印、不复制到 CPU。需要时设 `mlc_cfg.collect_stats=True`；`neck.get_mlc_scale_stats()` 按层返回本批次逐图像统计：img_id（无 ID 时回退 img_path）、view、band_type、候选距离、空间平均尺度权重、平均熵。熵用自然对数，三个尺度的最大值为 ln(3)。

`store_weight_maps=True` 可保留最新批次的 detached CPU `[B,K,H,W]` 权重图。原版 max 模式没有这种标量权重图。

直接按正常测试数据流程导出（单进程，保持模型 eval）：

```bash
python tools/analysis_tools/export_mlc_scales.py \
  configs/auxdet/auxdet_r50_mlc_p2_visual_metadata_1x_voc.py \
  work_dirs/auxdet_r50_mlc_p2_visual_metadata_1x_voc/epoch_12.pth \
  --output-dir results/mlc_visual_metadata_scales_run1

# 为指定图像 ID 额外导出特征分辨率的尺度图；ID 应匹配数据样本 img_id。
python tools/analysis_tools/export_mlc_scales.py \
  configs/auxdet/auxdet_r50_mlc_p2_visual_metadata_1x_voc.py \
  work_dirs/auxdet_r50_mlc_p2_visual_metadata_1x_voc/epoch_12.pth \
  --output-dir results/mlc_visual_metadata_maps_run1 --map-ids 12 37
```

导出目录必须不存在，以免覆盖旧记录。产生 `scale_stats.jsonl`、`resolved_config.py`，以及选中图像的 `.pt` 文件。`.pt` 中包含 img_id、dilations 和 `[K,H,W]` 权重；图在 P2 特征坐标中，未自动插值到原图。测试日志另存为输出目录同级的 `*_runner` 目录。可用 `--cfg-options` 传入与正常测试相同的数据路径覆盖选项。

导出 Hook 的序列化、选图及不覆盖已有目录已做模拟 runner 单元测试；本地没有数据集/检测器扩展，尚未执行完整服务器数据加载和检测导出。该单元测试不能当作实际数据端到端通过。

## 8. 验证结果与复现

本地环境：Python 3.12、PyTorch 2.2.2、mmengine 0.10.7、mmcv-lite 2.1.0，CPU。

验证命令：

```bash
python -m pytest -o addopts='' -q \
  tests/test_tools/test_meta_mlc.py tests/test_tools/test_meta_lfp_gate.py
python tools/analysis_tools/verify_meta_mlc.py --backend source \
  --baseline-ref a95371c145e38e9a673d47dd94e2f4b564747118 \
  --json results/meta_mlc_recheck.json
python tools/analysis_tools/verify_nsfpn.py --backend reference \
  --json results/nsfpn_after_mlc_recheck.json
```

本轮已运行结果：

- 新 Meta-MLC 与既有 Meta-LFP 联合测试 **25 passed**。
- DLC 独立坐标 oracle、真实官方 `circ_shift/cal_pcm` Python 源码的内部区域对比通过；没有运行 MXNet runtime。
- batch=1/3、非方形纯模块 forward/backward、方向/尺度聚合、边界无跨边缘循环、无 batch 混合通过。
- softmax 和为 1；逐图统计及 detached maps 通过。
- 固定视觉输入只改新分支元数据时，非零测试条件可以改变尺度权重；visual 不依赖新增元数据输入；梯度验证通过。
- 实际 AuxFPN 源码、实际 M2DM/LEEM/MLC 的 CPU 前向通过：batch=1/2（eval），P2 为 32×24，P3 为 16×12；batch=2 train 模式的反向及编码调用次数检查通过。
- 实际 `AuxFPN.init_weights()`、初始零条件、lambda=0.1 和 checkpoint 键兼容性通过。
- 默认关闭新分支时，与上面起点提交同权重 B0 的输出最大差为 **P2=0，P3=0**。
- B0–B4 配置构造与 reference 检查通过；原 `lfp`、`sfs`、`original_sfs_equivalence`、`b0_b4_necks` 均 PASS。
- 混合精度 CPU autocast 和 FP16 大幅值差分乘积、初始 1000 倍特征压力输入无 NaN/Inf。
- 三份配置的共同日程/数据/优化器、新分支参数量、其他可选模块关闭通过。
- 导出接口单测通过；代码编译和 Git diff 空白检查通过。

保留证据于 `results/meta_mlc_validation_20260930/`，包含 pytest XML、真实源码 neck 检查 JSON、旧模块检查 JSON、源码哈希和环境记录。

### 明确的受限项

普通完整模型包导入因缺少 `mmcv._ext` 失败；`--backend native --full-model` 的注册入口首先因缺少 `pycocotools` 失败。因此尚未验证完整检测器构建、CascadeRPN/RoI 的真实算子、CUDA/MPS forward、服务器 DDP、真实 checkpoint 文件、数据集训练/评测及端到端导出。

source/reference 入口绕过了检测器包聚合；旧 SFS 验证使用 MMCV 自带 PyTorch 注意力参考，不能声称真实 CUDA SFS 已通过。新 MLC 配置中 SFS 完全关闭，所以新 neck 前向没有执行 SFS 替身，也没有替换新 DLC、MLC、M2DM、LEEM 的计算。

最初 pytest 被缺少 xdoctest 插件阻断；`-o addopts=''` 仅取消项目配置中的额外 doctest 参数，没有跳过上述测试。

## 9. 建议的实验矩阵与结论边界

| 实验 | 增强方式 | 用于回答的问题 |
|---|---|---|
| 原始 AuxDet / B0 | M2DM + LEEM | 基础检测性能 |
| 现有 B2 | P2、after_edge 直接 LFP | 当前频域增强基线 |
| 现有 Meta-LFP visual / visual_aux | LFP 外部残差系数 | LFP 使用强度的条件来源 |
| original MLC | 相反方向 min、跨尺度 max | 本次局部对比迁移效果 |
| visual MLC | 局部视觉 softmax 尺度选择 | 可学习尺度选择效果 |
| visual_metadata MLC | 视觉 + 纯元数据 FiLM 尺度选择 | 元数据参与内部尺度融合的效果 |

第一轮使用相同数据、初始化方案、随机种子和训练预算。后续建议多种种子、按 view/band 分组评价，并结合已有离线评测工具检查召回、误检和小目标指标。

visual 和 visual_metadata 共享视觉网络，但后者多 12,707 条件参数；性能差异不能独立排除参数容量的影响。若需要进一步控制容量，可后续增加同等规模的视觉条件分支实验，本轮不通过 metadata 置零声称有效参数量完全相同。

如果做元数据扰动，只替换 **新 Meta-MLC 条件路径的 z_m**，保留原 M2DM、LEEM 正确输入；不要整体修改样本元数据，否则会同时改变基线路径。比较时固定 F，避免将上游特征变化误归因于内部尺度选择。已有 Meta-LFP visual_aux 使用含视觉补偿的 all_aux_fea，它与 visual 的差异不能单独证明纯元数据收益。

**元数据使尺度权重发生变化，只能证明条件路径有效，不能证明检测性能提升。本轮没有任何完整训练证据或性能提升结论。**

## 10. 修改文件与建议提交划分

正式新增/修改：

- `mmdet/models/necks/mlc.py`：DLC、三模式融合、FiLM 条件、残差与统计。
- `mmdet/models/necks/aux_fpn.py`：纯元数据可选返回、P2 接入、配置校验、图像统计和缺失元数据策略。
- `mmdet/models/necks/__init__.py`：导出新类/运算。
- 三份 `configs/auxdet/auxdet_r50_mlc_p2_*_1x_voc.py`：独立实验配置。
- `tests/test_tools/test_meta_mlc.py`：公式、条件、梯度、初始化、配置、真实 neck 源码及导出接口测试。
- `tools/analysis_tools/verify_meta_mlc.py`：可重复的小尺寸 neck 验证与可选历史 B0 比较。
- `tools/analysis_tools/export_mlc_scales.py`：正常测试循环的统计/权重图导出。
- `tools/analysis_tools/verify_nsfpn.py`：同一 pytest 进程中复用源码加载结果，避免重复注册模型类；算法不变。
- 本说明、README 入口与独立验证证据目录。

历史本地引用 cb97de2、8d82f33、b093858、43611fc 均可读。cb97de2 的可选返回元数据接口思路被保留，并扩展返回 size 分支；本次通过普通文件编辑实现。其他引用的 ring contrast 是邻域环形均值对比及外部残差/门控，不是 ALCNet 相反方向差分乘积，未迁移其算法。

建议由用户自行审查并分成：

1. `feat(neck): add metadata-conditioned local contrast scale fusion`：模块、AuxFPN 接口/接入、三配置。
2. `test(meta-mlc): add verification and scale export tools`：新测试、检查/导出脚本和重复注册修复；测试和所依赖的导出工具一起提交。
3. `docs(meta-mlc): record experiment protocol and validation`：说明、README、验证证据。

所有 Git 操作均由用户决定和执行。先检查实际分支，然后按明确文件列表暂存；不要在父目录或空的 AUXDET_REPO 变量下操作。服务器同步到同一提交后先做 native 检查，再启动训练。

## 11. 由用户执行的本地与服务器 Git 示例

以下仅供复制参考，Codex 未执行。先由用户处理当前 `feature/meta-lfp` 与预期分支的差异。示例要求本地最终位于 `feature/meta-mlc`，使用子 shell 避免检查失败影响当前终端。

本地审查及按三组提交：

```bash
(
  AUXDET_REPO='/Users/aaash/Documents/2026夏/AUXDET/AUXDET'
  cd "$AUXDET_REPO" || exit 1
  git rev-parse --show-toplevel
  git status --short --branch
  git diff --check
  test "$(git branch --show-current)" = 'feature/meta-mlc' || {
    echo '请先自行确认并处理目标分支；本示例停止。'
    exit 1
  }

  git add mmdet/models/necks/mlc.py mmdet/models/necks/aux_fpn.py \
    mmdet/models/necks/__init__.py \
    configs/auxdet/auxdet_r50_mlc_p2_original_1x_voc.py \
    configs/auxdet/auxdet_r50_mlc_p2_visual_1x_voc.py \
    configs/auxdet/auxdet_r50_mlc_p2_visual_metadata_1x_voc.py || exit 1
  git diff --cached --stat
  git commit -m 'feat(neck): add metadata-conditioned local contrast scale fusion' || exit 1

  # 新测试引用导出工具，因此将二者放在同一组，避免中间提交缺少依赖。
  git add tests/test_tools/test_meta_mlc.py \
    tools/analysis_tools/verify_meta_mlc.py \
    tools/analysis_tools/verify_nsfpn.py \
    tools/analysis_tools/export_mlc_scales.py || exit 1
  git diff --cached --stat
  git commit -m 'test(meta-mlc): add verification and scale export tools' || exit 1

  git add docs/zh_cn/meta_mlc.md README.md results/meta_mlc_validation_20260930 || exit 1
  git diff --cached --stat
  git commit -m 'docs(meta-mlc): record experiment protocol and validation' || exit 1

  git status --short --branch
  git log -3 --oneline
  # 审查完毕后，由用户执行同步：
  # git push -u origin feature/meta-mlc
)
```

若已有其他暂存内容，提交前应自行检查完整 `git diff --cached`，本示例不会清空或调整既有暂存。也可以选择一次提交或按其他分组提交。

服务器已有目标分支、且工作区干净时，可由用户同步：

```bash
(
  AUXDET_REPO='/path/to/AUXDET'
  cd "$AUXDET_REPO" || exit 1
  git rev-parse --show-toplevel
  git status --short --branch
  test "$(git branch --show-current)" = 'feature/meta-mlc' || {
    echo '请先自行准备服务器目标分支。'
    exit 1
  }
  test -z "$(git status --porcelain)" || {
    echo '服务器存在本地修改，请自行处理后再同步。'
    exit 1
  }
  git pull --ff-only origin feature/meta-mlc || exit 1
  git rev-parse HEAD
  # 与本地最终提交编号比对，再做本说明第 6 节的 native 验证和训练。
)
```

`--ff-only` 若失败，请自行处理分支差异；不要直接 reset 或覆盖服务器改动。若服务器没有该分支，请自行按既有 Git 工作流准备，Codex 没有代为 fetch、切换或合并。
