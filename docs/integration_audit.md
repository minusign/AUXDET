# AuxDet + PConv / HS-FPN / NS-FPN 源码集成审计

审计日期：2026-10-16（Asia/Shanghai）  
审计范围：`/Users/aaash/Documents/2026夏/AUXDET` 工作区；只读检查源码、配置、Git 元数据和本地依赖可见性。未训练、未安装依赖、未修改模型/配置/AGENTS.md、未改变分支/提交/远端。

## 0. 结论先行

当前工作区中，能够证明为“真正 AuxDet 基线”的完整配置只有 `AUXDET/configs/auxdet/auxdet_r50_fpn_1x_voc.py`。它继承 Faster R-CNN 的 detector/head 基础配置、`voc_aux.py` 数据配置、12 epoch 的 `schedule_1x.py` 和 `default_runtime.py`；其 `AuxFPN` 的 forward 明确接收 `batch_data_samples`，因此包含元数据驱动分支。普通 Faster R-CNN/FPN 配置不满足该基线条件。

六种实验理论上可以继续共用 `tools/train.py`/`tools/test.py`，因为入口按配置构建 Runner；但在实现阶段至少要解决四个阻塞项：

1. HS-FPN 代码采用旧版 MMDetection 注册和 MMCV API，且未注册为当前 AuxDet 的 `MODELS` neck。
2. NS-FPN 的 `forward(inputs)` 不接收元数据，而 AuxDet 当前 `TwoStageDetector.extract_feat` 无条件调用 `self.neck(x, batch_data_samples)`；NS-FPN 还依赖 `pytorch_wavelets` 和需编译的 CUDA/C++ MSDeformAttn。
3. AuxDet 当前 neck 返回 `tuple(outs[:2])`，与其 Cascade RPN/ROI 的两层 stride `[4, 8]` 配置一致；HS/NS 参考实现通常返回四层，直接替换会与 anchor/RoI 层数不一致。
4. PConv 参考实现的构造接口是 `PConv(c1,c2,k,s)`，不是 MMCV `ConvModule`/ResNet block 的 `conv_cfg` 插件；并且它没有残差连接，替换 ResNet 前两个 stage 的卷积会改变初始化和 state_dict key。

上述判断中，“源码确认”指已在本地文件中看到；“配置推导”指由配置字段和调用关系得到；“尚未运行验证”指本次未构建模型、未跑 forward、未编译算子。

## 一、仓库和环境

### 1.1 仓库真实路径、来源、commit 和状态

| 仓库 | 真实路径 | 来源 remote | 当前 commit | 分支 | 工作树状态 |
|---|---|---|---|---|---|
| AuxDet | `/Users/aaash/Documents/2026夏/AUXDET/AUXDET` | `https://github.com/minusign/AUXDET.git` | `a9f5cafcfca006056e6faff925f7f8727e329a1c` | `feature/meta-mlc` | 无未提交修改 |
| HS-FPN | `/Users/aaash/Documents/2026夏/AUXDET/HS-FPN` | `https://github.com/ShiZican/HS-FPN.git` | `d7be73e7f1da9be543a3c6d9ab0d25561b94f9b3` | `main` | 无未提交修改 |
| NS-FPN | `/Users/aaash/Documents/2026夏/AUXDET/NS-FPN` | `https://github.com/mengduann/NS-FPN.git` | `b857bef068ba48f1258b62de6bf082f73dbafde4` | `main` | 有未跟踪 `.DS_Store` |
| PConv-SDloss-Data | `/Users/aaash/Documents/2026夏/AUXDET/PConv-SDloss-Data` | `https://github.com/JN-Yang/PConv-SDloss-Data.git` | `a801f043c83f73aa9af9ab2f689e59ebef928fc4` | `main` | 无未提交修改 |

本地辅助材料真实路径包括：`/Users/aaash/Documents/2026夏/AUXDET/AuxDet.pdf`、`PConv.pdf`、`HS-FPN.pdf`、`NS-FPN.pdf`、`10_07开题报告v1.1.pdf`；另有临时 Word 锁文件和 `.proposal_defense_build/` 生成物。论文/开题材料没有被当作源码证据。

许可证：AuxDet 根目录 `LICENSE` 是 OpenMMLab Apache License 2.0；HS-FPN `LICENSE` 是 ShiZican 2025 MIT；PConv-SDloss-Data `LICENSE` 是 JiangNan-Yang 2024 MIT；NS-FPN 仓库根目录未找到 `LICENSE`（“未找到”，不能据论文推断许可）。NS-FPN 的 `SFS_MSDeformAttn/ops` 同时包含预编译 `.so/.egg` 和 C++/CUDA 源码，需在迁入时保留其许可/归属核查。

### 1.2 框架、版本和算子

- AuxDet `requirements/mminstall.txt:1-2` 要求 `mmcv>=2.0.0rc4,<2.2.0`、`mmengine>=0.7.1,<1.0.0`；README 安装说明为 Python 3.9、PyTorch/torchvision/torchaudio CUDA 11.7、editable 安装。
- AuxDet `requirements/runtime.txt:1-8` 只列出运行时通用包；`setup.py` 使用 `torch.utils.cpp_extension` 的通用 helper，但在 AuxDet Git 文件中未找到 `.cpp`、`.cu`、`CMakeLists.txt` 或实际 `CUDAExtension` 模块。因此 AuxDet 本身未找到自定义 CUDA/C++ 算子（源码确认）。
- HS-FPN `hsfpn_requirements.txt` 是一份旧环境快照，包含 `mmcv-full==1.3.17`、`torch==1.7.1`、`torch-dct==0.1.6`、`einops==0.6.1` 等；其代码 import `mmcv.runner.BaseModule`、`mmcv.runner.auto_fp16` 和旧 `mmdet.models.builder.NECKS`。
- NS-FPN `environment.yml` 要求 Python 3.10、`torch==2.1.2`、`numpy=1.23.5`、`pytorch-wavelets==1.3.0`；源码另依赖 `einops`、本地 `SFS_MSDeformAttn` 扩展。
- 本地审计解释器为 Python 3.13.5；`torch`、`mmcv`、`mmengine`、`torch_dct`、`pytorch_wavelets`、`einops` 均未安装可见，导入本地 `mmdet` 因缺少 `mmcv` 失败。服务器 CUDA、GPU、驱动、PyTorch、MMCV/MMEngine、编译器版本均为**未知**，不能猜测。
- NS-FPN 的 `SFS_MSDeformAttn/ops/setup.py` 和 `src/{vision.cpp,cpu/*.cpp,cuda/*.cu,cuda/*.cuh}` 明确是自定义 C++/CUDA 算子；仅有预编译 x86_64 CPython 3.9 `.so/.egg`，不能假定可用于服务器。

同名包风险：当前源码树含 `AUXDET/mmdet/__init__.py`，若以 editable 方式安装或从仓库根运行，优先可能是本地 `mmdet`；环境若另有 pip 安装的 `mmdet/mmcv/mmengine`，则会出现路径/API 混用风险。当前本地 `mmdet` 不能完成 import，未运行 `pip show`，已安装发行版版本为**未知**；这项需在服务器用 `python -c 'import mmdet; print(mmdet.__file__)'` 明确。

### 1.3 AGENTS.md 约定

在本工作区及其父级可见范围未找到仓库内 `AGENTS.md`；`/Users/aaash/.codex/AGENTS.md` 只包含图像生成路由约定，与本次源码审计无冲突。本报告遵守用户给出的“只允许更新 `docs/integration_audit.md`”约束。

## 二、真正的 AuxDet 基线

### 2.1 配置及继承链

唯一合适的完整基线是：

`AUXDET/configs/auxdet/auxdet_r50_fpn_1x_voc.py:1-5`

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/voc_aux.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
```

选择依据是它同时声明 `neck=dict(type='AuxFPN', ...)`、Cascade RPN/ROI 的 `[4, 8]` 两层使用方式，并且 neck forward 读取 `DetDataSample` 的 view/band/size；普通 `faster-rcnn_r50_fpn.py` 不包含这些元数据路径，不能作为 AuxDet 基线。

配置实际覆盖：ResNet-50、`out_indices=(0,1,2,3)`、`frozen_stages=1`、`norm_eval=True`、torchvision ResNet-50 预训练；AuxFPN `in_channels=[256,512,1024,2048]`, `out_channels=256`, `num_outs=4`, `start_level=0`, `end_level=3`；CascadeRPN 两 stage；ROI `featmap_strides=[4,8]`；train batch size 4。

### 2.2 实际类型

| 部分 | 实际类型 | 证据 |
|---|---|---|
| detector | `FasterRCNN` → `TwoStageDetector` | `configs/_base_/models/faster-rcnn_r50_fpn.py:7-33`；`mmdet/models/detectors/faster_rcnn.py:7-28` |
| backbone | `ResNet(depth=50)` | `configs/auxdet/auxdet_r50_fpn_1x_voc.py:10-18` |
| neck | `AuxFPN` | `mmdet/models/necks/aux_fpn.py:17-113`，并在 `necks/__init__.py:21-28` 导出 |
| RPN head | `CascadeRPNHead(num_stages=2)` | 基线配置 `:25-76` |
| ROI head | 基础 Faster R-CNN 的 `StandardRoIHead`/`Shared2FCBBoxHead` 配置，覆盖 extractor/head 参数 | `configs/_base_/models/faster-rcnn_r50_fpn.py:33-55` 和基线 `:77-90` |
| dataset | `VSBWILDVOCDetDataset` | `mmdet/datasets/voc_aux.py:10-24`；配置 `voc_aux.py:1,56-67` |
| evaluator | `VOCMetric(metric='mAP', eval_mode='area')` | `configs/_base_/datasets/voc_aux.py:91-94` |

### 2.3 入口和命令行

训练入口是 `AUXDET/tools/train.py:12-48,55-115`，位置参数为 config；支持 `--work-dir`、`--amp`、`--auto-scale-lr`、可选 `--resume [checkpoint|auto]`、`--cfg-options key=value ...`、`--launcher {none,pytorch,slurm,mpi}` 和 local-rank。测试入口是 `AUXDET/tools/test.py:23-64,70-151`，位置参数为 config 与 checkpoint；支持 `--work-dir`、`--out`、`--show`、`--show-dir`、`--wait-time`、`--cfg-options`、`--launcher`、`--tta`。README 的实际示例为 `python tools/train.py configs/auxdet/auxdet_r50_fpn_1x_voc.py` 与 `python tools/test.py ... best.pth`。

## 三、模型数据流与接口

### 3.1 输入、元数据和调用顺序

1. `VSBWILDVOCDetDataset.load_data_list`（`mmdet/datasets/voc_aux.py:26-77`）读 XML，写入 `width/height/view/band_type/instances`。
2. `voc_aux.py:27-46` 的 `PackDetInputs.meta_keys` 保留上述字段。
3. `TwoStageDetector.extract_feat`（`mmdet/models/detectors/two_stage.py:115-128`）先 `x=self.backbone(batch_inputs)`，再无条件 `self.neck(x, batch_data_samples)`。
4. `ResNet.forward`（`mmdet/models/backbones/resnet.py:631-646`）按 `out_indices` 返回四个 stage，ResNet-50 的通道为 `[256,512,1024,2048]`，输入 1024 尺寸时 stride 推导为 `[4,8,16,32]`，顺序为浅到深。
5. `AuxFPN.forward`（`mmdet/models/necks/aux_fpn.py:158-240`）先建四个 lateral；前两个层级使用 `downsamples`、`MetaFeatureProcessorWithSem`、`DMLPAttention` 和 `SpatialModulation`；随后 `EdgeConvSep` 边缘增强、top-down 融合、3×3 FPN conv；最后 `return tuple(outs[:2])`（`:300-314`）。
6. `TwoStageDetector.loss/predict`（`:161-209`、`:211-258`）将 neck 输出同时传给 Cascade RPN 和 ROI head；训练中 RPN 生成 proposals，ROI head 计算最终损失；预测中同样先 RPN，再 ROI。

M2DM/LEEM：在当前 AuxDet源码中未找到名为 `M2DM` 或 `LEEM` 的类/文件。源码实际对应的是 AuxFPN 内的 `MetaFeatureProcessorWithSem`（`:318-389`，元数据+语义补偿）以及其 `EdgeConvSep`/动态调制路径。若项目论文将这些路径命名为 M2DM、LEEM，则这是命名映射而非源码中独立的 M2DM/LEEM 类，需实现阶段确认，不能把不存在的类写成已找到。

### 3.2 Neck/head 接口约束

- AuxFPN 构造与标准 FPN 相同的 `in_channels/out_channels/num_outs/start_level/end_level/add_extra_convs` 参数（`:38-113`），但内部额外固定创建两层 edge conv、两层 depthwise-like downsample（`groups=out_channels//8`）、两层 dynamic attention，并固定只返回前两层。
- 基线 RPN anchor generator 配置在 `auxdet_r50_fpn_1x_voc.py:38-45`：两层 stride `[4,8]`、每层一个 scale `[8]`、ratio `[1.0]`。ROI extractor `featmap_strides=[4,8]`（`:78-82`），因此 neck 输出必须至少按该顺序提供对应空间步长的两层。
- 额外输出规则：AuxFPN 的标准逻辑可 max-pool 或 extra conv 产生 `num_outs`（`:263-314`），但本基线最终截断为两层；这属于源码确认，不是标准 FPN 配置字段的推测。

训练/预测的元数据传递：数据 sample 由 PackDetInputs 放入 `batch_data_samples`，`TwoStageDetector.extract_feat` 将整个列表传入 neck；AuxFPN 的 `MetaFeatureProcessorWithSem.forward`（`:318-389`）直接读取 `sample.view`, `sample.band_type`, `sample.width`, `sample.height`。RPN/ROI 仍接收同一批 `batch_data_samples` 以读取图像尺寸和 GT。未运行验证：本地依赖缺失，尚未确认 PackDetInputs 在服务器版本中是否把自定义 key 保持为属性而非字典键。

## 四、PConv 接入条件

### 4.1 参考实现

准确位置：`/Users/aaash/Documents/2026夏/AUXDET/PConv-SDloss-Data/model/APConv.py:15-54`。类名 `PConv`，构造 `PConv(c1,c2,k,s)`。它直接依赖同文件 `Conv`（`:15-34`）和 PyTorch `torch/nn/functional`，不依赖 Partial Convolution，也不包含 SD Loss。

核心源码片段：

```python
p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
self.pad = [nn.ZeroPad2d(padding=p[g]) for g in range(4)]
self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
self.cat = Conv(c2, c2, 2, s=1, p=0)

def forward(self, x):
    yw0 = self.cw(self.pad[0](x)); yw1 = self.cw(self.pad[1](x))
    yh0 = self.ch(self.pad[2](x)); yh1 = self.ch(self.pad[3](x))
    return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
```

`Conv` 是 Conv2d(bias=False) → BatchNorm2d → 默认 SiLU（`:20-29`）；PConv 的 `c2` 必须能被 4 整除，`k`/`s` 由调用方传入，四个非对称 ZeroPad2d 后分别做 `(1,k)`/`(k,1)`，最后 2×2 Conv 将拼接结果压回 `c2`。没有显式残差。`APBottleneck`/`APC2f`（`:57-103`）是另一条 YOLO 模块路径，不应误认为最小 ResNet 替换。

### 4.2 AuxDet 候选和风险

ResNet-50 前两个 stage 的原始 3×3 卷积完整名称是 `ResNet.layer1.*.conv2`、`ResNet.layer2.*.conv2`，构造在 `mmdet/models/backbones/resnet.py:154-207` 的 `Bottleneck` 中；stride 对 PyTorch style 位于 `conv2`（`:154-159`、`:178-186`）。这是“适合讨论的最小替换候选”，但不是现成配置扩展点：每个 bottleneck 的 `conv2` 输入/输出通道分别为 64/64（stage1）、128/128（stage2），均满足 c2 可除以 4；stage 首块 stride=2，其余 stride=1。周边顺序是 `conv2 → norm2 → conv3 → norm3 → residual add → ReLU`（`:67-94`、`:178-208`）。

直接把 PConv 替换 `conv2` 会改变卷积核形状、BN 命名和初始化，且 PConv 自带 BN+SiLU，随后 ResNet 还会再接 `norm2`，造成双归一化/激活语义变化；若只替换卷积主体则需重写适配类，把 padding/BN/激活明确对齐。`conv_cfg` 不能直接构造该类，因为 ResNet 使用 `build_conv_layer` 且 PConv 参数名不同。最稳妥方向是实现阶段增加小型 `Bottleneck` 子类或显式 `pconv` 开关，并对仅前两个 stage 的层名、stride、state_dict 映射写清楚；当前源码没有可直接启用的配置扩展。

## 五、HS-FPN 与 NS-FPN 源码复用条件

### 5.1 HS-FPN

文件：`HS-FPN/hs_fpn.py`。完整实现类和位置：`DctSpatialInteraction:27-58`、`DctChannelInteraction:65-116`、`HFP:122-140`、`SDP:146-170`、`SDP_Improved:176-202`、`HS_FPN:209-369`。HFP 使用 `torch_dct.dct_2d/idct_2d`；p1/p2 `isdct=True` 做低频区域置零后的高频掩码，p3/p4 `isdct=False` 走 1×1/池化注意力（`:25-51`, `:88-110`）。HS_FPN 固定建立 `SelfAttn_p4/p3`、`SelfAttn_p2/p1` 和 `CrossAtten_p4_p3/p3_p2/p2_p1`（`:283-290`），forward 按 p4→p1 处理（`:312-325`），之后又进行一次标准 top-down 融合并返回所有 `outs`（`:327-369`）。

构造参数为标准 FPN 参数加 `ratio=(0.25,0.25)`；每层 lateral/3×3 conv 均以 `out_channels` 对齐，HFP 的 GroupNorm 固定 32 组，故 `out_channels` 必须适合 32 组（基线 256 可满足）。SDP 的 `patch_size` 由 forward 固定传 `[h,w]`，并通过 `einops.rearrange` 假定尺寸可整除；代码的 `self.fpn_upsample` 固定 scale factor 2，却在 SDP 调用中传 `[h,w]`，需核对输入尺寸后才能判断是否一致。

可直接迁入：DCT/HFP/SDP 纯 PyTorch 模块的算法主体。必须适配：`mmcv.runner.BaseModule/auto_fp16` → 当前 `mmengine.model.BaseModule`/当前 autocast；旧 `NECKS`（`:18-20`）→ 当前 `MODELS.register_module()`；forward 签名需允许 `meta_inf=None`，或在 detector 侧分支调用；只输出与 AuxDet head 兼容的前两层或同步修改 anchor/ROI。算法含义风险：p1/p2 高频掩码、固定 patch/scale、重复 top-down 不是普通 FPN 的等价替换；改变 p 层选择或返回层数会改变 HS-FPN 算法。

### 5.2 NS-FPN

文件：`NS-FPN/model/NS_FPN.py`。完整实现：`ConvDWT:25-47`、`ConvIDWT:49-65`、`LearnableGaussianFilterBank:104-130`、`wav_Enhance`（LFP）`:132-162`、`NS_FPN`:164-264`；SFS 类 `SpiralAware_CrossDeformAttn2D` 在 `NS-FPN/model/diff_cross_attns.py:38-125`。NS-FPN 每个 lateral 先可选 LFP（前四层，`:199-211`），再用 SFS 跨层 top-down（`:213-247`），最后 3×3 FPN conv 和 max-pool/extra conv（`:249-264`）。

NS-FPN 的 LFP 实际是 Haar DWT → LL 引导高频空间注意力 → 可学习 Gaussian filter → IDWT（`:145-162`），不是仅凭名称推断的普通低频卷积。SFS 使用 8 heads、4 points 的 spiral offset 和 `MSDeformAttn_for_sfs`（`diff_cross_attns.py:66-81,113-125`）；依赖 `pytorch_wavelets` 与 `SFS_MSDeformAttn` 自定义 C++/CUDA 扩展。

构造参数为标准 FPN 子集加 `upsample_mode`、`use_wav_enhance`、`use_crossattn_topdown`（`:164-194`）。通道假设是所有 lateral 先映射到相同 `out_channels`，SFS 的 dim 等于该通道数；其默认 `n_heads=8` 要求通道可被多头实现接受。forward 只接受 `inputs`（`:230`），没有元数据参数。可直接迁入：DWT/IDWT/LFP 的纯 Python/PyTorch部分和 NS_FPN 拓扑；必须适配：注册、AuxDet neck 签名、返回层数、算子编译与导入路径、服务器 CUDA/PyTorch ABI。算法含义风险：把 SFS 换成插值会失去 spiral-aware deformable attention；跳过 wave enhance 或改变仅前四层的规则会改变 NS-FPN 定义。

## 六、初始化与权重

- 基线 ResNet `init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')`（`configs/auxdet/auxdet_r50_fpn_1x_voc.py:10-18`）；README 另提供外部 AuxDet 预训练权重链接，但本地未找到权重文件，来源 key 结构未知。
- `default_runtime.py:23-24` 设置 `load_from=None`, `resume=False`。`tools/train.py:99-105`：`--resume auto` 设置 `cfg.resume=True, cfg.load_from=None`；给路径则同时 `resume=True`、`load_from=路径`。`tools/test.py:88` 把位置参数 checkpoint 写入 `cfg.load_from`。
- `frozen_stages=1` 和 `norm_eval=True` 在 ResNet 构造配置中（`:381-447`）；`_freeze_stages` 在 `resnet.py:614-629` 冻结 stem 和 layer1，`train()` 在 `:648-657` 将 BN 设 eval。优化器是 `schedule_1x.py:20-25` 的全模型 SGD，由 Runner/OptimWrapper 构建；本仓库没有针对 PConv/HS/NS 的特殊冻结逻辑。
- AuxFPN、HS/NS 新模块的参数若作为 `nn.Module` 挂入 neck，默认会进入优化器；但若误挂到 frozen backbone、误复制 `requires_grad=False` 或被自定义 optimizer paramwise 规则排除，会意外冻结。实现后必须检查 `named_parameters()` 和 optimizer param group。
- PConv 替换后原 `conv2.weight` 形状/key 无法直接加载到 PConv 的 `cw/ch/cat`；严格加载会 missing/unexpected keys。建议加载时允许非严格并记录 key，或写显式转换；不能假定 torchvision ResNet 权重可覆盖 PConv。

## 七、配置与服务器运行

当前 MMEngine 配置支持 `_base_` 继承（基线 `:1-5`），但未见 `custom_imports` 配置；自定义 AuxFPN 依靠 `mmdet/models/necks/__init__.py` 的静态导入注册。未来独立文件若不加入该导入链，必须在配置中使用 `custom_imports` 或显式 import，否则 registry 找不到模块。

真实可覆盖字段来自 `--cfg-options`，例如：`train_dataloader.batch_size=...`、`optim_wrapper.optimizer.lr=...`、`train_cfg.max_epochs=...`、`data_root=...`（数据配置中的最终字段）、`randomness.seed=...`（若配置/Runner支持该字段）。本次未运行 DictAction，复杂 list/tuple 必须按 `tools/train.py:36-47` 的引号规则传入。`--seed` 不是 `tools/train.py` 参数，不能把它写成已支持的独立 CLI 参数；应核对 MMEngine 配置的 `randomness` 字段或服务器脚本。

修改 `train_cfg.max_epochs` 时必须同步 `param_scheduler` 中 MultiStepLR 的 `end` 和 `milestones`（`schedule_1x.py:7-17`）；否则调度器仍按 12 epoch 的边界工作。warmup `end=500` 是否需调整取决于 batch/数据量，尚未运行验证。

图像/标注/元数据路径最终链路为 `data_root`（`configs/_base_/datasets/voc_aux.py:2`）+ dataset `data_prefix.sub_data_root='VOC2007/'` + `img_subdir='PNGImages'`/`ann_subdir='Annotations'` + `ann_file='VOC2007/ImageSets/Main/{train,val}.txt'`（`:50-87`）；XML 的 `<size>`, `<view>`, `<band_type>` 进入 sample。审计时配置曾使用旧机器绝对路径；2026-10-17 路径更新：AuxDet 数据配置统一为 `/root/projects/AUXDET/data_root/`，毕业配置默认计算仓库根目录下的 `data_root`，服务器运行时显式指定 `AUXDET_DATA_ROOT=/root/projects/AUXDET/data_root`。

训练输出默认 `./work_dirs/<config basename>`，可由 `--work-dir` 覆盖（`train.py:73-79`）；CheckpointHook 每 epoch 保存（`default_runtime.py:3-10`），最优权重只有在配置额外设置 `save_best` 等策略时才会生成，当前默认配置未确认有 `save_best`；README 所称 `best.pth` 不是由当前文件直接保证。恢复训练由 `--resume` 或配置 `resume`/`load_from` 控制，评估使用 `tools/test.py config checkpoint`。

六种配置推荐共用一条入口：一个原始 AuxDet 基础 config，分别以 `_base_` 覆盖 `model.neck.type`、可选 backbone PConv 开关/类型和 neck 参数；HS-FPN 与 NS-FPN 两个派生配置互斥，不在同一 config 同时设置。最终六个文件应只表达组合矩阵，训练/验证脚本保持不变。

## 八、建议的最小改动清单

| 目标功能 | 参考源码路径/类 | 拟修改或新增的位置 | 需要的接口适配 | 验证方法 |
|---|---|---|---|---|
| 原始 AuxDet | `AUXDET/mmdet/models/necks/aux_fpn.py:17-314`, 基线 config | 新增组合基线 config（实现阶段） | 保持 `neck(x, batch_data_samples)` 与两层输出 | 构建 config；单 batch shape/metadata smoke test |
| AuxDet + PConv | `PConv-SDloss-Data/model/APConv.py:PConv:36-54` | ResNet Bottleneck/新 PConv block 文件；注册与 config | 对齐 stride、c1/c2、BN/激活、残差和 state_dict | 前两个 stage 输出 shape、参数梯度、非严格加载 key 检查 |
| AuxDet + HS-FPN | `HS-FPN/hs_fpn.py:27-369` | 新当前 MMDet neck 文件；`necks/__init__.py` 或 `custom_imports` | `BaseModule`/registry、元数据可选签名、只返回 `[P2,P3]` 或同步 head | DCT/SDP 输入尺寸整除；两层 stride 与 RPN/ROI 对齐 |
| AuxDet + NS-FPN | `NS-FPN/model/NS_FPN.py:25-264`, `diff_cross_attns.py:38-125` | 新 neck、SFS 包和编译脚本；注册 | wavelet、MSDeformAttn ABI、元数据签名、两层输出 | 服务器编译扩展；单 forward、反向和多尺度 shape |
| AuxDet + PConv + HS-FPN | 上述 PConv + HS-FPN | 组合 config/兼容 block | 同时满足 PConv backbone 与 HS neck 接口 | 先分别 smoke test，再组合构建和一批训练迭代 |
| AuxDet + PConv + NS-FPN | 上述 PConv + NS-FPN | 组合 config/编译依赖 | 同时满足 PConv 权重映射与 NS 算子 | 服务器扩展编译、组合 forward/backward、恢复测试 |

推荐继承结构：`auxdet_r50_fpn_1x_voc.py` 保留可运行基线；新增一个公共实验基础 config（数据/训练/运行时保持原样），再由 `auxdet_pconv.py`、`auxdet_hsfpn.py`、`auxdet_nsfpn.py` 以及三个组合 config 仅覆盖模型字段。HS/NS 必须互斥；PConv 作为 backbone 的显式开关或替代 block 配置，避免复制六份数据和调度器。

必须先解决的阻塞：服务器实际 Python/PyTorch/CUDA/MMCV/MMEngine 版本；NS-FPN 自定义算子编译；HS/NS 到当前 `MODELS` registry 的迁移；neck 两层输出与 `[4,8]` head 的契约；元数据缺失时验证/预测的行为（AuxDet README 明确提到元数据可能只在训练集提供）。可以在实现阶段正常处理的适配：文件拆分、`custom_imports`、参数命名、PConv 初始化策略、extra level 的选择、日志和 work_dir 命名。

## 精简摘要与缺失信息

源码审计确认：AuxDet 当前基线是 `FasterRCNN + ResNet50 + AuxFPN + CascadeRPN + ROI head + VSBWILDVOCDetDataset + VOCMetric`；AuxFPN 以 `DetDataSample` 中的 view/band/width/height 生成辅助调制，并只向检测头返回两层。PConv 可参考 MIT 仓库的 `PConv`，但不是 Partial Convolution；HS-FPN 是 DCT 高频 HFP + SDP，NS-FPN 是 Haar DWT/LFP + spiral deformable SFS，二者都需要当前 MMDetection API 适配，NS 还需要编译自定义算子。

仍缺少、不能从本地确认的信息：服务器 GPU/驱动/CUDA、Python/PyTorch/MMCV/MMEngine 实际版本；服务器是否已安装 `torch-dct`、`pytorch-wavelets`、`einops` 及可编译工具链；正式数据根路径和验证集元数据完整性；AuxDet 预训练 checkpoint 的真实文件与 state_dict key；NS-FPN/其算子完整许可证；目标实验是否要求保留四层 FPN 还是严格沿用当前两层 `[4,8]` head。上述信息应在实现和服务器 smoke test 前补齐。
