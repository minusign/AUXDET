# AuxDet 六配置集成说明

## 代码状态

已实现六个配置：

| 实验 | 配置 |
|---|---|
| 原始 AuxDet | `configs/graduation/auxdet_base.py` |
| AuxDet + PConv | `configs/graduation/auxdet_pconv.py` |
| AuxDet + HS-FPN | `configs/graduation/auxdet_hsfpn.py` |
| AuxDet + NS-FPN | `configs/graduation/auxdet_nsfpn.py` |
| AuxDet + PConv + HS-FPN | `configs/graduation/auxdet_pconv_hsfpn.py` |
| AuxDet + PConv + NS-FPN | `configs/graduation/auxdet_pconv_nsfpn.py` |

所有配置继续使用 `tools/train.py` 和 `tools/test.py`。数据、元数据、检测头、损失、SGD 类型及 momentum/weight_decay、12 epoch 调度器和 VOC 评估均继承真实 AuxDet 基线。毕业设计公共配置显式设置 `lr=0.005`、`auto_scale_lr.enable=False`，原始 schedule 保持不变。HS 与 NS 通过 `model.neck.fusion_type` 互斥。

## 数据流和实现边界

`TwoStageDetector.extract_feat` 仍调用 `neck(backbone_outputs, batch_data_samples)`。`AuxFPN` 新增 `_prepare_aux_laterals`，保留原横向投影、两层语义补偿、元数据 MLP、动态通道/空间调制和 `EdgeConvSep` 的模块属性及计算顺序；原 `AuxFPN.forward` 只是在该方法之后执行原 top-down 和输出卷积。

`AuxFPNFusion` 继承 `AuxFPN`，因此不会把旧 AuxDet 模块搬入新的前缀。它复用同一组旧参数，调用同一个前处理方法，然后把四层已投影特征交给 HS 或 NS 融合主体。HS/NS 自己完成其融合所需的 top-down 路径，AuxFPNFusion 不再重复原普通 top-down，最后复用原输出卷积并严格返回两层。检测头和 anchor 配置没有修改，仍为 stride `[4, 8]`、256 通道。

## PConv

`mmdet/models/backbones/pconv.py` 从 `PConv-SDloss-Data/model/APConv.py` 迁入官方 `PConv`、内部 Conv、非对称 padding、共享横/纵卷积、BN 和 SiLU。`ResNetPConv` 只替换：

```text
layer1.0.conv2
layer2.0.conv2
```

实际替换时读取原层的 `stride`，默认 kernel 为 3；不把所有阶段首块都假定为 stride 2。原 Bottleneck 的 `norm2`、残差分支和最终 ReLU 保留，因此这是复合卷积模块替换，PConv 内部 BN/SiLU 与外部 ResNet norm/激活会叠加。

替换模块只保留在原 `layer*.0.conv2` 注册路径下；内部维护列表是非注册引用，不会额外产生 `_pconv_modules.*` state-dict 前缀。

基线冻结策略继续作用于 stem 和 layer1 的原始参数；`ResNetPConv.train()` 每次重新进入 train mode 后重新保证新增 PConv 参数可训练，并让 PConv 内部 BN 在训练阶段更新统计。eval mode 下整个模型正常切换到推理状态。原 `conv2.weight` 不会被 reshape 到 PConv，加载时应预期对应 missing/unexpected keys。

## HS 和 NS

HS 主体保留 DCT 高频路径、p1/p2 高频层级、p3/p4 注意力路径、HFP、分块 SDP 和参考实现中的 top-down 顺序。注册改为当前 `MODELS`，并在构造时检查 `torch-dct` 与 `einops`。

NS 主体保留 Haar DWT/IDWT、LL 引导高频调制、可学习高斯相关参数、spiral offset 初始化和 `MSDeformAttn_for_sfs`。SFS 源码放在 `mmdet/models/necks/sfs_ms_deform_attn/`，只迁入 Python/C++/CUDA 源码和构建文件，不迁入 `.so`、`.egg` 或 build 缓存。NS 依赖在选择 NS 配置时才加载；缺少 wavelet 或编译扩展会明确报错，不会静默跳过。

NS 算子来源包含 Deformable DETR 的 Apache 2.0 文件头；NS-FPN 根目录本次仍未找到独立 LICENSE，需在服务器发布前继续核实。

## 数据根目录、检查和评估

六个配置使用：

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate auxdet
cd /root/projects/AUXDET
export AUXDET_DATA_ROOT=/root/projects/AUXDET/data
export AUXDET_RUN_ROOT=/root/projects/AUXDET/work_dirs/graduation
```

环境变量未设置或为空时，通过 MMEngine 预定义变量 `{{fileDirname}}` 计算仓库根目录下的 `data_root` 绝对路径，不依赖命令工作目录。设置非空环境变量时使用用户给定值。配置解析不要求完整数据集存在。公共配置同时覆盖 train/val/test dataset 的 `data_root`，保留 `VOC2007/ImageSets/Main/{train,val}.txt`、`PNGImages` 和 `Annotations` 结构。

运行元数据检查：

```bash
python tools/analysis_tools/check_auxdet_data.py "$AUXDET_DATA_ROOT"
```

该检查统计 XML 缺失、尺寸非法、`view` 缺失/非法、`band_type` 缺失/非法；不会生成虚构元数据。

配置解析和模型 registry 检查：

```bash
python tools/analysis_tools/check_auxdet_integration.py
python tools/analysis_tools/check_auxdet_integration.py --build
```

不带 `--build` 只检查六份配置解析；带 `--build` 才构建六个 detector。脚本会把缺少框架、配置解析失败和模型构建失败分别报告，不会把未执行的 tensor、loss 或真实数据检查标成成功。

`VOCMetric` 默认 prefix 为 `pascal_voc`，默认 IoU 为 0.5；当前公共配置显式使用 `iou_thrs=[0.5]`，因此输出键为 `pascal_voc/AP50`，数值为 0–1 范围的 AP。`mAP` 在单一 IoU 阈值时等于该 AP50，但实验记录应保留 AP50 键。CheckpointHook 设置 `save_best='pascal_voc/AP50'` 和 `rule='greater'`。

当前 VOCMetric 的 `recall` 分支是 proposal-style `recall@N@IoU`/AR 汇总，不能直接声称是课题最终检测 Recall；本次没有擅自把它改成新的检测 Recall 定义。

## 依赖和 SFS 构建

AuxDet/PConv/HS 不需要导入 NS 的 CUDA 扩展。HS 服务器环境需要与当前 MMDetection/MMEngine 兼容的 `torch-dct` 和 `einops`。NS 还需要 `pytorch-wavelets==1.3.0`，并在服务器按实际 PyTorch/CUDA 编译：

```bash
cd /root/projects/AUXDET/mmdet/models/necks/sfs_ms_deform_attn
python setup.py build_ext --inplace
```

编译成功需由服务器实际日志和 import 结果确认；本地 Python 3.13 没有 PyTorch/MMCV/MMEngine，因此本地没有声称编译成功。

## 运行命令

先完成下文“公共配置修复验证”中的六组配置解析检查，再按依赖条件运行现有检查脚本的 `--build` 模型构建检查，最后才执行以下真实训练命令。配置解析、模型构建和真实训练验证是三个不同阶段。训练前激活环境并明确设置路径：

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate auxdet
cd /root/projects/AUXDET
export AUXDET_DATA_ROOT=/root/projects/AUXDET/data
export AUXDET_RUN_ROOT=/root/projects/AUXDET/work_dirs/graduation
```

`AUXDET_RUN_ROOT` 由 shell 展开并通过 `--work-dir` 传入，不会自动由公共配置读取。原始单卡 batch=4 的缩放结果为 `0.02 × 4 / 16 = 0.005`；毕业实验现在固定该默认值，改变 batch size 不会自动调整 LR。

```bash
python tools/train.py configs/graduation/auxdet_base.py --work-dir "$AUXDET_RUN_ROOT/e0" \
  --cfg-options train_dataloader.batch_size=4 optim_wrapper.optimizer.lr=0.005 auto_scale_lr.enable=False randomness.seed=0
python tools/train.py configs/graduation/auxdet_pconv.py --work-dir "$AUXDET_RUN_ROOT/e1" \
  --cfg-options train_dataloader.batch_size=4 optim_wrapper.optimizer.lr=0.005 auto_scale_lr.enable=False randomness.seed=0
python tools/train.py configs/graduation/auxdet_hsfpn.py --work-dir "$AUXDET_RUN_ROOT/hs" \
  --cfg-options train_dataloader.batch_size=4 optim_wrapper.optimizer.lr=0.005 auto_scale_lr.enable=False randomness.seed=0
python tools/train.py configs/graduation/auxdet_nsfpn.py --work-dir "$AUXDET_RUN_ROOT/ns" \
  --cfg-options train_dataloader.batch_size=4 optim_wrapper.optimizer.lr=0.005 auto_scale_lr.enable=False randomness.seed=0
```

组合配置分别使用 `auxdet_pconv_hsfpn.py` 和 `auxdet_pconv_nsfpn.py`，每个实验使用独立 work directory。学习率、batch size、轮数、数据根目录示例：

```bash
python tools/train.py configs/graduation/auxdet_pconv_hsfpn.py \
  --work-dir "$AUXDET_RUN_ROOT/pconv_hs" \
  --cfg-options \
    optim_wrapper.optimizer.lr=0.01 auto_scale_lr.enable=False \
    train_dataloader.batch_size=2 \
    train_cfg.max_epochs=24 \
    param_scheduler.1.end=24 \
    param_scheduler.1.milestones='[16,22]' \
    randomness.seed=123
```

修改 epoch 时同步修改 `param_scheduler` 中 MultiStepLR 的 `end` 和 `milestones`。训练入口实际支持 `--resume auto` 或 `--resume /path/to/checkpoint.pth`：

```bash
python tools/train.py configs/graduation/auxdet_base.py \
  --work-dir "$AUXDET_RUN_ROOT/e0" --resume auto
python tools/train.py configs/graduation/auxdet_base.py \
  --work-dir "$AUXDET_RUN_ROOT/e0" --resume "$AUXDET_RUN_ROOT/e0/epoch_12.pth"
```

验证：

```bash
python tools/test.py configs/graduation/auxdet_base.py "$AUXDET_RUN_ROOT/e0/best_pascal_voc_AP50.pth" \
  --work-dir "$AUXDET_RUN_ROOT/e0/eval"
```

实际 checkpoint 文件名以服务器 CheckpointHook 输出为准，不要预先假定一定叫 `best.pth`。每次服务器实验应记录 commit、最终解析配置、seed、完整命令、work_dir 和实际 checkpoint 路径；运行期间不要更新正在使用的源码目录。

## 验证状态

### 代码已完成

- PConv 官方方向分支和指定两层 ResNet 替换。
- AuxFPN 公共前处理提取，旧参数属性名保留。
- AuxFPNFusion 的 HS/NS 选择机制。
- HS 的 DCT/HFP/SDP 融合主体。
- NS 的 DWT/LFP/SFS 融合主体及源码构建目录。
- 六份配置、数据根目录覆盖、XML 元数据检查脚本和本说明。
- 六份配置的解析/registry 检查脚本（同一进程顺序检查）。

### 本地已验证

- 已检查目标仓库 Git 状态，仅目标仓库新增/修改集成文件。
- 已完成 Python 源码静态阅读和路径/注册链核对。
- 已确认本地 Python 3.13 缺少 PyTorch、MMCV、MMEngine、torch-dct、pytorch-wavelets、einops；因此没有伪造模型构建或训练成功结果。

### 服务器待验证

- 六份配置在兼容环境中的解析和 registry 构建。
- AuxFPN 重构前后相同输入/元数据的输出对比。
- 四层内部融合、两层输出、256 通道和 stride `[4,8]`。
- PConv 非方形输入、残差、冻结、BN 统计、梯度和 optimizer param group。
- HS 的 DCT/SDP 合法尺寸和反向。
- NS wavelet、SFS CUDA 编译、forward/backward、checkpoint reload 和预测。
- 一次实际 loss、反向、更新、保存、恢复和评估。

### 真实阻塞项

- 服务器已知 Python 3.9、MMEngine 0.10.7，解释器 `/root/anaconda3/envs/auxdet/bin/python`；PyTorch/CUDA/MMCV/编译器版本仍待确认。
- `torch-dct`、`pytorch-wavelets` 和 `einops` 的兼容版本需要服务器确认。
- NS-FPN 根目录许可证未找到，SFS 迁入发布前仍需核实。
- 本地无法运行 PyTorch/MMCV 相关检查；NS CUDA 编译和 GPU 运行必须由服务器完成。

## 公共配置修复验证（2026-10-17）

本次仅修改 `configs/graduation/auxdet_base.py` 和本说明，保留 `configs/_base_/datasets/voc_aux.py` 已有用户修改。公共配置使用 `{{fileDirname}}` 替代配置上下文中未定义的 `__file__`；环境变量缺失或为空时回退到仓库根目录/data_root；三个 dataset 根目录覆盖保持不变。只覆盖 `optim_wrapper.optimizer.lr=0.005` 和 `auto_scale_lr.enable=False`，继承 SGD、momentum、weight_decay、base_batch_size、预热、调度器和默认轮数，不使用 `_delete_`。

本地状态：**BLOCKED：未执行 Config.fromfile 验证。** 本地解释器缺少 MMEngine。语法检查不代表配置解析通过。以下服务器命令尚未在本次修复中执行，不需要 GPU、模型构建或完整数据集。

先激活环境、设置数据和输出根目录，再优先复用现有检查：

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate auxdet
cd /root/projects/AUXDET
export AUXDET_DATA_ROOT=/root/projects/AUXDET/data
export AUXDET_RUN_ROOT=/root/projects/AUXDET/work_dirs/graduation
python -c "import sys, mmengine; print(sys.executable); print(mmengine.__version__)"
python tools/analysis_tools/check_auxdet_integration.py
```

补充检查四种路径场景和 CLI 等效覆盖，每组配置在独立子进程中使用真正的 `Config.fromfile`，失败时保留 traceback 并返回非零：

```bash
/root/anaconda3/envs/auxdet/bin/python -B - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

repo = Path('/root/projects/AUXDET')
names = ('auxdet_base', 'auxdet_pconv', 'auxdet_hsfpn',
         'auxdet_nsfpn', 'auxdet_pconv_hsfpn', 'auxdet_pconv_nsfpn')
child = r"""
import sys
from mmengine.config import Config
path, expected = sys.argv[1:]
cfg = Config.fromfile(path)
for split in ('train', 'val', 'test'):
    actual = cfg[split + '_dataloader'].dataset.data_root
    assert actual == expected, (split, actual, expected)
assert cfg.optim_wrapper.optimizer.lr == 0.005
assert cfg.auto_scale_lr.enable is False
assert cfg.auto_scale_lr.base_batch_size == 16
assert cfg.optim_wrapper.optimizer.type == 'SGD'
assert cfg.optim_wrapper.optimizer.momentum == 0.9
assert cfg.optim_wrapper.optimizer.weight_decay == 0.0001
cfg.merge_from_dict({'optim_wrapper.optimizer.lr': 0.001,
                     'train_dataloader.batch_size': 2,
                     'auto_scale_lr.enable': False})
assert cfg.optim_wrapper.optimizer.lr == 0.001
assert cfg.train_dataloader.batch_size == 2
assert cfg.auto_scale_lr.enable is False
print('CONFIG_PARSE_AND_OVERRIDE_OK', path, expected)
"""
for label, value, cwd in (
    ('unset', None, repo),
    ('empty', '', repo),
    ('explicit', '/root/projects/AUXDET/data', repo),
    ('other_cwd', None, Path('/tmp')),
):
    env = os.environ.copy()
    env.pop('AUXDET_DATA_ROOT', None)
    if value is not None:
        env['AUXDET_DATA_ROOT'] = value
    expected = value or str(repo / 'data_root')
    for name in names:
        print('CHECK', label, name, flush=True)
        subprocess.run([sys.executable, '-B', '-c', child,
                        str(repo / 'configs/graduation' / (name + '.py')),
                        expected], cwd=str(cwd), env=env, check=True)
print('ALL_24_CONFIG_CASES_PASSED')
PY
```

以上成功只表示真实配置解析、路径和覆盖检查成功；`--build` 才执行模型构建，真实数据上的 loss、反向、checkpoint 和评估需要另行验证。本次没有执行模型构建或训练。
