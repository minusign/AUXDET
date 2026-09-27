# AuxDet / NS-FPN 消融实验工具使用报告

更新日期：2026-09-27。

## 1. 本次完成的代码

| 工具 | 使用情境 | 输入 | 主要输出 |
| --- | --- | --- | --- |
| [eval_common.py](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/auxdet_tools/eval_common.py) | 供其他脚本复用 | 配置、XML、PNG、划分、官方缓存 | 严格对齐的数据、VOC 匹配记录、来源信息 |
| [06 分域评测](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/auxdet_tools/06_ap50_recall_by_view_band.py) | 核验总体结果并比较不同域 | 官方预测缓存及对应配置 | 总体/分域指标、一致性报告、评测元信息 |
| [09 尺度分析](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/auxdet_tools/09_target_size_analysis.py) | 判断召回变化是否集中在较小目标 | 训练标注或验证集预测、固定尺度边界 | 尺度分布、尺度 Recall、相对基线变化 |
| [10 预测对比](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/auxdet_tools/10_compare_predictions.py) | 查找改善/退化样本，选择论文案例 | B0 与候选模型缓存 | 逐图/逐 GT/逐预测统计、PR/FPPI 曲线、三列案例图 |
| [11 性能统计](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/auxdet_tools/11_profile_models.py) | 比较模型运行成本 | 实验清单、模型配置、checkpoint、CUDA | 参数量、延迟、显存、FLOPs 覆盖情况 |
| [12 结果汇总](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tools/auxdet_tools/12_summarize_results.py) | 汇总初筛、重复实验和论文图表 | 实验清单及前述结果 | CSV、Markdown 表、PNG/PDF 图、配对 seed 差值 |
| [正确性测试](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/tests/test_tools/test_auxdet_analysis.py) | 修改分析逻辑后做回归检查 | 临时合成数据 | 官方匹配对照与命令行全流程测试 |

另外补充了 LFP/SFS 配套配置校验、[NS-FPN 依赖](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/requirements/nsfpn.txt)、[分析依赖](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/requirements/analysis.txt)和[实验清单模板](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/experiment_manifest.example.csv)。

本轮没有启动真实训练、重新生成 B0–B4 预测或覆盖已有 final_analysis CSV。新工具的正式数据输出由运行命令时指定的目录决定。

## 2. 建议使用顺序

1. 在原服务器环境检查 CUDA 模型运行，并确认 config/checkpoint 对应关系。
2. 用官方 test.py 为 B0、B4 导出预测，优先完成 06 的一致性核验。
3. 对 B1–B3 使用同一流程，得到同源的总体与分域结果。
4. 用训练集标注固定尺度边界，再分析验证集尺度 Recall。
5. 用 10 定位改善和退化案例；必要时再使用原 08 做 RPN 专项诊断。
6. 使用统一 CUDA 协议运行 11。
7. 填写实验清单，用 12 汇总论文表和图。
8. 根据分析结果选择 B0 和 1–2 个候选进行至少 3 个相同 seeds 的重复训练，重新导出缓存并汇总。

离线评测与分析不构建检测器。读取缓存时会验证对应 XML、PNG、划分和元数据。性能统计及官方预测导出在已有 GPU 环境运行。

## 3. 环境与实验清单

### 3.1 服务器环境

下列命令以已有服务器目录为例；REPLACE_B0、REPLACE_B4 必须替换为实际训练目录。

~~~bash
conda activate auxdet118
AUXDET_PROJECT=/home/ices/cjh/AuxDet
cd "$AUXDET_PROJECT"

python -m pip install -r "$AUXDET_PROJECT/requirements/nsfpn.txt"
python -m pip install -r "$AUXDET_PROJECT/requirements/analysis.txt"

B0_CONFIG="$AUXDET_PROJECT/configs/auxdet/auxdet_r50_fpn_1x_voc.py"
B4_CONFIG="$AUXDET_PROJECT/configs/auxdet/auxdet_r50_lfp_after_edge_sfs_p3_p2_1x_voc.py"
B0_CHECKPOINT="$AUXDET_PROJECT/work_dirs/REPLACE_B0/best.pth"
B4_CHECKPOINT="$AUXDET_PROJECT/work_dirs/REPLACE_B4/best.pth"
~~~

保留训练时匹配的 PyTorch/MMCV/CUDA 版本。分析依赖文件不主动替换它们。单独建立 CPU 分析环境时，需额外安装 torch；09 从训练标注计算 resize 比例还需要 mmcv-lite 或现有完整 MMCV。同一个环境中使用其中一种 MMCV 包。

当前工具明确支持本项目直接配置的 VSBWILDVOCDetDataset、本地 XML/PNG 和普通单尺度 test pipeline。数据集包装器、远程存储、裁剪/TTA 等没有等价坐标假设的输入会被拒绝。

### 3.2 实验清单怎么填

~~~bash
cp "$AUXDET_PROJECT/experiment_manifest.example.csv" \
   "$AUXDET_PROJECT/experiment_manifest.csv"
~~~

相对路径以清单所在目录为基准。各字段含义如下。

| 字段 | 填写要求 |
| --- | --- |
| experiment | 唯一运行名，例如 B4 或 B4_seed42 |
| model | 模型组名；重复实验也保留 B0/B4，便于跨 seed 汇总 |
| phase | screening 表示一次训练的初筛；repeat 表示多 seed 重复训练 |
| config | 对应这次实验的实际配置；优先使用保存的配置快照 |
| checkpoint | 同时计算 AP50、Recall 所用的同一个权重文件 |
| epoch | 该 checkpoint 的 epoch；repeat 必填 |
| seed | 实际训练 seed；repeat 必填整数，不从结果文件名猜测 |
| training_log | 对应训练日志文件 |
| predictions | 官方 test.py 导出的对应预测文件 |
| split | 这次评测用的 ID 清单 |
| training_protocol | 相同训练条件使用相同标识；另会比较配置中的优化器、调度、batch、数据和 hooks |
| evaluation_protocol | 相同评测协议使用相同标识；另会核对实际评测参数与数据哈希 |
| eval_dir | 06 输出目录 |
| size_dir | 可选，09 的验证集分析输出目录 |
| profile_dir | 可选，11 输出目录；多个实验可以指向同一个目录 |

模板中的 REPLACE 字段需要手动核实。旧单次实验的 seed/epoch 确实不详时可以留空，结果会归入 screening；这不能用于声称训练随机性已受控。repeat 中缺 seed/epoch、重复 seed、配对 seed 集不一致或复用同一 checkpoint 会被拒绝。

清单记录的 seed 属于实验来源声明；若配置本身有 randomness.seed，汇总工具会与其核对。它不会从预测结果反推出训练 seed。训练时用命令行覆盖的设置，应保存在对应配置快照和日志中。

## 4. 情境一：准备官方预测缓存

**适合：** checkpoint 已训练完成，希望后续分析共用完全相同的预测。

~~~bash
mkdir -p "$AUXDET_PROJECT/results/ablation/cache"

python "$AUXDET_PROJECT/tools/test.py" \
  "$B0_CONFIG" "$B0_CHECKPOINT" \
  --out "$AUXDET_PROJECT/results/ablation/cache/B0.pkl" \
  --work-dir "$AUXDET_PROJECT/results/ablation/official_B0"

python "$AUXDET_PROJECT/tools/test.py" \
  "$B4_CONFIG" "$B4_CHECKPOINT" \
  --out "$AUXDET_PROJECT/results/ablation/cache/B4.pkl" \
  --work-dir "$AUXDET_PROJECT/results/ablation/official_B4"
~~~

需要保留官方测试日志，之后用于一致性核验。B1–B3 的配置分别为：

- B1：auxdet_r50_lfp_p2_1x_voc.py。
- B2：auxdet_r50_lfp_p2_after_edge_1x_voc.py。
- B3：auxdet_r50_sfs_p3_p2_1x_voc.py。

所有正式缓存都应来自相同划分、输入处理、NMS 和预测分数设置。旧 04 helper 的 img_id 固定为 0，不能用它生成正式多模型对齐缓存。

缓存中的 GT 已由官方 DumpDetResults 移除。新工具通过配置和实际 ID 清单重读 XML，校验图像尺寸、域信息和 resize 参数。缺图、缺预测、额外 ID、重复 ID、缺元数据都会终止分析。

## 5. 情境二：核验总体指标与分域结果

**适合：** 判断候选模型总体是否改善、在哪些域发生变化，并核对与官方测试的一致性。

~~~bash
python "$AUXDET_PROJECT/tools/auxdet_tools/06_ap50_recall_by_view_band.py" \
  --config "$B0_CONFIG" \
  --predictions "$AUXDET_PROJECT/results/ablation/cache/B0.pkl" \
  --checkpoint "$B0_CHECKPOINT" \
  --model B0 \
  --output-dir "$AUXDET_PROJECT/results/ablation/B0"

python "$AUXDET_PROJECT/tools/auxdet_tools/06_ap50_recall_by_view_band.py" \
  --config "$B4_CONFIG" \
  --predictions "$AUXDET_PROJECT/results/ablation/cache/B4.pkl" \
  --checkpoint "$B4_CHECKPOINT" \
  --model B4 \
  --output-dir "$AUXDET_PROJECT/results/ablation/B4"
~~~

缓存模式中的 checkpoint 仅用于记录来源与哈希，不会创建模型或再推理。12 的正式汇总要求该来源信息，所以建议每次提供。

输出文件：

| 文件 | 用途 |
| --- | --- |
| overall_metrics.csv | 全部图像一起调用官方 eval_map 得到的 AP50/Recall |
| domain_metrics.csv | 按 view、band_type 分组重新评测 |
| parity_report.json | 逐框匹配与官方实现的一致性，以及可选外部核验 |
| evaluation_metadata.json | 配置、划分、权重、缓存、算法源码哈希和评测参数 |

数值保持未提前舍入的精度，展示时再取三位小数。Recall 为 IoU=0.5 下的最终检测召回率；AP50/Recall 使用 0–1 数值。没有普通 GT 的组，Recall 留空。

### 5.1 如何核对官方结果

在同一命令后添加：

~~~text
--official-metrics /绝对路径/official_B0.json
--require-parity
~~~

参考 JSON 的结构如下。以下数字仅说明格式，必须替换成**同一个 checkpoint、同一次官方测试**的真实值：

~~~json
{"AP50": 0.9, "Recall": 0.8, "gt_count": 100}
~~~

支持官方标量 JSON/JSONL 中的 pascal_voc/AP50、pascal_voc/mAP 等带前缀键；文件包含多次评测时，用 --official-step 指定实际 step。不要把 proposal recall@N 当作这里的 Recall，也不要用新 06 的结果反过来生成核验参考。

状态含义：

- pass：外部 AP50、最终 Recall 和 GT 数均可用且一致。
- partial：能检查的值一致，但外部参考缺某些指标。官方默认 mAP 标量文件通常没有 Recall、GT 数。
- fail：存在差异，命令返回非零退出码。
- not_requested：没有提供外部参考。

--require-parity 要求三个外部检查全部 pass；缺项也返回非零。没有 --require-parity 时，partial/not_requested 可以输出分析结果，但不会被标注成完整外部核验成功。

内部核验始终执行：逐框 TP/FP 与仓库官方 tpfp_default 对照，累计 GT/TP 与 eval_map 对照，分域计数之和与总体核对。外部 AP50/Recall 默认允许官方三位小数展示带来的半个末位舍入差；--reference-decimals 可指定参考精度。

### 5.2 兼容已有调用方式

旧 --config、--checkpoint、--ann-dir、--image-dir、--id-list、--output 和 --include-overall 仍可使用。没有 --predictions 时，06 通过实际官方 dataloader 做在线推理；--save-predictions 可保存结果。

建议正式分析优先使用 test.py 导出的缓存，减少不同推理运行之间的差异。正式 AP 对照时不要额外设置 --score-thr；指定它会改变分析所用预测，并记入协议。

## 6. 情境三：分析目标尺度与 Recall

**适合：** 总体或 Space/NIR Recall 降低，需要判断退化是否集中在某一目标尺度。

### 6.1 先用训练标注固定边界

~~~bash
python "$AUXDET_PROJECT/tools/auxdet_tools/09_target_size_analysis.py" \
  --config "$B0_CONFIG" --fit-bins \
  --output-dir "$AUXDET_PROJECT/results/ablation/frozen_size_bins"
~~~

默认用普通训练 GT 的输入尺度 sqrt(area) 的 1/3、2/3 分位点分成三组。统计时采用确定性的测试 Resize，描述目标在检测输入中的大小，排除训练随机增强的波动。--quantiles 可指定其他预先确定的分位点。

先检查 size_distribution.csv、size_distribution_summary.csv。若分位点因大量相同尺寸而重合，脚本会报错；此时根据训练分布明确选择边界，例如：

~~~bash
python "$AUXDET_PROJECT/tools/auxdet_tools/09_target_size_analysis.py" \
  --config "$B0_CONFIG" --fit-bins \
  --edges 0 4 8 16 \
  --output-dir "$AUXDET_PROJECT/results/ablation/frozen_size_bins"
~~~

4、8、16 只是命令格式示例，不是推荐阈值。固定 size_bins.json 后，B0–B4 共用同一文件。区间为左闭右开，最后一组无上界；例如值恰好为 8 时进入以 8 为下界的组。

### 6.2 对验证集计算尺度 Recall

~~~bash
python "$AUXDET_PROJECT/tools/auxdet_tools/09_target_size_analysis.py" \
  --config "$B4_CONFIG" \
  --predictions "$AUXDET_PROJECT/results/ablation/cache/B4.pkl" \
  --model B4 \
  --bins "$AUXDET_PROJECT/results/ablation/frozen_size_bins/size_bins.json" \
  --baseline-config "$B0_CONFIG" \
  --baseline-predictions "$AUXDET_PROJECT/results/ablation/cache/B0.pkl" \
  --output-dir "$AUXDET_PROJECT/results/ablation/B4/size"
~~~

为 B0 和其他候选执行同样的分析，并把各自 size_dir 写入实验清单。B0 可将自己的缓存同时作为 baseline，得到零变化量。

输出 target_size_metrics.csv 包含模型、域、尺度组、图像数、GT、TP、FN、Recall、baseline_Recall 和 delta_Recall，覆盖全验证集及 Space/NIR。

尺度定义明确为：

- 原图宽高：x2-x1+1、y2-y1+1，与本项目 VOC 含端点像素跨度一致。
- 输入宽高：上述跨度分别乘真实 sx、sy。
- 输入分组尺度：sqrt(input_width × input_height)。
- 验证集 sx、sy 直接使用缓存 scale_factor；padding 不算缩放。
- 匹配始终在原图坐标下完成；difficult 不进入普通 GT 的尺度 Recall 分母。
- image_count 是整个分析视角的图像数；images_with_gt_in_bin 是含该组普通 GT 的图像数，后一列不能跨尺度求和。
- 无 GT 的尺度组 Recall、变化量留空。该版本提供尺度 Recall，未定义尺度 AP。

## 7. 情境四：比较误检、漏检和论文案例

**适合：** 需要找出“B0 命中而 B4 漏检”的具体对象，或确认候选模型增加了哪些正确检出。

~~~bash
python "$AUXDET_PROJECT/tools/auxdet_tools/10_compare_predictions.py" \
  --baseline-config "$B0_CONFIG" \
  --baseline-predictions "$AUXDET_PROJECT/results/ablation/cache/B0.pkl" \
  --candidate-config "$B4_CONFIG" \
  --candidate-predictions "$AUXDET_PROJECT/results/ablation/cache/B4.pkl" \
  --baseline-name B0 --candidate-name B4 \
  --output-dir "$AUXDET_PROJECT/results/ablation/B0_vs_B4" \
  --render-max 12
~~~

主要结果：

| 文件 | 阅读方式 |
| --- | --- |
| per_image_comparison.csv | 比较每张图的 TP/FP、改善、退化及误差类别数量 |
| per_gt_comparison.csv | lost=基线命中候选漏检；gained=候选新增命中；both_missed=双方漏检 |
| per_prediction_matches.csv | 原缓存中的 prediction_index、匹配 GT、IoU、TP/FP/ignored |
| case_candidates.csv | 论文案例候选，保留改善和退化标签 |
| pr_curve.csv | 可实现的分数阈值下 Precision/Recall |
| fppi_curve.csv | 相同阈值下 Recall 和 FPPI=FP/图像数 |
| cases/*.png | 原图与 GT、B0 预测、候选预测三列图 |

匹配与错误类别采用：

- 按预测分数降序、按类别，与最大 IoU 的 GT 匹配。
- 最大 IoU 达到 0.5 且普通 GT 未被命中：TP。
- 同一普通 GT 再次命中：duplicate FP，不改配给第二近 GT。
- 最大 IoU 的 GT 为 difficult：ignored，包括重复命中。
- 未达到匹配阈值，但最大同类 IoU ≥ 0.1：localization；更低为 background。

localization 是按 IoU 定义的分析标签，不直接证明模型内部退化原因；可用 --localization-iou 调整并记录阈值。

同分数预测在 PR/FPPI 曲线中同时纳入，避免出现无法实现的阈值点。正式 AP 仍以 06 的官方 eval_map 为准，不从这些分组后的阈值曲线另算 AP。曲线只覆盖缓存中已通过模型分数过滤和 NMS 的最终框，无法恢复被删除的候选。

可编辑候选清单，只保留想展示的 image_id，再在相同命令中添加 --case-list /绝对路径/selected_cases.csv。该清单只决定展示哪些图，完整比较统计仍覆盖整个划分。--render-max=0 默认关闭自动渲染；传入 --case-list 且不限制数量时，会渲染清单中的全部图像。

--score-thr 会同时影响统计和渲染。为视觉清晰而提高阈值的结果应存放到不同目录，并明确标注阈值。

RPN proposal 不包含在这些最终预测缓存中。需要进一步排查 proposal 覆盖或中心偏移时，再使用原有 08_diagnose_space_nir.py 对选定样本做专项推理。

## 8. 情境五：统一测量模型代价

**适合：** 填写 Params、FLOPs、latency、memory 表。先完成 CUDA 自检：

~~~bash
python "$AUXDET_PROJECT/tools/analysis_tools/verify_nsfpn.py" \
  --backend cuda --large --full-model \
  --json "$AUXDET_PROJECT/results/ablation/cuda_checks.json"
~~~

然后填好实验清单中的真实 config/checkpoint，执行：

~~~bash
python "$AUXDET_PROJECT/tools/auxdet_tools/11_profile_models.py" \
  --manifest "$AUXDET_PROJECT/experiment_manifest.csv" \
  --experiments B0 B1 B2 B3 B4 \
  --device cuda:0 --shape 1024 1024 --batch-size 1 \
  --precision fp32 --warmup 20 --iterations 100 \
  --scope predict \
  --output-dir "$AUXDET_PROJECT/results/ablation/profile"
~~~

把相应实验的 profile_dir 都指向这个目录。

测量协议：

- 所有模型使用同一固定 seed 的 CPU 合成图像、输入大小、batch 和辅助元数据。默认 view=Space、band_type=NIR，可统一覆盖。
- 这是固定输入的可复现成本测试，不代表整份真实数据集的平均吞吐。
- predict 范围：已预处理、已搬到 GPU 的输入，到最终预测；包含检测头/NMS，排除预处理和 H2D。
- test-step 范围：包含预处理与 H2D；两种范围均排除读图和 dataloader I/O，并包含实际 Python 调用/样本复制开销。
- 每次计时前后 CUDA synchronize；预热后分别重置各模型显存峰值。
- latency_*_ms 单位为每个 batch 的毫秒；throughput_images_per_second 按 batch 大小换算。
- peak_allocated_mb 包含常驻模型、输入及运行峰值；incremental_peak_mb 为超过测量前常驻分配量的增量。reserved 单独报告。
- 成本测试使用严格 checkpoint 加载，误将 B0 权重配给 B4 配置会报错。

输出 complexity_metrics.csv 和 profiling_metadata.json。后者包含设备、版本、协议、实际输入形状、原始计时样本、来源哈希和未覆盖算子。

FLOPs 使用 MMEngine FlopAnalyzer 的 FP32 mode=tensor 路径，采用其 FMA=1 计数约定，与最终预测延迟的测量范围不同。flops_status=partial 表示存在未覆盖算子；error 表示分析失败；counted 表示工具没有报告未知算子，仍应按该工具的计数约定解释。--skip-flops 可以先完成参数、延迟和显存测量。

部分 FLOPs 必须在表中注明覆盖情况。不要把 unsupported_ops 当成零成本，也不要用单独 LFP/SFS 的显存代替完整模型显存。

## 9. 情境六：汇总论文图表和重复实验

**适合：** 前述结果已生成，准备写论文或比较多个 seeds。

~~~bash
python "$AUXDET_PROJECT/tools/auxdet_tools/12_summarize_results.py" \
  --manifest "$AUXDET_PROJECT/experiment_manifest.csv" \
  --baseline B0 \
  --output-dir "$AUXDET_PROJECT/results/ablation/paper"
~~~

输出包括：

- overall_runs.csv / overall_summary.csv：读取总体评测结果，不平均各域 AP。
- domain_runs.csv / domain_summary.csv：各域指标及相对 B0 差值。
- size_runs.csv / size_summary.csv：固定尺度组 Recall 及变化。
- complexity_runs.csv / complexity_summary.csv：成本统计与 FLOPs 覆盖标记。
- paired_seed_deltas.csv / paired_delta_summary.csv：候选与相同 seed 的 B0 差值。
- tables.md：可检查的 Markdown 表。
- figures：总体柱状图、分域 AP50 变化图、尺度 Recall 图、复杂度图，导出 300 dpi PNG 和 PDF。
- summary_metadata.json：所用清单、权重、训练日志、seed、评测身份和 parity 状态。

没有提供 size_dir/profile_dir 时，对应表和图跳过。需要这些比较时，应为基线与候选都提供相应结果。--no-plots 只导出表和元信息。

### 9.1 多 seed 使用方式

为每个实际训练运行新增一行，例：B0_seed0、B0_seed42、B0_seed2026、B4_seed0、B4_seed42、B4_seed2026。model 分别保持 B0/B4，phase 填 repeat。这里的 seeds 只是示例，必须填实际值。

每个 seed 的训练、checkpoint、官方缓存、06/09 输出都应单独存放。为 06、09 的 --model 传入 B0/B4 组名；用输出目录区分 seed。

默认至少 3 个 seeds，基线与候选的 seed 集必须一致。统计使用样本标准差，即 ddof=1；n=1 的初筛结果标准差留空，不伪装成 0。screening 与 repeat 分开汇总、出图。

汇总前会验证：

- checkpoint、预测、划分、配置文件哈希与 06 记录一致。
- 继承后的配置未改变；AP50/Recall 来自同一个 evaluation_id。
- 各运行使用相同训练协议、评测协议、数据划分、VOC 评测源码。
- 同一 model 的重复运行没有改变模型结构。
- size_bins 一致，性能测量协议一致。
- 分组样本数一致，GT/TP/FP/FN 与总体计数对应。
- repeat 具有完整 seed/epoch、无重复 seed、无复用 checkpoint。

汇总会读取清单引用的配置、checkpoint、缓存、划分和日志来核验哈希。文件都在服务器时，可直接在服务器运行 12，再把表格和图复制到写作机器；不需要 GPU 推理。

## 10. 验证状态与维护

### 10.1 本轮已执行

- 38 项 CPU 测试通过，包括重复命中、多 GT 竞争、difficult、空预测/无 GT、legacy IoU=0.5 边界、同分数排序、尺度边界和分组总数。
- 检查了真实 torch Tensor 格式的 pickle 读取、重复/缺失/额外 ID、元数据和 resize 错误、NaN、图像缺失。
- 检查了裁剪、TTA、随机测试尺度、无效 resize 和错误的 GT 加载顺序会被拒绝，避免在错误坐标下分析。
- 合成数据运行了 06→09→10→12 的实际命令行流程，生成了案例图、CSV、Markdown、PNG 和 PDF。
- 检查了 checkpoint 文件被替换、不同训练协议、重复 seeds、复用 checkpoint 的拒绝逻辑。
- LFP/SFS 不完整配置现在明确报 ValueError；现有 B0–B4 的合法配置仍由模块自检覆盖。

测试记录：[pytest.xml](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/ablation_tool_validation/pytest.xml)。模块复查：[neck_checks.json](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/ablation_tool_validation/neck_checks.json)。

### 10.2 服务器验收项

本机没有 CUDA，真实数据和训练权重也不在当前工作区。本轮没有执行 06 的 GPU 在线模式、11 的实际 GPU 测量，或真实 B0/B4 缓存与官方日志的一致性核验。它们的可执行入口和记录格式已经提供；正式论文结果应在服务器生成并验收。

11 的计时统计函数、命令行接口经过 CPU 测试，完整 GPU 执行仍待服务器验证。测试生成的 checkpoint 和指标都是合成夹具，不属于实验结果。

### 10.3 修改后如何重跑测试

~~~bash
cd "$AUXDET_PROJECT"
python -m pytest -o addopts='' \
  "$AUXDET_PROJECT/tests/test_tools/test_auxdet_analysis.py" -q
~~~

-o addopts='' 仅覆盖仓库默认的 xdoctest 参数，方便在没有安装该插件的分析环境运行。配置校验测试复用模块自检加载器，因此这组测试还需 NS-FPN 依赖和 pytest。

macOS 受限沙箱可能阻止 OpenMP 创建共享内存；本轮在获准的沙箱外运行这组本地合成测试后通过。该环境问题不会被记录成模型性能结果。

以后修改匹配、坐标处理、ID 对齐、尺度定义或统计汇总时，应重跑这组测试，再用同一份 B0/B4 缓存做实际评测一致性检查。
