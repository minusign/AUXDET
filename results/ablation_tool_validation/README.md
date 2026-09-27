# 消融实验工具验证记录

日期：2026-09-27。

## 已通过

- 评测与分析测试：**38 passed in 29.96s**，见 [pytest.xml](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/ablation_tool_validation/pytest.xml)。
- 模块复查：LFP、SFS、与原 SFS 的数值/梯度等价性、B0–B4 neck 四组检查均通过，见 [neck_checks.json](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/results/ablation_tool_validation/neck_checks.json)。
- 合成数据实际运行 06 → 09 → 10 → 12，覆盖缓存读取、官方匹配对照、尺度分组、案例渲染、实验来源核验、统计表及 PNG/PDF 出图。
- 完成总体柱状图和分域变化热图的视觉检查；单次实验省略误差条，热图文字按背景选择颜色。

测试环境使用临时 Python 3.12.14、PyTorch 2.2.2、NumPy 1.26.4、MMEngine 0.10.7 和 MMCV-lite 2.1.0；模块依赖版本及模型源码哈希保存在 neck_checks.json。测试需要 OpenMP 共享内存，在获准的沙箱外执行。

## 适用范围

这些是 CPU 参考实现与合成数据正确性检查。测试中的 checkpoint 文件和指标只用于验证输入与输出，不是训练成果。模块复查使用 reference backend，不能代替 MMCV CUDA 算子的验收。

当前工作区没有真实数据、B0–B4 checkpoint 或 CUDA，尚未执行：

- 真实 B0/B4 官方预测缓存与官方日志的指标核验。
- 06 的完整检测器在线推理。
- 11 的 CUDA 延迟、显存、完整模型 FLOPs 测量。

服务器命令、数据要求、输出解释和各工具的使用情境见 [完整使用报告](/Users/aaash/Documents/2026夏/AUXDET/AUXDET/docs/zh_cn/auxdet_ablation_tools.md)。
