# C&W 补充攻击评估

在不重新训练模型的条件下，比较课程原模型、全数据 Hybrid37 加翻转、以及 Public 0.85725 的最终等权集成。三个预测器均被冻结；优化器只更新输入扰动。结果见 [完整汇总](results/cw/summary.json) 和 [中文版 Task 3 章节](task3_report_zh.md)。

## 方法

显式实现非定向 C&W-L2：最小化平方 L2 距离加 c 倍分类间隔惩罚 `max(z_y − max_other(z) + κ, 0)`，用 tanh 重参数化和 Adam 搜索输入，逐图片搜索 c，并保留历次成功候选中距离最小者。参数为 κ=0、初始 c=0.001、学习率0.01、7轮常数搜索，每轮最多500步，批量64。总批损失停滞时提前退出，因此批量与样本顺序也是本次配置的一部分。

- 标准 C&W-L2：像素范围为 [0,1]，不限制项目的 L∞ 预算。
- 受限变体：同一 L2 目标，另将搜索范围限制为 `[原图−8/255, 原图+8/255] ∩ [0,1]`。这是附加 L∞ 盒约束的 L2 变体，不是论文原始 C&W-L∞ 算法。

原模型使用原始 logits；Hybrid 与集成使用最终概率分布的 log 概率作为 logits。梯度经过所有模型、概率平均和水平翻转。未成功时返回干净输入；原本错误的输入保持不变。

使用既有固定分层200张采样，每类20张，种子5577，恢复原 CSV 顺序。ID 哈希为 `bc3f92d85740da6d8427aaad07f125f69a56b1105517ee27cb8a83ba37b93d01`。同时在同一采样上重新运行 PGD20×3，预算8/255、步长2/255。ASR 的分母是干净预测正确的样本数，分别为200、198、190。本地采样来自已有模型接触过的数据，用于安全诊断。

## 测得结果

以下 F1 均为全部200张受攻击后的浮点 Macro-F1。

| 预测器 | 干净 F1 | PGD20×3 | 标准 CW-L2 | 受限 CW-L2 | 受限 ASR |
|---|---:|---:|---:|---:|---:|
| 原课程模型 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 100.00% |
| Hybrid 与翻转 | 0.989994 | 0.387896 | 0.000000 | 0.392553 | 62.12% |
| 最终集成与翻转 | 0.949982 | 0.071876 | 0.000000 | 0.142899 | 85.79% |

标准 C&W-L2 的输出分别有20、195、124张超过8/255，因此与受限列分开解释。受限变体相对原模型有所改善，但最终集成的163/190张原本正确图片仍被攻击成功。Hybrid 的白盒表现更好，Public 最优方案与本地白盒最优方案不同。

Hybrid 上两种算法各发现5张对方没有发现的失败；同时通过干净、PGD和受限C&W检查的图片为70/200。最终集成上C&W未发现PGD之外的新失败，PGD则额外发现14张；三项联合正确为13/200。本轮C&W对Hybrid提供了补充覆盖。

### 量化结果

直接将最小距离浮点候选四舍五入到8位后，受限C&W的 F1 分别为1.000000、0.605241、0.741563。κ=0候选通常靠近决策边界；舍入改变分类并不代表离散输入上的优化攻击已经失败，本轮没有进行量化感知搜索。CPU与CUDA对部分边界候选也有预测差异。

[同GPU复核](results/cw/gpu_replay.json) 在三模型、两种范围、浮点及8位共12种条件下各重复两次，全部保存预测均一致。设置为 float32 matmul highest、CUDA matmul TF32关闭、cuDNN TF32开启、确定性算法开启、benchmark关闭、CUBLAS workspace `:4096:8`；精度标志在主运行结束后按未改变的环境记录。

## 耗时与资源

| 预测器 | 标准 CW 秒 | 受限 CW 秒 | 标准峰值 allocated MiB | 受限峰值 allocated MiB |
|---|---:|---:|---:|---:|
| 原课程模型 | 28.95 | 68.97 | 695.52 | 695.52 |
| Hybrid 与翻转 | 65.15 | 116.92 | 1271.78 | 1271.78 |
| 最终集成与翻转 | 271.27 | 353.50 | 3584.28 | 3582.28 |

六组累计904.77秒。计时包含输入复制、优化和浮点/8位预测，排除预热、初始载入、指标计算与文件写入。设备与运行版本保存在逐模型JSON。

## 复现与档案

在仓库根目录和既有 CUDA 环境运行，每条命令依次执行：

```powershell
../ml-env/Scripts/python.exe scripts/evaluate_cw.py --model baseline
../ml-env/Scripts/python.exe scripts/evaluate_cw.py --model hybrid
../ml-env/Scripts/python.exe scripts/evaluate_cw.py --model ensemble
../ml-env/Scripts/python.exe scripts/replay_cw.py
../ml-env/Scripts/python.exe scripts/summarize_cw.py
../ml-env/Scripts/python.exe -m unittest discover -s tests -v
```

主运行目录为 `runs/cw_evaluation_{model}`。浮点扰动张量留在本地；公开归档包括全部21份预测CSV、三份指标JSON、汇总与GPU复核记录。原模型运行使用的 runner 快照为 [baseline_run_source.py](results/cw/baseline_run_source.py)；其运行后补充了环境和辅助代码元数据，`provenance_attached_after_run` 已标明。Hybrid与集成直接使用当前 runner；攻击核心在三次运行间没有变化。

57项单元测试通过，其中4项新增测试覆盖C&W间隔方向、解析边界、扰动盒约束、逐图片常数搜索和模型不变性。三个模型的状态与检查点哈希均保持不变；Public0.85725的提交CSV哈希仍为 `180c2302fc163c247d5e7b2e7e36be46dfc80ccab446ecb385afd021f4b750c4`。

参考：[Carlini 与 Wagner 论文](https://arxiv.org/abs/1608.04644)、[作者的原始 L2 实现](https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py)、课程Week5第24至27页。课程PDF与全文转换结果不在公开仓库中分发。
