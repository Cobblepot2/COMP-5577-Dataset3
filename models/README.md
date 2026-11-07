# 模型权重与训练对应表

2026-11-07 按本机实际训练记录整理。11 次训练的已保存最佳与最终权重共 21 份，另有 4 份原始推理导出和 21 份 BN 重校准权重，共 46 个权重文件。两组十轮损失权重实验 clean50、clean75 按用户要求排除。

## 每次训练的权重

编号用于阅读分组。每行代表一次独立训练，best 使用该次实验原有的选择规则，last 对应实际完成的最后一轮。文件名直接包含保存轮次。历史训练器保留 best 和 last，全部逐轮指标见原有结果记录。

| 编号 | 原始训练目录名 | 训练方案 | 初始化 | 完成轮次 / 余弦计划轮次 | 最佳权重 | 最终权重 |
|---|---|---|---|---|---|---|
| 01 | clean_provided | 干净样本微调 | 课程权重 | 10 / 10 | [best_epoch007](training/01_clean_provided/best_epoch007.pt) | [last_epoch010](training/01_clean_provided/last_epoch010.pt) |
| 02 | fgsm_provided | FGSM 对抗微调 | 课程权重 | 10 / 10 | [best_epoch007](training/02_fgsm_provided/best_epoch007.pt) | [last_epoch010](training/02_fgsm_provided/last_epoch010.pt) |
| 03 | pgd_provided | PGD7 对抗微调 | 课程权重 | 10 / 10 | [best_epoch009](training/03_pgd_provided/best_epoch009.pt) | [last_epoch010](training/03_pgd_provided/last_epoch010.pt) |
| 04 | pgd_random_short | 随机初始化 PGD7 短程训练 | 随机 | 10 / 10 | [best_epoch010](training/04_pgd_random_short/best_epoch010.pt) | [last_epoch010](training/04_pgd_random_short/last_epoch010.pt) |
| 05 | pgd_augmented | PGD7 加增强微调 | 课程权重 | 40 / 40 | [best_epoch039](training/05_pgd_augmented/best_epoch039.pt) | [last_epoch040](training/05_pgd_augmented/last_epoch040.pt) |
| 06 | pgd_random_converged | 随机初始化 PGD7 加增强训练 | 随机 | 50 / 50 | [best_epoch050](training/06_pgd_random_converged/best_epoch050.pt) | [last_epoch050](training/06_pgd_random_converged/last_epoch050.pt) |
| 07 | improve_trades_beta03 | TRADES β=0.3 微调 | 课程权重 | 40 / 40 | [best_epoch003](training/07_improve_trades_beta03/best_epoch003.pt) | [last_epoch040](training/07_improve_trades_beta03/last_epoch040.pt) |
| 08 | improve_trades_beta1 | TRADES β=1 微调；单模型低于预设干净 F1 门槛 | 课程权重 | 40 / 40 | [best_epoch039](training/08_improve_trades_beta1/best_epoch039.pt) | [last_epoch040](training/08_improve_trades_beta1/last_epoch040.pt) |
| 09 | improve_hybrid | 自身与课程源 Hybrid PGD 微调 | 课程权重 | 40 / 40 | [best_epoch037](training/09_improve_hybrid/best_epoch037.pt) | [last_epoch040](training/09_improve_hybrid/last_epoch040.pt) |
| 10 | public85_hybrid_fullrefit | Hybrid 全 9,469 张固定轮次重训 | 课程权重 | 37 / 40 | 固定轮次重训 | [last_epoch037](training/10_public85_hybrid_fullrefit/last_epoch037.pt) |
| 11 | public85_residual_source | 随机 SmallResNet18 课程源攻击训练；单模型低于预设干净 F1 门槛 | 随机 | 80 / 80 | [best_epoch069](training/11_public85_residual_source/best_epoch069.pt) | [last_epoch080](training/11_public85_residual_source/last_epoch080.pt) |

前六次实验按固定半攻击的自身白盒 PGD10 混合 Macro-F1 选择 best，干净 F1 用于打破平局。Public85 的分割训练先比较干净 F1 门槛，再比较课程源 FGSM/PGD 最差混合 F1。完整配置、选择规则、实际保存轮次及对应指标见 [manifest.json](manifest.json)。

Hybrid 全数据重训采用已冻结的 40 轮余弦计划，在第 37 轮结束，使用全部 9,469 张图片。Residual 完成 80 轮训练，第 69 轮被原有规则选中。TRADES β=1 和 Residual 单模型的门槛状态见对应清单。最终集成的两位成员及权重如下节所列。

六次早期训练的记录见 [training_comparison.json](../docs/results/training_comparison.json) 和 [results_5070ti.md](../docs/results_5070ti.md)。Public85 后续记录见 [study_summary.json](../docs/results/public85/study_summary.json) 和 [results_public85.md](../docs/results_public85.md)。固定种子为 5577，分割训练使用 7,575 张训练图片与 1,894 张诊断验证图片。课程原权重的微调模型和课程源攻击相关的验证解释沿用上述实验记录。

## 已提交预测所用的推理权重

以下四份文件按原有导出的字节原样保存，SHA-256 与历史推理清单一致。它们与上表相应训练检查点的张量逐项相同。

| 权重文件 | 对应模型 | 相同权重的训练导出 |
|---|---|---|
| [pgd_augmented.pt](deployed/pgd_augmented.pt) | 40 轮增强 PGD 的第 39 轮最佳模型 | [pgd_augmented](training/05_pgd_augmented/best_epoch039.pt) |
| [pgd_random_converged.pt](deployed/pgd_random_converged.pt) | 50 轮随机 PGD 的第 50 轮最佳模型 | [pgd_random_converged](training/06_pgd_random_converged/best_epoch050.pt) |
| [public85_hybrid_full37.pt](deployed/public85_hybrid_full37.pt) | 全数据 Hybrid 第 37 轮，最终集成成员 | [public85_hybrid_fullrefit](training/10_public85_hybrid_fullrefit/last_epoch037.pt) |
| [public85_residual_source.pt](deployed/public85_residual_source.pt) | Residual 第 69 轮，最终集成成员 | [public85_residual_source](training/11_public85_residual_source/best_epoch069.pt) |

最终 Public 0.85725 对应 Residual 第 69 轮与全数据 Hybrid 第 37 轮，各占 50% softmax 概率，随后平均原图与水平翻转两种视图。历史 Public 分数还包括增强 PGD 第 39 轮的 0.80418、随机 PGD 第 50 轮的 0.79277，以及全数据 Hybrid 第 37 轮加水平翻转的 0.81816。分数与原始证据见 [早期清单](../docs/results/final_artifacts.json) 及 [Public85 记录](../docs/results/public85/kaggle_scores.json)。

在仓库根目录使用现有 CUDA Python 环境，可通过原有预测入口加载最终集成。

```powershell
python -m dataset3.predict_defense --config configs/experiment.json --checkpoint models/deployed/public85_residual_source.pt --ensemble-checkpoint models/deployed/public85_hybrid_full37.pt --ensemble-weight 0.5 --horizontal-flip --output runs/reproduce_published_ensemble
```

表中任一训练权重均可作为 `--checkpoint` 传给现有 `dataset3.predict_defense` 或 `dataset3.evaluate_defense`。输入为 [0,1] RGB，归一化由模型内部执行。模型架构由权重中的 training_config 指定，Residual 使用仓库内的 SmallResNet18。

## BN 重校准实验权重

这些文件对应 Public85 中已记录的 21 次训练集 BN 统计重校准。可学习参数保持冻结，更新的是 BN 缓冲区。表中轮次是其基础训练模型的保存轮次，clean、mixed_fgsm、mixed_pgd10 标明校准数据。原始测量见 [bn_recalibration_probe.json](../docs/results/public85/bn_recalibration_probe.json)。

| 权重文件 | 基础模型轮次 | 校准方法 | 记录的干净 Macro-F1 |
|---|---|---|---|
| [checkpoints__pgd_augmented__bn_clean](bn_recalibration/checkpoints__pgd_augmented__bn_clean.pt) | 39 | clean | 0.862243 |
| [checkpoints__pgd_augmented__bn_mixed_fgsm](bn_recalibration/checkpoints__pgd_augmented__bn_mixed_fgsm.pt) | 39 | mixed_fgsm | 0.860738 |
| [checkpoints__pgd_augmented__bn_mixed_pgd10](bn_recalibration/checkpoints__pgd_augmented__bn_mixed_pgd10.pt) | 39 | mixed_pgd10 | 0.862217 |
| [improve_hybrid__best__bn_clean](bn_recalibration/improve_hybrid__best__bn_clean.pt) | 37 | clean | 0.880890 |
| [improve_hybrid__best__bn_mixed_fgsm](bn_recalibration/improve_hybrid__best__bn_mixed_fgsm.pt) | 37 | mixed_fgsm | 0.880222 |
| [improve_hybrid__best__bn_mixed_pgd10](bn_recalibration/improve_hybrid__best__bn_mixed_pgd10.pt) | 37 | mixed_pgd10 | 0.879680 |
| [improve_hybrid__last__bn_clean](bn_recalibration/improve_hybrid__last__bn_clean.pt) | 40 | clean | 0.881395 |
| [improve_hybrid__last__bn_mixed_fgsm](bn_recalibration/improve_hybrid__last__bn_mixed_fgsm.pt) | 40 | mixed_fgsm | 0.880265 |
| [improve_hybrid__last__bn_mixed_pgd10](bn_recalibration/improve_hybrid__last__bn_mixed_pgd10.pt) | 40 | mixed_pgd10 | 0.879640 |
| [improve_trades_beta03__best__bn_clean](bn_recalibration/improve_trades_beta03__best__bn_clean.pt) | 3 | clean | 0.971841 |
| [improve_trades_beta03__best__bn_mixed_fgsm](bn_recalibration/improve_trades_beta03__best__bn_mixed_fgsm.pt) | 3 | mixed_fgsm | 0.946958 |
| [improve_trades_beta03__best__bn_mixed_pgd10](bn_recalibration/improve_trades_beta03__best__bn_mixed_pgd10.pt) | 3 | mixed_pgd10 | 0.948533 |
| [improve_trades_beta03__last__bn_clean](bn_recalibration/improve_trades_beta03__last__bn_clean.pt) | 40 | clean | 0.948022 |
| [improve_trades_beta03__last__bn_mixed_fgsm](bn_recalibration/improve_trades_beta03__last__bn_mixed_fgsm.pt) | 40 | mixed_fgsm | 0.875017 |
| [improve_trades_beta03__last__bn_mixed_pgd10](bn_recalibration/improve_trades_beta03__last__bn_mixed_pgd10.pt) | 40 | mixed_pgd10 | 0.919178 |
| [improve_trades_beta1__best__bn_clean](bn_recalibration/improve_trades_beta1__best__bn_clean.pt) | 39 | clean | 0.841771 |
| [improve_trades_beta1__best__bn_mixed_fgsm](bn_recalibration/improve_trades_beta1__best__bn_mixed_fgsm.pt) | 39 | mixed_fgsm | 0.839469 |
| [improve_trades_beta1__best__bn_mixed_pgd10](bn_recalibration/improve_trades_beta1__best__bn_mixed_pgd10.pt) | 39 | mixed_pgd10 | 0.839928 |
| [improve_trades_beta1__last__bn_clean](bn_recalibration/improve_trades_beta1__last__bn_clean.pt) | 40 | clean | 0.842351 |
| [improve_trades_beta1__last__bn_mixed_fgsm](bn_recalibration/improve_trades_beta1__last__bn_mixed_fgsm.pt) | 40 | mixed_fgsm | 0.839469 |
| [improve_trades_beta1__last__bn_mixed_pgd10](bn_recalibration/improve_trades_beta1__last__bn_mixed_pgd10.pt) | 40 | mixed_pgd10 | 0.839928 |

## 文件格式与核验

training 目录保存由本机原始检查点提取的推理权重，包含 model_state_dict、训练配置、完成轮次、来源与选择元数据。deployed 和 bn_recalibration 保存原有推理文件的逐字节副本。全部 46 份文件均支持 `torch.load(path, map_location="cpu", weights_only=True)`。完整训练续跑状态保留于本机原始 runs 目录。

[manifest.json](manifest.json) 记录每份文件的字节数、SHA-256、原始来源、来源 SHA-256、张量摘要，以及每次训练的配置、数据量和选择记录。[verification.json](verification.json) 记录本次逐张量、严格加载与前向一致性核验。

课程提供的初始权重仍可直接从 [provided_model.pt](../data/dataset3/provided_model.pt) 下载，其 SHA-256 也列在清单中。
