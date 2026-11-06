# 干净与对抗损失权重的十轮对照

2026-11-07（Asia/Shanghai）。这是用户审阅报告后补做的事后实验，用于检验初始增强 PGD 方案中 0.75/0.25 的权重选择。不能将本次结果叙述为此前 40 轮实验前已经完成的调参证据。

## 测量前固定的方案

只比较干净/对抗交叉熵权重 0.5/0.5 与 0.75/0.25；其余配置完全相同。两组均独立从课程提供权重开始，固定 seed 5577、7,575/1,894 划分，训练 10 轮，batch 128，SGD 初始学习率 0.01、momentum 0.9、weight decay 0.0005，余弦计划长度为 10 轮。训练使用随机裁剪和水平翻转；每张图片保留增强后的干净版本，并针对当前模型生成单随机起点 PGD7，步长 2/255，L∞ 预算 8/255，取最后迭代。

主比较预先固定为第 10 轮的干净和自身白盒 PGD10 全攻击/固定半攻击 Macro-F1，不按干净最高分重新挑选轮次。同时保留全部逐轮指标，以及现有训练入口按白盒 PGD10 混合 F1 所选的最佳检查点信息。

两份第 10 轮检查点还使用统一的 FGSM8、PGD10、PGD20×3 完整评估，分别报告自身白盒与冻结课程原模型来源的攻击，使用同一固定 947/947 混合掩码。不会据本次诊断修改最终集成或提交新的 Kaggle 预测。

十轮余弦计划与原先四十轮计划的前十轮学习率不同；本次也与早期无增强、学习率 0.005 的 PGD 十轮实验不同。不能将这些跨配置差异归因于单个损失权重。课程权重已见过原始有标签数据，因此这些本地分数作为诊断比较。

## 复现命令

在仓库根目录运行，使用现有 CUDA Python 环境：

```powershell
.venv/Scripts/python.exe -m dataset3.training --training configs/training_pgd_augmented10_clean50.json --output runs/loss_weight_10/clean50
.venv/Scripts/python.exe -m dataset3.training --training configs/training_pgd_augmented10_clean75.json --output runs/loss_weight_10/clean75
.venv/Scripts/python.exe -m dataset3.evaluate_defense --checkpoint runs/loss_weight_10/clean50/last.pt --suite full --output runs/loss_weight_10/eval_clean50 --transfer-cache runs/loss_weight_10/transfer_cache
.venv/Scripts/python.exe -m dataset3.evaluate_defense --checkpoint runs/loss_weight_10/clean75/last.pt --suite full --output runs/loss_weight_10/eval_clean75 --transfer-cache runs/loss_weight_10/transfer_cache
```

## 结果

尚未测得；训练与评估完成后填写实际结果。
