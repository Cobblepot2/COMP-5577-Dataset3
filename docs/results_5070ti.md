# Dataset 3：新设备训练与防御结果

> Historical stage snapshot: the later [Public85 study](results_public85.md) records the newest user-reported Public Score 0.85725. Values below describe the earlier stage.

日期：2026-11-05（Asia/Shanghai）。代码及过程见 [PR #4](https://github.com/Cobblepot2/COMP-5577-Dataset3/pull/4)。

已完成六组训练、七个模型的全量强攻击评估及两份测试集预测。预提交的白盒鲁棒性选择为随机初始化 PGD 50 epoch；用户实际上传后，增强 PGD 微调 40 epoch 的 **Kaggle Public Score 0.80418** 领先，随机初始化模型为 **0.79277**，基线为 **0.51194**。白盒指标和真实 Public Score 的选择依据分别保留。

## 已验证的 Kaggle Public 成绩

| 提交文件 | Public Score | 相对基线增加 |
|---|---:|---:|
| submission_baseline.csv | 0.51194 | — |
| submission_pgd_random50.csv | 0.79277 | +0.28083 |
| submission_pgd_finetuned40.csv | **0.80418** | **+0.29224** |

微调模型比随机初始化模型高 0.01141。本次已测 Public Score 优先候选为微调 40-epoch 模型；这不推翻随机模型在直接白盒强攻击下的优势。截图中三个提交均为 Complete，最终 Select 框均未勾选；没有代用户更改最终选择。证据接收于 2026-11-05（Asia/Shanghai），未显示 numeric submission ID 或绝对提交时间，不推测填写。

证据：[Kaggle 成绩截图](results/kaggle_defense_public_scores.png)。下方实验表格继续表示本地 Macro-F1，不能与这里的 Public Score 混填。

## 方法与数据边界

- RTX 5070 Ti 16GB、Core Ultra 9 285K、64GB RAM；Python 3.12.13、torch 2.9.1+cu128、CUDA 12.8、NumPy 2.2.6、Pillow 11.3.0。
- 只用提供的数据和允许的课程权重；原始 13,400 文件的大小和 SHA-256 全部一致。固定 seed=5577，训练 7,575、验证 1,894；验证混合集固定 947 张被攻击，mask seed=5578。
- 图像在 [0,1] 像素空间；规范化仅在 ProvidedNet 内部。FGSM/PGD 和训练算法直接实现，无现成攻击或防御黑盒。
- 训练 PGD 为单随机起点、7 步、步长 2/255、预算 8/255，交叉熵上升后取最后迭代；生成攻击时 eval，外层参数更新时 train。评估 PGD 按每张图片优先保留误分类候选，再比较 CE，遍历所有迭代和起点。
- 四组短训练均为 10 epoch、batch 128、SGD momentum 0.9、weight decay 0.0005、初始 LR 0.005、cosine、无增强；对抗组干净 CE 权重 0.5。长微调为 40 epoch、LR 0.01、干净权重 0.75、随机裁剪/翻转；长随机初始化为 50 epoch、LR 0.05、干净权重 0.5、相同增强。长实验同时改变多项设置，不能把收益单独归因于增强。
- 每轮检查全部验证集，以固定 50% 白盒 PGD10 混合 Macro-F1 选检查点，干净 F1 打破平局。最终候选按 FGSM8、PGD10、PGD20×3 三个白盒条件中最低的混合 F1 选择，干净 F1 打破平局；未用 Kaggle 分数调参。
- 课程模型已见过整个原训练集，微调后代的本地分数仍是诊断结果。随机初始化模型的梯度更新排除验证 ID，但验证集参与检查点和设置选择，因此也不是独立最终测试估计。

## 全量强攻击结果

白盒攻击直接针对当前候选；迁移攻击由冻结的原课程模型生成，再送入候选。训练期验证和最终评估的 PGD10 随机起点流不同；最终使用每批 seed=5577+batch_index，使 PGD10 轨迹包含在 PGD20×3 的首起点轨迹中，因此两处 PGD10 分数不保证完全相同。全部预算为 8/255，PGD 步长 2/255。混合 F1 从实际混合预测重新计算，不平均两个 F1。

| 候选 | 干净 F1 | 白盒 PGD20×3 F1 | 白盒 50% 混合 F1 | 迁移 PGD20×3 F1 | 迁移 50% 混合 F1 |
|---|---:|---:|---:|---:|---:|
| 原课程模型 | 0.999476 | 0.000000 | 0.512275 | 0.000000 | 0.512275 |
| 干净微调 10 epoch | 0.996238 | 0.000000 | 0.518964 | 0.000000 | 0.511772 |
| FGSM 微调 10 epoch | 0.900929 | 0.060370 | 0.479801 | 0.643473 | 0.762599 |
| PGD 微调 10 epoch | 0.785492 | 0.287391 | 0.546408 | 0.748946 | 0.767100 |
| 随机初始化 PGD 10 epoch | 0.644040 | 0.328063 | 0.492536 | 0.639883 | 0.641741 |
| 增强 PGD 微调 40 epoch | 0.862639 | 0.333362 | 0.607821 | 0.831076 | 0.848462 |
| 随机初始化增强 PGD 50 epoch | 0.796880 | 0.447401 | 0.630491 | 0.787553 | 0.791862 |

原模型在 PGD20×3 下 F1 为 0。随机 50-epoch 候选在同一测试协议下为 0.447401，混合集为 0.630491；增强微调候选干净 F1 0.862639、迁移 PGD20×3 F1 0.831076、迁移混合集 0.848462。二者面对不同威胁有不同优势；本次真实 Public Score 已观测为微调 0.80418、随机初始化 0.79277，未观测 Private/最终排名。

FGSM 微调的白盒 FGSM F1 为 0.431209，但白盒 PGD20×3 仅 0.060370，不能据 FGSM 结果宣称鲁棒。更强源攻击不保证对迁移目标也更强；这些迁移分数不能替代白盒分数。

所有七个模型的 PGD20×3 都保留了 PGD10 已找到的全部浮点失败样本。主候选额外出现 18 个失败，准确率从 0.465153 降到 0.455649；没有恢复为正确的旧失败。所有浮点和 8-bit 量化攻击均通过预算/像素边界检查（数值容差 1e-6）。完整量化分数、ASR、混淆矩阵和每类 F1 见机器可读结果。这里只是有限攻击集下的实测，没有鲁棒性认证。

主候选仍有明显弱点：白盒 PGD20×3 下 chain saw（类 3）召回率仅 0.168605（29/172），French horn（类 5）为 0.277487；parachute（类 9）召回率为 0.625。不能把总体提升解释为所有类别都安全。

## 耗时与复现

以下是实际每轮训练和验证用时之和，排除初始图片缓存、进程启动和轮后文件写入。峰值为 PyTorch allocated，不能等同整个进程显存或硬件总占用。

| 训练 | 最佳 epoch | 总 epoch 秒数 | 峰值分配显存 MiB |
|---|---:|---:|---:|
| 干净微调 10 epoch | 7 | 32.93 | 697.26 |
| FGSM 微调 10 epoch | 7 | 48.87 | 1277.44 |
| PGD 微调 10 epoch | 9 | 87.55 | 1281.95 |
| 随机初始化 PGD 10 epoch | 10 | 87.11 | 1277.44 |
| 增强 PGD 微调 40 epoch | 39 | 512.48 | 1296.31 |
| 随机初始化增强 PGD 50 epoch | 50 | 632.01 | 1296.31 |

原设备的步骤 3 已足以证明基线脆弱，不必因为硬件变化重复所有旧设置。新设备复现了完全相同的基线 CSV，并把算力用于训练后模型的强攻击评估。单批 GPU 检查通过；22 项单元测试通过；GPU 中断/恢复短试跑与连续训练参数逐位一致。精确恢复需原输出目录、完整 last.pt、相同设置和总 epoch horizon。

完整命令与约定见 `docs/training_and_defense.md`。主候选复现训练：

```powershell
python -m dataset3.training --training configs/training_random_convergence.json --output runs/pgd_random_converged
python -m dataset3.evaluate_defense --checkpoint runs/pgd_random_converged/best.pt --suite full --output runs/eval_pgd_random_converged --transfer-cache runs/transfer_cache
python -m dataset3.predict_defense --checkpoint runs/pgd_random_converged/best.pt --output runs/inference_pgd_random_converged
```

导出的模型是推理用紧凑检查点，含 model_state_dict 和普通 JSON 类型元数据，可使用 weights_only=True 加载。它们与实际受评估的源检查点参数逐位一致；优化器/RNG 不在推理文件内，不能代替训练恢复用 last.pt。实现版本为 `331ec5551dfdd33dc9c200a5ea01b38b31f08170`。

## 交付文件与下一步

| 角色 | CSV | 模型 | CSV SHA-256 |
|---|---|---|---|
| primary | `submission_pgd_random50.csv` | `pgd_random_converged.pt` | `579cf12d2df419dc7c84cc6600ab1601082c5b2dfc267d0ac1e9b97c4ebb0617` |
| comparison | `submission_pgd_finetuned40.csv` | `pgd_augmented.pt` | `ad5667a2dbfd816142abb558883db3f6b076a282bc902de7e3d2a2c6e26a4b4a` |

两份 CSV 都是 3,925 行、唯一 ID、样本模板顺序、整数标签 0–9。每个模型均重复完整测试推理，预测和 logits 逐位相同，并重新读取 CSV 检查。文件/模型 SHA-256 及模型对应关系见 `artifact_manifest.json`。

原预提交白盒鲁棒候选为 `submission_pgd_random50.csv`；真实 Public Score 最高的已测候选现为 `submission_pgd_finetuned40.csv`（0.80418，对比基线增加 0.29224）。保留两个模型和原始实验选择记录，由团队确认最终勾选；截图时均未勾选，未代用户操作。推理 JSON 保留生成时尚未提交的历史快照，最新成绩以 `kaggle_status.json` 和截图为准。本地分数不填入 Kaggle 成绩栏。PPT 继续按交接约定，等待用户提供指定的 presentation skill。
