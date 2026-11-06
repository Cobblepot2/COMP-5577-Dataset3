# C W 额外安全评估

用户要求测试 Week5 的不同攻击算法并加入中文 Task3 章节，不重新训练模型。C&W 与 PGD 都可以属于白盒攻击；本次比较算法，不更换已提交模型。

固定此前 label-only stratified200（20/class，seed5577，原 CSV 顺序），三条冻结预测器：课程原模型、full37 hybrid+flip、residual69/full37 equal probabilities+flip。先于新结果确定样本、模型和两种攻击设置。

实现 untargeted C&W-L2：min ||xadv-x||2² + c max(zy-maxother,0)，tanh reparameterization、Adam input modifier、per-image binary search c。confidence0，initialc.001，learningrate.01，binary7，最多500steps/constant，canonical loss-stall early abort。每张保留最小L2的成功误分类；初始已错计入总体F1但不计作新攻击成功。模型参数与BN不更新。

两种坐标范围独立测试：(1) standard box [0,1]；(2) additional coordinate bounds [max(0,x-8/255),min(1,x+8/255)]。第二种明确称为 C&W-L2 with additional Linf box，不称论文原始 C&W-Linf。同时记录L2与Linf，标准L2超预算攻击不作同预算鲁棒性证据。离散化仅为成功样本后处理，不声称PNG-aware搜索。

报告 float/8bit Macro-F1、initially-correct ASR、same200固定100/100混合F1、成功样本距离、时间/显存、所有文件/管线/代码身份及模型不变校验。比较同200 PGD20x3/CE50x5/margin50x5，不与全1894混写。各部署模型含fullfit成员的结果是in-sample诊断，课程原模型也已见这些训练ID；没有Private或新Public分数推断。

先用解析可解的线性边界、不可达预算、shift-invariant margin、逐样本批处理及参数不变测试验证。短GPU基准只用于估算运行成本；如有必要扩充迭代，仅作明示的收敛检查，不按结果选择防御模型。53旧测试与新测试均需通过。算法与结果独立review后归档、更新HANDOFF并经PR集成。

输出新版中文Task3报告，保留此前Word/Markdown原件。英文终版继续等待用户确认中文内容。原模型、checkpoint、CSV均保持身份不变。
