# 论文路线

## 固定目标

以 Fremond 等的 **Resilient multi-agent reinforcement learning for centralised tactical conflict resolution under uncertain perturbations and non-cooperative traffic in urban air mobility**（TR-C 184, 105542, 2026）为参照，研究执行延迟下的低空多机冲突解脱：先量化延迟的影响，再改善安全表现，同时保留任务完成率与运输效率的约束。

[原文 PDF](../resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf) · [原文逐项核对](REPRODUCTION.md)

目标期刊是 **Transportation Research Part C: Emerging Technologies（TR-C）**，目标特刊为 [Intelligent and Safe Operations of Low-Altitude Aerial Transportation Systems](https://www.sciencedirect.com/special-issue/333589/intelligent-and-safe-operations-of-low-altitude-aerial-transportation-systems)。2026-09-05通过[Elsevier期刊页](https://shop.elsevier.com/journals/transportation-research-part-c-emerging-technologies/0968-090X)确认期刊名称及安全、效率、实时运行等范围；特刊正文和作者指南仍访问失败，截止日期、当前征稿状态和具体投稿要求尚未核实。目标明确不等于已确认特刊仍开放；核实后再排投稿日期。

## 实验顺序

| 阶段 | 问题 | 需要得到什么 |
| --- | --- | --- |
| 1 | 无延迟、无扰动时，原文式模型能否减少冲突？ | 真实 BlueSky 同链路训练/评价，与同初态不避让对照，记录安全和运输效率；小规模先验证趋势 |
| 2 | 给同一个冻结模型加入动作输出延迟，会退化多少？ | 同场景、同模型只改变延迟，明确排队/覆盖/生效语义；不退化情形也报告 |
| 3 | 我们的方法是否改善延迟下表现？ | 时序记忆、预测、动作调度和安全过滤器的组合及消融，合理强基线与公平预算 |
| 4 | 是否足以形成论文？ | 多训练种子、新测试场景、密度/延迟泛化、成本与失败分析、理论假设和可复现材料 |

分钟级 smoke 和构造测试不等于基线有效。各阶段先做小闭环，再按实测速率扩大，不一次运行整个大实验计划；最新进展只在NOW和任务中维护。

## 保留的改进主线

1. 时序记忆：利用历史识别当前状态无法区分的趋势。
2. 生效时刻预测：预测指令真正生效时及随后一段时间的多机风险；必须与原文当前航迹 CPA 外推比较。
3. 动作调度与连续性：减少高频指令覆盖，明确机动完成、动作保持和延迟队列。
4. 数学安全过滤器：在明确动力学、误差、延迟界和可行性假设下约束神经网络输出；不可行、失配和回退单独报告。

以上是拟验证的改进方向。具体记忆结构、预测模型、动作频率及安全过滤器形式在有效基线和延迟诊断后确定，不能提前写成已经实现的算法或已证明的优势。

组合可以形成创新，但需要问题驱动的必要性与实验支持。借鉴近邻具体技术，不因某个模块已有研究就放弃主题；也不把新场景加模块直接宣布为已证明新颖。

## 复现边界

优先继承 BlueSky、共享策略 PPO、同一时刻的 intruder attention、7/10维观测、60类组合动作、5秒决策和机动完成约束。12类机型、奖励与指标按原文核对。OpenAP不是原文已明确指定的依赖，而是可能的重建选择。

未披露参数和原文歧义记录在 REPRODUCTION，不静默猜成作者设置。不得使用 surrogate 训练来替代 BlueSky 基线，也不先加入我们的预测/过滤/模仿学习再称其为原文 baseline。

旧24个已看过场景只作诊断，不能重新标成新 holdout。新评价集不参与模型选择。成功、失败、无改善都如实保留；不承诺录用。
