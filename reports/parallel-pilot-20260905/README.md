# 4环境同步PPO接入与小训练

2026-09-05：并行实现及真实恢复验证已完成；64场景pilot尚未启动，结果将继续写入本报告。

4个持久CPU环境用同一策略各完成一个场景，按绝对episode序号汇总原逐机GAE样本，父进程单线程做一次epoch1/minibatch64 PPO更新。动作随机源由绝对episode序号固定，完整样本通过有界NumPy payload传回且逐字段hash复核；失败/超时不提交部分批次。该调度改变原每场景后更新及优势归一化节奏，属于明确的并行变体。

40项聚焦回归通过。首次39项中3error来自Pickle5大数组的PickleBuffer无len()，已改用buffer nbytes并补大数组往返/限额回归。另修正script与worker异常类身份不一致，统一rollout_errors模块。失败记录保留。

真实BlueSky：同一resume配置、开发前1场景，连续8与4+恢复8均完成，0/4/8评价时点相同。审计确认全样本hash、模型、Adam、父随机状态、PPO指标、最佳模型与评价聚合精确相同。只支持本链路恢复正确性，不当作科研效果或全部12开发评价。

| 运行 | 用途 | 证据 |
| --- | --- | --- |
| 20260905T234346Z-parallel-integration-tests-774e3457 | 首次失败回归 | [manifest](records/20260905T234346Z-parallel-integration-tests-774e3457/manifest.json) · [log](records/20260905T234346Z-parallel-integration-tests-774e3457/log.txt) |
| 20260905T234453Z-parallel-integration-tests-5bdd3b53 | 最终40项通过 | [manifest](records/20260905T234453Z-parallel-integration-tests-5bdd3b53/manifest.json) · [log](records/20260905T234453Z-parallel-integration-tests-5bdd3b53/log.txt) |
| 20260905T234554Z-parallel-native-continuous8-b17f5f63 | 连续8 | [manifest](records/20260905T234554Z-parallel-native-continuous8-b17f5f63/manifest.json) · [log](records/20260905T234554Z-parallel-native-continuous8-b17f5f63/log.txt) |
| 20260905T234819Z-parallel-native-split4-a9ae09f8 | 分段4 | [manifest](records/20260905T234819Z-parallel-native-split4-a9ae09f8/manifest.json) · [log](records/20260905T234819Z-parallel-native-split4-a9ae09f8/log.txt) |
| 20260905T235015Z-parallel-native-resumed8-ddfe012b | 恢复8 | [manifest](records/20260905T235015Z-parallel-native-resumed8-ddfe012b/manifest.json) · [log](records/20260905T235015Z-parallel-native-resumed8-ddfe012b/log.txt) |
| 20260905T235146Z-parallel-native-resume-audit-ccb87dbf | 独立精确比较 | [manifest](records/20260905T235146Z-parallel-native-resume-audit-ccb87dbf/manifest.json) · [log](records/20260905T235146Z-parallel-native-resume-audit-ccb87dbf/log.txt) |

[恢复审计](records/20260905T235146Z-parallel-native-resume-audit-ccb87dbf/artifacts/result.json) · [备份索引](backup-index.json) · [pilot配置](../../configs/paper_train_parallel.json) · [恢复诊断配置](../../configs/paper_train_parallel_resume.json)。三个验证checkpoint包含模型/Adam/RNG/配置及嵌入best，仅用于验证，不用于初始化正式pilot。

正式pilot将从独立seed开始，CPU12–15/单lab、无GPU，最多64完整场景（16次池化更新），840s内/900s外，原12开发场景固定JSON与NR配对，训练前/32/64评价。保留原观测、奖励和所有边界指标，不开启延迟或安全过滤研究。
