# 4环境同步PPO接入与小训练

2026-09-06 00:04 UTC：并行实现、真实恢复验证及64场景pilot均已完成。训练尚未形成有效基线。

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

正式pilot从独立seed开始，CPU12–15/单lab、无GPU，完成64个完整场景（1920计划训练架次、16次池化更新），原12开发场景固定JSON与NR配对，完成训练前/32/64评价。保留原观测、奖励和所有边界指标，没有开启延迟或安全过滤研究。

## 本轮训练结果

一次初始化种子，64场景、261383样本、4091次Adam小批次步；所有episode/policy版本和种子连续、每批四场景全部终止后更新，没有部分批次或重复样本计数。源码/恢复证据已在训练前以`8926036`提交推送。训练job从2026-09-05 23:54:05至2026-09-06 00:04:34 UTC，内部627.593s/外部约629.48s，正常到达64目标并完成最终评价，低于840s内/900s外预算。

| 策略 | 完成架次/计划 | 完成率 | NMAC暴露秒/飞行小时 | LoWC暴露秒/飞行小时 |
| --- | ---: | ---: | ---: | ---: |
| No Resolution | 360/360 | 100.00% | 162.103 | 654.129 |
| 初始未训练 | 245/360 | 68.06% | 80.356 | 449.290 |
| 32场景（所选best） | 248/360 | 68.89% | 87.954 | 421.381 |
| 64场景（latest） | 244/360 | 67.78% | 75.954 | 411.223 |

以上是同一原12开发场景、同一评价随机种子的描述性结果；NR聚合在三个评价时点完全相同。最终NMAC率相对初始下降5.48%、LoWC率下降8.47%，完成架次少1架；中途NMAC一度变差。不能据此声称已稳定学到有效解脱。未训练策略本来就比NR有更低NMAC率，同时有明显任务失败，因此“低于NR”不能全部归功于学习。没有多训练种子、独立测试集或统计显著性结论。

所有候选完成率都低于预定95%门槛，所以选择规则先比较完成率；best32的248架高于latest64的244架，尽管best32的NMAC率更高。best在这里不是最佳安全模型或已有效模型；不冻结它开展延迟收益实验。

任务失败也保留：初始/32/64分别53/53/53超时、62/59/63路线耗尽。原横向越界分别337/331/328架、49394.5/49437.0/48014.5飞机秒，垂向均0；用户要求先搁置越界研究，本轮未改其观测/奖励/指标或将其伪装成已解决。全部风险pair秒、真实航时/航程及失败暴露保存在原始JSON。

## 实际训练吞吐与资源

完整四场景采样、传输及PPO更新平均11.44s/批（10.50–13.18s），合计182.98s、1428.45样本/s。三个开发评价用时410.18s，worker池启动25.98s；包括初始化、评价和保存的全job吞吐为416.48样本/s。没有同样学习/评价工作量的串行训练配对臂，不能把本轮直接称训练加速3.5倍。前轮3.5倍仍只属于固定策略采样测速。

完整NumPy payload累计94316035字节经私有pipe回传，包含发送完成回执、完整内容hash和父进程解码/合法性检查；不使用torch共享内存句柄。单worker最高记录RSS578.06MiB，不是父子总RAM硬配额。每进程4GiB虚拟地址上限、单lab、CPU12–15/nice15/idleIO。截止或故障仅关闭本池创建的Process对象。

[训练前](host-before-pilot.json)、[中途两次查询](host-during-pilot.jsonl)、[结束后](host-after-pilot.json)中两个原有PID3059945/3062529启动ticks均相同，未退出或重启；没有向它们发信号、调度或改动工作区。GPU使用仍20886MiB/free3162MiB；本轮CPU程序未获得GPU设备。观测支持它们存活，不支持其吞吐完全未受共享CPU竞争影响。

## 产物与下一步

- [真实训练manifest](records/20260905T235405Z-shared-parallel-pilot64-85c0c4fe/manifest.json)、[result](records/20260905T235405Z-shared-parallel-pilot64-85c0c4fe/artifacts/result.json)、[逐批记录](records/20260905T235405Z-shared-parallel-pilot64-85c0c4fe/artifacts/training.jsonl)、[逐开发场景评价](records/20260905T235405Z-shared-parallel-pilot64-85c0c4fe/artifacts/development.jsonl)。
- [独立归约汇总](records/20260906T000728Z-parallel-pilot-summary-ad898170/artifacts/summary.json)及[其manifest](records/20260906T000728Z-parallel-pilot-summary-ad898170/manifest.json)：逐批计数/种子/样本访问数、完整12场景、NR一致性及时长/效果归约全部通过。归约脚本完整保存于manifest命令。
- [latest64](checkpoints/latest.pt)、[best32](checkpoints/best.pt)均保存模型、Adam、RNG、严格配置/源码身份；latest内嵌best，后续从latest64继续同一配置，不能把best32当作已经达成安全目标。

本轮8个lab job中7个成功、首个测试失败已修复并保留；最终40项聚焦测试和真实恢复审计通过。实验最后00:04:34结束，归约00:07:28结束；没有自动继续更多训练。全部索引文件逐字节核对，包含五个可恢复checkpoint；完整source快照和临时BlueSky缓存仍在本机runs，未声称全量异地备份。

下一科研步骤是从latest64做一次有界、配置不变的延长训练，沿既定12开发评价节奏检查完成率是否上升。还没有证据要求改奖励、机型特征或先引入我们的预测/过滤模块；64场景的单种子pilot也不足以判断原文方法成败。
