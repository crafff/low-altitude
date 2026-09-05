# 当前状态

更新：2026-09-05 11:18 UTC。发布分支 `research/trc-baseline-system-20260905`。

## 当前任务

[001：无延迟、无扰动基线](../tasks/001-baseline.md)。12类Table3包络、场景入场/退出、60动作、7/10维观测与奖励、共享attention PPO已在真实BlueSky接通。**仍无可信有效baseline，不冻结或开始延迟效果结论。** 所有已观察场景均为开发诊断，未打开新held-out。

用户10h授权：**05:15:13–15:15:13 UTC**，**14:45:13开始收尾**。有效名义基线为主，论文扰动/后续延迟机制核验为辅；不跨截止启动负载、不提前发送完成通知。

## 正在运行与下一步

[refresh-ppo-100](../runs/20260905T103813Z-refresh-ppo-100-4e103166/status.json)已于11:03:57正常完成100轮及最终评价（内部1541.836s），[小恢复点](../checkpoints/refresh-100-20260905/README.md)含model/Adam/RNG与best50，待本批私有备份。下一段计划同配置100→300、内3300s/外3360s；继续原科学源码535cad4，不跨谱系恢复。

新100/原0的[四模式对照](../reports/policy-modes-refresh-100-20260905/README.md)已完成48例、两sample精确复现，模型/输入不变；100argmax306/360、334width、NMAC162.230/h，原0argmax240/360，仍未形成有效baseline。8项模式fixture、标题实际导出回归及[0–100五点图](../reports/refresh-learning-20260905/README.md)已验证。当前本批待push，随后恢复100→300。

| 新refresh谱系，原12开发例/360架次 | 完成 | 超时 | 导航耗尽 | 横向越界架次 | NMAC / LoWC无向pair-s每flight-hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| NR初评 | 360 | 0 | 0 | 16 | 161.164 / 648.535 |
| 未训练sample | 246 | 53 | 61 | 339 | 78.926 / 435.953 |
| 25轮sample | 240 | 53 | 67 | 334 | 88.123 / 443.939 |
| 50轮sample | 252 | 53 | 55 | 336 | 76.568 / 423.420 |
| 75轮sample | 251 | 53 | 56 | 337 | 84.148 / 459.933 |
| 100轮sample | 240 | 53 | 67 | 341 | 78.075 / 415.963 |

上述各项均0高度越界；50轮较初始有所改善，75轮冲突暴露回升，尚无稳定趋势或可信containment。完成本段后保存实际checkpoint/完整评价与曲线，再按剩余墙钟及实测吞吐决定同配置续训。新四模式诊断v2的8项fixture和native重放均通过；sample保持主指标，argmax单独报告。不得把两种解码/旧新执行版本混成一条学习曲线。

## 支持当前选择的证据

- [导航刷新验证](../reports/navigation-refresh-20260905/README.md)：19相关core测试+3比较测试通过；205.973s原生集成全部通过。默认关闭原12例NR/sample科学摘要逐项复现；开启四固定响应2382物理行，以及30机NR/sample每机记录、实际调用与物理流，均精确等于已审计外部刷新。出生/删除/reset/恢复已实测。
- [50例固定响应](../reports/route-response-20260905/README.md)：减速后原生缓存仍1422.908m，而半速Amzn当前半径355.727m；中心最大偏离916.518→102.676m。1.05倍速度并非改善，不能声称普遍安全。Table3、bank、宽度、动作、观测、奖励及终止口径保持；两例唯一末步越界位于有效出口之后，保留原始统计与单独解释。
- [旧400轮模型](../checkpoints/execution-400-20260905/README.md)及[0–400曲线](../reports/execution-learning-20260905/README.md)完整保留，暂停旧400→600。旧sample400为236/360、53超时71耗尽、343width；[四解码对照](../reports/policy-modes-400-20260905/README.md)中400argmax298/360但338width，仍不构成有效基线。严格源码/配置身份禁止将其恢复或迁移到新refresh谱系。
- [训练预算](../reports/training-budget-20260905/README.md)：原文250k指每轮计划30机场景的外层收集/训练次数；旧400仅0.16%。已存全部400轮实际1616717转移、25463Adam步。新旧轮数不相加，不能称按论文规模验证失败。[真实Adam探针](../reports/ppo-gradient-100-20260905/README.md)完整原更新精确重现，未发现可据此确诊的优化器bug。
- [位置/通信规范](PERTURBATIONS.md)和[传感器六例](../reports/encoded-sensors-20260905/README.md)独立接口已验证，零扰动逐bit/物理重放一致；尚无扰动PPO/抗扰收益或NC实现。[暴露审计](../reports/exposure-audit-20260905/README.md)显示同走廊占NR NMAC92.90%，共同单位缩放不能解释原文暴露比例差异。

## 运行与恢复约束

本块只用**CPU14、单线程、nice15、idle IO、一个lab job**；每进程地址空间4096MiB、当前输出预算256MiB。用户要求不干预其他人的两个GPU实验；不向GPU提交计算，不发信号/调整其进程。10:30只读快照仍见PID3059945/3062529，显存9891/9771MiB、GPU98%；快照不证明外部吞吐绝对零影响。负载紧张则缩减/暂停我们自己的任务。

Python3.11.13/BlueSky1.1.1由uv独立管理，PyTorch2.9.1+cpu已锁定并完成真实前向/反向更新。CUDA12.8曾通过16KiB小核函数诊断，仅证明可见性/计算；本10h不用GPU。入口见[环境](../docs/ENVIRONMENT.md)与[系统](../docs/SYSTEM.md)。所有项目测试/实验走lab/Bubblewrap固定只读源码快照，不运行legacy或裸负载。旧143中断原因仍未知，后续正常退出不能反推发送者；原证据保留在任务中。

已备份私有GitHub：旧400及诊断`cc482f0`，已验证刷新源码/报告`535cad4`，均核对远端SHA。新100小副本已保存、待本批远端备份；原始runs、本地PDF及历史恢复归档仍无完整独立备份。Git实际状态以查询为准。

主线程完成本10h后沿已确认ntfy渠道执行 `python3 -B tools/notify.py --complete`；子agent不通知。完成前更新本页和任务、保存模型/报告并同步私有分支。

## 按需入口

[PLAN：目的、期刊和研究路线](PLAN.md) · [REPRODUCTION：论文参数/重建差异](REPRODUCTION.md) · [SOURCES：源索引](SOURCES.md) · [DECISIONS：选择依据](DECISIONS.md) · [LESSONS：可复用教训](LESSONS.md)。系统重建历史见[任务000](../tasks/000-system.md)，不重新启用旧控制链。
