# 当前状态

更新：2026-09-05 22:51 UTC。分支 `research/trc-baseline-system-20260905`。

[001：无延迟、无扰动基线](../tasks/001-baseline.md)仍进行中；先有效paper-like BlueSky MARL，再冻结测延迟，最后研究改进。[目的/论文/期刊](PLAN.md)。旧850未有效，不冻结、不做延迟收益实验。

## 最新完成：共享基础导航与动作反馈

用户本次只授权完成1）把修正导航接入训练环境；2）核对动作执行和反馈。本轮已完成，[完整报告/结果/来源](../reports/shared-navigation-20260905/README.md)。没有训练或加载模型，没有继承已过期10h。

- PaperEnvironment在入场前安装共享限速器；NR和策略使用同一基础导航。保留模型的速度/高度/横向目标、锁及显式返回，修正变速后旧弯重复限制、偏移lane出弯对准和CAP/反向方位兼容问题。
- 原12开发场景，两调用臂各360/360到达、0横向/高度越界；12case逐步审计物理字段哈希相同，共1585482飞机物理步。共享NR的科学摘要与此前已验证NR限速精确一致；每臂41架实际限速、190行限速反馈。
- 新info明确请求/下发/实际TAS、实际跟踪、锁释放、完整本区间越界及终止状态。完成不会自动回中；step(None)保持之前目标；返回需显式动作。
- 8个速度/横移升高脚本全部到达，7次发出并完成返回。两例偏移弯道仍横向越界41.5/163.5s，均完整反馈。另1例因脚本在315s准备后多等5s、晚于318.75s出口而没发返回，不是锁未释放或无机会返回；1例没有初始请求速度/边界位置同时稳定的证据，不称完整目标保持成功。

原7/10维输入和奖励没改；**实际横向余量不在策略特征中，奖励没有直接越界罚项**。新增info只是诊断，不能据此说模型已获得边界学习信号。共享名义限速不是任意避让轨迹保证；完整在线认证/拒绝器、实验内收/强制回中/保持仍不作为当前基线前置。

52项唯一聚焦测试通过；11个lab job成功。CPU14/单线程/nice15/idleIO，单job、无GPU/torch/他人进程操作；最后负载22:44:48 UTC结束。41份结果/日志/manifest/旧源码文本4328246B逐字节核对，完整逐步轨迹只在线归约未保存，完整source快照仍本机runs，不称全量异地备份。代码/结果提交`1ee5ecfc62d7a7ac2a2df5b7e1859721ef40729c`已push原私有研究分支，本地/远端SHA精确一致。

## 下一步

本轮1和2授权工作到此完成。[未来小训练配置](../configs/paper_train_shared_navigation.json)已准备但未启动；后续若训练需从共享执行身份重新开始，不续旧850。训练解释必须包含当前边界观测/奖励缺口；若补充追踪特征或越界惩罚，应明确另一个训练变体。在线安全过滤保留后续研究，不自动恢复为训练前置项。

## 保留的前期证据

- [NR限速](../reports/nr-containment-20260905/README.md)：原12无干预16架217.5s越界→0，360到达；本轮已共享。
- [固定原12场景认证脚本](../reports/fixed-route-20260905/README.md)：每侧339完成/21保守拒绝，全部名义到达/0越界；固定脚本仍独立，没有接成policy安全过滤器。
- [横向条件证明](LATERAL_EXECUTION_PROOF.md)、[144构造见证](../reports/lateral-reference-20260905/README.md)及[旧动作1008例387越界](../reports/action-containment-20260905/README.md)保留各自范围，不追溯改写。
- [旧850与曲线](../reports/refresh-learning-20260905/README.md)：best775到达260/360、越界336；最终850sample233/340、argmax295/321，未有效。[模型/Adam/RNG](../checkpoints/refresh-850-20260905/README.md)独立保留。扰动/延迟仅[源规范](PERTURBATIONS.md)和编码诊断。

项目uv Python3.11.13/BlueSky1.1.1/torch2.9.1+cpu；[环境](../docs/ENVIRONMENT.md) · [系统/已授权完成通知](../docs/SYSTEM.md)。[REPRODUCTION](REPRODUCTION.md) · [LESSONS](LESSONS.md) · [SOURCES](SOURCES.md) · [DECISIONS](DECISIONS.md)。详细手交接与负结果在任务；旧审批链不启用。
