# 可复用教训

事实、解释和假说分开。持续学习指“记录经验→改规则或代码→回归验证→后续复用”，不是声称模型权重会自动学习。

| 已确认事实/观察 | 下次做法 | 来源/验证 |
| --- | --- | --- |
| 测试把空stdout拼成项目根，递归清理导致误删 | 不从子进程输出推导清理路径；独立随机夹具；项目测试整体放在隔离运行器中 | [事故记录](../legacy/recovery-20260904/INCIDENT.md)；tests/test_lab.py |
| 仅有LAB_RUN_DIR和hash不限制程序写入 | 固定源码快照只读，输出/临时目录单独挂载；无沙箱不启动 | tools/lab.py；源码写/删/改名及越界构造测试 |
| 嵌套沙箱不能提高父进程已经限制的资源硬上限 | 请求上限与继承硬上限取最小值，记录实际生效值，不为跑通测试放宽外层限制 | system-review失败与system-final回归；test_nested_limits_never_raise_inherited_hard_limit |
| 搜索用的后缀/大小过滤会遗漏实际源码资源，单纯复制字节还会丢可执行位 | 搜索可限小文本；源码快照完整保留普通资产和可执行位，超限或链接明确失败 | [系统复核](../tasks/000-system.md)；test_snapshot_preserves_assets_and_executable_mode |
| runtime若只检查默认项目，会遗漏实际的自定义项目根和祖先 | 以实际执行项目校验显式依赖目录，拒绝私人目录及项目祖先 | [系统复核](../tasks/000-system.md)；test_runtime_rejects_selected_project_ancestors_and_private_paths |
| 主模型配置不代表子agent实际模型 | 每次显式选Astra，核验线程元数据；禁止静默降级 | .codex/config.toml；任务000 |
| surrogate训练后的旧模型在BlueSky未改善冲突 | baseline先做真实BlueSky同一训练/评价链；保留负结果，不延长错误路径冒充复现 | [旧结果](../legacy/recovery-20260904/verified-heldout/result.json) |
| 228等构造测试通过不代表基线有效 | 少量关键语义测试后尽快跑训练→冻结→不避让对照的小闭环 | 旧E000/E008资料；任务001 |
| 删除/失败会使观测暴露变短 | 保留计划分母、失败和终止原因，同时报告完成/飞行暴露；不把缺失当零风险 | legacy中C001/E000审查 |
| 原文pair计数与旧pair-seconds不一定同口径 | 同时保留有向/无向计数和flight-hour分母，再比较绝对数值 | PDF p.11；REPRODUCTION.md |
| 高频共同下降是观察，不是已证明失败根因 | 明确部署sample/argmax、随机源和动作分布，做对照后再归因 | 旧session与原文Algorithm1 |
| 旧宿主的GPU可用记录不能代表新的解释器与沙箱 | 在将要训练的同一解释器/沙箱检查CUDA，再讨论驱动 | 任务001待办；旧GPU审计仅历史参考 |
| PATH中找不到python3.11，不代表uv没有管理该解释器；默认uv缓存还可能无写权限 | 检查uv及其解释器清单，为项目指定可写缓存/托管解释器位置，再在真实沙箱验证项目venv | [uv环境接入](../tasks/001-baseline.md)；本机uv清单已有3.11.13，上轮仅查PATH不足 |
| 文献模块有重合不自动否决组合创新 | 比较具体问题、假设和机制，借鉴强基线并测试新增部分必要性 | 用户确认路线；PLAN/SOURCES |

不要把尚未运行的测试或待核对解释写成已证实经验。新bug只补最小相关回归，避免将每次失败升级成新审批层。
