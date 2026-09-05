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
| BlueSky1.1.1从默认性能模型直接切到PERF OFF后步进产生RecursionError；预调用PerfBase()也未得到所需实例 | 基类构造会按当前generator分派并返回proxy。诊断用具名PerfBase子类保留原生动力学，再断言实际实例；不修改vendor | [失败与修复记录](../tasks/001-baseline.md)；两次成功native smoke覆盖create/step/reset/delete |
| BlueSky自定义cfg只写enabled_plugins会缺失navdata_path并初始化失败 | 从锁定版本的default.cfg继承必需路径，再覆盖本轮选项，所有生成配置放在沙箱临时目录 | [配置失败](../runs/20260905T030204Z-bluesky-nr-smoke-9c8c3aba/log.txt)与任务001最终成功运行 |
| 源码检查：BlueSky1.1.1的DEST带有目的地高度/VNAV语义，普通航点则可明确保持平飞 | 平飞诊断采用普通经纬度航点、关闭VNAV并验证实际高度；不能把该机制直接判为旧模型共同下降根因 | installed traffic/route.py；[单机与交叉实测](../runs/20260905T030644Z-bluesky-nr-final-b9a03656/artifacts/result.json)高度误差为0 |
| BlueSky原生垂直速度更新使用高度差决定方向、取VS幅值 | 非对称爬降包络须按目标高度方向选上/下限，不能只看命令VS符号 | paper_performance.clip_intent；[12机型实测](../runs/20260905T043537Z-paper-performance-probe-1319e270/artifacts/performance_probe.json)，正VS命令下降也受正确下降限制 |
| 仅看终点距离会使近闭合路线在起点附近提前完成 | 同时要求最后航点已激活；保留终点状态和最后区间暴露 | nr_pilot.arrived_on_route；test_nearly_closed_route_cannot_arrive_at_its_origin |
| 240架次全到达仍有7架Amzn越出走廊半宽 | 单列路线偏离、终止完整性和冲突指标；先追踪实际转弯状态，再改变动力学或宣称复现 | [NR新诊断](../runs/20260905T044040Z-paper-nr-final-44ce1be4/artifacts/result.json)；fly-by/25°bank只是当前有依据的原因解释 |
| 高度已捕获时BlueSky仍可能保留非零VS；下一次5s命令反向会越出高度边界 | 完成语义区分位置误差与速度稳定，同时记录锁释放和实际命令时刻，不能仅看前者 | [8例原生对照](../runs/20260905T065518Z-vertical-lock-eight-fixed-a518ad01/artifacts/result.json)；test_settled_vertical_variant_keeps_lock_at_target_with_residual_climb |
| native末端LNAV关闭不代表已经满足25m到达或有限宽高出口；9超时中只有2例严重远飞 | 单列真实慢速超时、导航耗尽和成功，保留删前轨迹；不要只扩大到达阈值 | [原random逐步复现](../runs/20260905T062910Z-action-failure-trace-89478f96/artifacts/result.json)；route_completion聚焦回归 |
| 改mask或删除时间会改变全局Random.choice后续draw分配 | 同seed只保证初始输入/随机源，不保证同一逐机动作流；纯因果动作对照需固定可比干预与流分配 | [执行候选比较](../tasks/001-baseline.md)；paper_rollout的sorted-ID共享Random |
| BlueSky Proxy缓存bound method，普通setattr可能只改底层对象而未改变调用入口 | 包装实际调用facade并精确finally恢复，核对真实调用计数，不只看赋值成功 | [navigation敏感性](../runs/20260905T065855Z-navigation-sensitivity-seven-c4774d42/artifacts/result.json)；test_navigation_sensitivity的fake Proxy回归 |
| 假环境恢复测试不能覆盖BlueSky跨进程reset的隐式状态 | 首次集成保留真实native断点续训与连续运行参数/Adam/RNG比较；随后源码变更严格校验身份 | [native三轮精确比较](../runs/20260905T062109Z-literal-native-resume-compare-fe7da12f/artifacts/result.json)；checkpoint_compare.py |
| 条件动作熵约2.1不一定是策略塌缩：当前合法数大多只有4或20，H/log(valid_count)仍约.992 | 同时报合法支持、条件熵和跨状态混合熵，不能直接除log60；动作占比按决策加权而非飞行时间 | [100轮原生策略诊断](../reports/policy-diagnostic-100-20260905/README.md)，12例与原评价精确一致 |
| 当前53个超时全部是Mnet/Tecnalia，均匀四档速度的5NM平均航时约1279s超过1200s | 先核对机型包络、动作分布、路径长度和期限，再判断学习进展；不能靠放宽任务期限称改善 | [100轮逐机诊断](../runs/20260905T082427Z-policy-diagnostic-100-96e8dc93/artifacts/result.json)，29+24架次实测 |
| 一个共同时间分母或pair方向因子无法改变LoWC/potential与NMAC/potential比值 | 比较分母不变量，再定位场景/计数差异；不要只让两项绝对值接近便认定作者分母 | [原生双分母审计](../reports/exposure-audit-20260905/README.md)，当前15.456%/3.837%与论文34.42%/9.32% |

不要把尚未运行的测试或待核对解释写成已证实经验。新bug只补最小相关回归，避免将每次失败升级成新审批层。
