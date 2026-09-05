# 001 — 原文式无延迟、无扰动基线

状态：`doing`（uv/BlueSky原生无控制诊断完成，论文式环境/PPO待实现）。负责人：主线程；实现时再指定builder的具体文件。目标顺序由用户确认：先有效基线，再冻结后加延迟。尚未执行新模型训练。

最近执行块（用户已明确“开始”）：2026-09-05 02:53:15 UTC起，最多15分钟至03:08:15 UTC，输出总目标≤2GiB；环境安装、单机与固定种子多机NR诊断，不训练。最后负载03:06:51 UTC结束，约13分37秒内完成；6个job严格串行，保存产物约75.5MiB（含源码快照与沙箱临时导航缓存，不含uv环境/缓存）。随后仅整理记录和Git交付。

## 要回答的问题

在真实BlueSky的同一训练/部署链路中，原文式共享策略PPO能否相对于相同初态No Resolution降低冲突，同时保持可接受的任务完成和运行代价？

原文依据：[REPRODUCTION](../paper/REPRODUCTION.md)。原文未披露值必须在新配置标明重建选择。不要修复旧路径后直接启动legacy半成品；它没有实现完整PPO、冻结评价或实际接通的配置。

## 已有事实

[旧FIT-002](../legacy/recovery-20260904/verified-fit002/result.json)训练8192 surrogate episodes，约40.65分钟；不是本轮训练。

[旧BlueSky汇总](../legacy/recovery-20260904/verified-heldout/result.json)：

| 指标 | No Resolution | Learned |
| --- | ---: | ---: |
| LoWC unordered pair-seconds | 88292.75 | 91252.4375 |
| NMAC unordered pair-seconds | 21070.4375 | 21865.0625 |
| 完成架次 | 667/720 | 660/720 |

旧模型未改善，不能证明原文方法无效。训练/评价链不一致和动作处理差异是调查线索；共同下降是否是根因仍未证实。旧24个世界已经被观察，后续只能作诊断。

## 连续DEV顺序，不再拆几十个VAL

1. **最小环境**：固定一个干净解释器/BlueSky版本与依赖，先在实际执行环境检查CUDA和CPU；不要混入旧py310二进制路径。初始环境只需能做headless step，不安装视觉模拟器或改驱动。
2. **一条物理链路**：场景/异质机型、入场与退出、Table2特征、CPA、动作锁/临时平行路线、Eqs.19–22奖励、终止和有向/无向安全暴露。训练和评价使用同一实现；配置真实读取。
3. **分钟级NR与随机/固定策略诊断**：验证路线、原始冲突和动作后果，测步速/内存/输出量；不据此宣布baseline复现。
4. **共享PPO小训练**：逐机GAE、PPO更新、保存/载入checkpoint、开发集NR配对曲线和动作分布。训练seed与开发seed分开，不碰最终测试。
5. **有效基线评价**：看到可信学习趋势后再扩大训练与新场景评价，固定checkpoint选择规则、部署方式和指标；多个训练种子验证稳定性。
6. **冻结后加延迟**：作为后续任务处理，先固定动作队列/覆盖/生效语义；0延迟走同一接口。没有退化也报告，不倒推挑场景。

## 第一批范围与资源

优先小闭环：单个launcher job、最多一个GPU训练；先≤15分钟、≤2GiB输出的DEV，用实测吞吐决定下一批预算。原文250k episodes是完整规模参照，不是首次启动就要跑满的数量。独立备份策略在重要长训练前落实。

2026-09-05早前接手：[系统精简复核](000-system.md)完成，22项隔离回归通过。当时只检查系统PATH和3.10的pip/tomllib入口，遗漏了uv托管解释器。下述核验补全了环境清单，并建立本项目独立环境；系统启动器、项目解释器和科学依赖的可用性分别记录。

训练配置至少包含：明确环境/源码版本、场景和机型参数、物理/决策步长、所有重建选择、训练/开发/测试划分、PPO超参数、随机源、部署方式、checkpoint选择规则、终止及缺失处理、资源预算。

## 评价与解释

同时报告LoWC/NMAC暴露、flight-hour分母、有向/无向计数、完成/失败架次、航程/延误、控制次数和动作分布。失败与删除不计为零风险；跨场景比较保留计划总体。绝对值对原文先统一计数口径。

小pilot只判断链路与趋势。最终评估不参与调参，不用物理tick冒充独立统计样本；效应不清楚时增加有信息量的重复或诊断，不补造成功故事。

## 下一次交接

- 问题：将已运行的原生BlueSky诊断扩成论文式环境。本批顺序由用户再次确认：先实现Table3异质性能、场景生成与入/退出，跑少量NR并统计冲突事件数、暴露、无冲突比例与完成率；之后才验证固定动作，再接观测/奖励/训练。
- 写入范围：src中的性能/场景/适配器、configs、必要回归及本文；训练/评价复用同一路径，不修改legacy。
- 依据：本任务、REPRODUCTION、原PDF pp.7、14–17；下述1.1.1 API审查与实测。
- 输出：参数对应表、速度/爬降/转弯的实际轨迹、入场与完成/失败统计，以及与原文未披露部分的明确区别。
- 停止条件：下一批按小DEV预算执行，到达已定预算或语义闭环完成即交接；本次15分钟授权已完成，不继承为无限训练授权。

## 2026-09-05：uv环境、经验复用与研究记录核对

用户追问系统学习能力、独立uv管理，以及研究目标/原文/改进/投稿记录。本轮范围是Python管理与隔离接入、相关说明和官方来源核验；未安装或导入BlueSky/PyTorch，未启动训练。

- 环境事实：`uv --version`为0.8.22；默认`uv python list`因用户全局缓存只读失败，`uv --no-cache python list 3.11`发现已有全局托管3.11.13。纠正“PATH无python3.11即可判断没有该解释器”的遗漏；不修改用户全局环境。
- 新环境：项目`.python-version`固定3.11.13，pyproject要求3.11系列且初始依赖为空；uv生成锁文件。`environments/python/`存放本项目托管解释器，`.venv/`为虚拟环境，`.cache/uv/`为缓存，三者不提交Git。最初选择依据是本机uv可提供的3.11版本；BlueSky/PyTorch兼容性仍待实际核验。
- 安装/锁定命令使用`UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/environments/python"`前缀，依次执行`uv python install --no-bin 3.11.13`、`uv lock --offline --managed-python`、`uv sync --locked --offline --managed-python --no-build`。下载解释器约28.8MiB；没有改变全局bin或驱动。
- 隔离接入：[uv-environment](../runs/20260905T024306Z-uv-environment-49a6e0d4/artifacts/environment.json)实际报告3.11.13，sys.prefix指向项目.venv，sys.base_prefix指向项目托管解释器；30秒/32MiB预算。两个runtime均只读挂入，版本/依赖元数据自动进入源码快照。
- 回归：[uv-system-tests](../runs/20260905T024317Z-uv-system-tests-e042d353/log.txt)在新venv解释器及外层launcher中23项全部通过、无跳过，90秒/256MiB预算；新增环境元数据不可变且不泄漏venv的回归。可复用命令见[ENVIRONMENT](../docs/ENVIRONMENT.md)。`uv sync --locked --check --offline --managed-python --no-build`报告无需改变环境；用新解释器的tomllib实际解析四份Codex主/角色配置通过。
- 学习机制：AGENTS补充新实现/诊断开始时定向检索LESSONS/SOURCES；SYSTEM明确文件记忆、知识纠正、代码/规则与回归的复用循环。没有自动微调Codex模型或后台学习服务。LESSONS补入本次uv清单遗漏；具体研究政策仍需未来PPO实验训练。
- 投稿只读交接`verify_venue`：问题为目标名称/截止日期/要求；范围为活动计划和官方出版商页面；交付来源链接、事实与未知；5分钟左右的一轮核验，不写文件、不运行项目代码、不嵌套委派。线程`01a06f6f-d3b3-7e11-94e8-7fcf0ef28f10`显式Astra/xhigh、fresh context，主线程核对实际turn_context为gpt-6-astra/xhigh。
- 投稿核验结果：读到Elsevier期刊介绍，可确认TR-C全称和一般范围；特刊正文/作者指南仍访问失败。未从混有多个征稿条目的搜索片段推导日期。PLAN/SOURCES保留特刊意向，明确当前开放状态、截止日期及投稿细则仍待官方核验。

剩余问题与下一步：建立BlueSky同链路最小环境；随后再固定PyTorch/CUDA依赖并验证实际GPU可见性。独立Python运行、系统回归通过均不代表模拟器或基线有效。

## 2026-09-05：首批真实BlueSky无控制诊断

用户授权开始上述15分钟DEV。本批完成环境安装、单机转弯、双机交叉、200ft垂直分离对照与交叉reset复跑；没有训练、没有读取最终测试集、没有改驱动。

**环境与实现。** `uv add --no-sync 'bluesky-simulator==1.1.1'`更新声明和锁文件。官方PyPI版本要求Python≥3.10，提供CPython3.11 Linux wheel，已在本项目3.11.13环境安装。`uv sync --locked --managed-python --no-build`初次失败于`zmq==0.0.0`只有sdist；内存检查官方966字节tar包及setup.py，确认只是声明pyzmq依赖的setuptools元包，SHA256 `6b1a1de53338646e8c8405803cffb659e8eb7bb02fff4c9be62a7acfac8370c9`。随后以ENVIRONMENT中的指定主要包禁源码构建命令成功sync，实际仅构建该元包、20个依赖安装完成。关键实测版本BlueSky1.1.1、numpy2.4.6、OpenAP2.6.0；全部解析结果见uv.lock。OpenAP是此BlueSky包依赖，不能据此认定原文使用它。

新入口[src/bluesky_diagnostic.py](../src/bluesky_diagnostic.py)和[配置](../configs/bluesky_smoke.json)使用原生Traffic、FMS/Autopilot与sim.step()，关闭CR、CD、风、噪声、VNAV，不提供学习策略。具名`DiagnosticPerformance(PerfBase)`不覆盖动力学，仅绕过基类实例选择问题；继承默认加速度2m/s²及不裁剪包络，B744仅为创建时的类型标识，未映射论文12类无人机。0.25s步长、25m/s、350ft；普通航点控制转弯，25m端点半径由wrapper判到达。指标独立计算Eq.21的垂直缩放阈值，左端点积分，先记录终点再删除；意外丢机直接报错。所有这些是显式诊断选择。

**保留的失败。**

- [初次初始化/步进](../runs/20260905T025757Z-bluesky-init-bd14082e/log.txt)：初始化成功，默认模型后直接`PERF OFF`使proxy自引用，在读取axmax时RecursionError。
- [最小自定义配置](../runs/20260905T030204Z-bluesky-nr-smoke-9c8c3aba/log.txt)：只设置enabled_plugins，缺navdata_path而初始化失败；改为继承固定版本default.cfg后覆盖plugins。
- [预构造PerfBase的尝试](../runs/20260905T030509Z-bluesky-nr-smoke-739c797d/log.txt)：实例身份断言失败。源码显示基类构造沿当前generator分派、再返回proxy，预调用不解决该问题；具名子类构造才返回具体实例。未修改第三方安装文件。初次将此归因于空TrafficArrays的猜测经源码审查否定，未作为事实保留。

**实测结果。** [首次完整成功](../runs/20260905T030546Z-bluesky-nr-smoke-944aee4d/artifacts/result.json)后，补充实际性能类型断言并修正配置文字，再做[最终成功运行](../runs/20260905T030644Z-bluesky-nr-final-b9a03656/artifacts/result.json)。最终各case目录含CSV轨迹和result.json，所有检查通过。

| 场景 | 完成/计划 | LoWC无向pair-s | NMAC无向pair-s | 累计flight-s |
| --- | ---: | ---: | ---: | ---: |
| 单机90°转弯 | 1/1 | 0 | 0 | 117 |
| 同高交叉 | 2/2 | 28 | 8.75 | 238.5 |
| 交叉、初始高度差200ft | 2/2 | 0 | 0 | 238.5 |
| 同高交叉reset复跑 | 2/2 | 28 | 8.75 | 238.5 |

同高交叉有向LoWC/NMAC为56/17.5 pair-s，飞行小时分母0.06625，对应无向422.64/132.08 s/flight-hour。这是人为相撞几何的诊断，不能与原文统计样本的6.88/.59直接比较。实际高度误差0，TAS误差<0.000046m/s，交叉积分起点的最小采样水平距离约2.03m；安全暴露非零证明相撞过程没有被路线/删除逻辑隐藏，不证明阈值具有实机安全保证。复跑CSV字节相同仅证明同进程reset的此场景可重复，正确性另外由完成、高度、速度、暴露等检查支持。

最终job总耗时约7.28s，其中冷初始化6.21s；1–2机rollout约5005–5439 steps/墙钟秒，含CSV记录、不含初始化，不能外推30机/PPO吞吐。进程累计峰值RSS约325.4MiB，不是独立case内存或硬聚合限额。六次job保存文件总计79,133,947字节，低于2GiB目标。

复用运行命令：

```bash
python3 -B tools/lab.py run --label bluesky-nr-smoke --stage dev \
  --config configs/bluesky_smoke.json --seconds 30 --disk-mib 128 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  -- "$PWD/.venv/bin/python" -B -m bluesky_diagnostic \
  --config configs/bluesky_smoke.json
```

**验证与只读审查。** [外层隔离回归](../runs/20260905T030608Z-bluesky-metrics-tests-6f6949ec/log.txt)25项通过、无跳过、6.623s；新增2项检查Eq.21根号形式/垂直容差及距离单位，未把它们称为端到端积分证明。最终native smoke另外覆盖初始化、具体性能实例、步进、reset、路线、终点记录和删除。只读Astra调查`bluesky_api_audit`交接：问题为1.1.1 headless/性能/航点与计数语义；范围活动文档和已装官方源码，禁止修改/实验/嵌套委派；来源为本任务、REPRODUCTION与包源码；输出API映射、风险及最小建议；约4分钟调查，后续2分钟代码复核。线程`01a06f7c-856f-7860-a0dc-f7a35552dc2c`两个turn_context均核对为gpt-6-astra/xhigh。调查者没有执行测试；主线程落实修复并执行所有job。

剩余问题：尚无原文12机型性能、30架次走廊人口、动作/观测/奖励、PPO或CUDA实测；单一诊断seed不是统计评价。下一步按上方交接实现论文式物理与动作链路，然后才启动共享PPO小训练。

## 2026-09-05：研究第一步前补齐完成通知

用户要求先增加主agent完成后的通知。仅完成系统通知与下一步准备，本轮没有开启新的研究执行块，未继承此前已结束的15分钟预算。

- 查明已有用户级Codex `notify`指向项目父目录`.codex-tools/codex_stage_notify.py`，接收topic已配置且权限0600，历史发送有成功记录；低空分支必须看到旧`CODEX_CONTROLLER_END`标记才发送，而活动系统已没有该规则。这是本轮定位出的接口脱节，未将历史发送记录当作近期设备送达证据。
- 本机Codex0.153.4；官方文档支持`agent-turn-complete` JSON参数。复用已有ntfy.sh接收渠道，新增受Git维护的`tools/notify.py`；本机分发器仅将低空事件转发到它，未修改用户全局认证/模型配置。原脚本可恢复备份为`/home/magic/ruitaozhou/.codex-tools/codex_stage_notify.before-low-altitude-20260905.py`，该本机分发器和私有topic均不随本仓库push保存，恢复方法见SYSTEM。
- 核对当前真实会话首条metadata为source=vscode、agent_path为空、cwd为本项目；脚本依会话UUID读取相应首条身份元数据，排除子agent/未知来源。显式模式读取同一会话最新turn_context，和原生事件共用session/turn去重。成功索引在忽略的`.cache/notifications.sqlite3`，只保存散列和UTC时间、最多4096条。传输失败后不标记成功，可重试；超时/进程崩溃后的远端是否已收到仍可能不确定。
- 主线程在记录完毕、最终回复前显式调用`python3 -B tools/notify.py --complete`，应对当前Remote/IDE可能没有本地notify事件的情形。仅发送固定“本轮结果已准备好”文字；不声称整个项目成功、不发送研究结果/路径/线程ID/对话。通知发送是用户授权的交付动作；离线代码测试继续经lab执行，未给科研负载放开网络。
- [初次回归](../runs/20260905T041815Z-notification-tests-4bae44d4/log.txt)29项通过；补充并发去重和最新turn选择后，[最终回归](../runs/20260905T042102Z-notification-final-tests-deb5a926/log.txt)31项通过。新增6项通知测试全用随机临时夹具和假网络，覆盖子agent/跨项目/异常事件过滤、显式与原生去重、同时回调、失败可重试且错误脱敏、固定发送内容/超时以及最新turn选择。只读AST比对确认本机分发器的发送/接收配置读取/项目识别/日志函数及其他项目分类逻辑未改变。
- 实际主线程执行新通知入口返回`{"state":"sent"}`，服务HTTP接受；topic值及响应中的私有内容未输出。用户随后明确确认“收到了，继续使用这个渠道”，本次接收已得到验证。原生客户端真实结束回调尚未单独证实，不能用构造测试替代，因此主线程仍保留显式通知。交付前再次调用将检验本轮去重。
- 只读review交接：问题为主线程识别/去重/失败隔离；范围仅tools/notify.py、tests/test_notify.py及AGENTS，禁止修改/实验/嵌套委派；来源官方notify协议和新代码；交付实质缺陷/证据/剩余项；≤3分钟。线程`01a06fca-1331-7f22-b7a7-3fdcb8b25781`实际turn_context核对为gpt-6-astra/xhigh。未发现阻断缺陷，指出显式发送早于最终回复、网络成功与本地落盘间不能保证恰好一次；已修正文案并记录界限。reviewer没有执行测试。

下一项研究工作仍为上方第一步：12类性能＋场景/入场退出＋小批NR。冲突事件次数需明确连续进入/退出的定义；原有28/8.75是无向pair-seconds，3个诊断配置和一次复跑不是足够的训练人口。NR检查信号分布，后续动作诊断才进一步检查可解脱空间。
