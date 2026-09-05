# 001 — 原文式无延迟、无扰动基线

状态：`doing`（12类性能、场景、动作/观测/奖励和共享PPO已接通；正在验证小训练及路线执行缺陷）。负责人：主线程。目标顺序由用户确认：先有效基线，再冻结后加延迟。当前尚无有效新基线证据。

最新范围：用户已启动10h，2026-09-05 05:15:13–15:15:13 UTC；14:45:13起预留收尾。用户要求不停止/不影响其他人的两个GPU实验，当前块使用CPU且不提交GPU负载；执行预算与交接见末节。此前15分钟预算均已结束。

本次执行块已结束：用户再次明确“开始”，2026-09-05 04:29:42 UTC起，预算至04:44:42 UTC、≤15分钟/≤2GiB输出；12类性能、论文式场景生成/入场退出、小批无避让诊断，不训练。6个串行job最后于04:41:05 UTC结束，约11分23秒内完成；保存80,354,700字节产物（约76.6MiB），随后仅记录和Git交付。两个具名Astra/xhigh builder分别只写paper_performance及机型配置/测试、paper_scenarios及NR配置/测试；只读scholar核对原PDF场景语义。主线程写NR执行器/事件统计、集成并独占实验执行。

此前执行块（用户已明确“开始”）：2026-09-05 02:53:15 UTC起，最多15分钟至03:08:15 UTC，输出总目标≤2GiB；环境安装、单机与固定种子多机NR诊断，不训练。最后负载03:06:51 UTC结束，约13分37秒内完成；6个job严格串行，保存产物约75.5MiB（含源码快照与沙箱临时导航缓存，不含uv环境/缓存）。随后仅整理记录和Git交付。

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

- 问题：先定位Amzn原生转弯跟踪偏离，再固定路线跟随语义与入场即冲突的处理解释；随后验证固定动作，再接观测/奖励/训练。
- 写入范围：src中的可复用适配器/定向诊断、configs、必要回归及本文；不调NR seed筛掉失败，不修改legacy。
- 依据：下述最终NR保存的seed51005/F025、C04路线；REPRODUCTION及原PDF pp.7、14–17；BlueSky1.1.1 autopilot/activewpdata/traffic源码。
- 输出：该机位置、活动航点、目标/实际heading、bank、TAS、启动转弯距离的定向轨迹；区分已证实原因与解释，再决定最小跟踪修正。另保留17个LoWC/3个NMAC入场起始事件，不能通过未声明的延迟入场隐藏它们。
- 停止条件：下一批小DEV按实测速率定预算；本次15分钟执行块已完成，不继承为无限训练授权。原始NR必须留存，动作/训练尚未获得有效基线证据。

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

## 2026-09-05：12类性能、论文式场景与小批NR

**实现与范围。** 新增[机型表](../configs/uav_types.json)、[原生性能适配](../src/paper_performance.py)、[场景生成器](../src/paper_scenarios.py)、[NR执行器](../src/nr_pilot.py)和[实际配置](../configs/nr_pilot.json)。Table3的12个最大速度、非对称爬降上限和±3.5m/s²进入原生Traffic，不再使用B744占位或无包络PerfBase。速度按TAS、巡航取最大值80%；bank、垂直加速度及高度捕获仍保留原生值，未补造实机气动包络。

场景在2NM正方形抽样相隔≥1630ft的走廊起点，每条路线5NM、0–3中间航点、等长腿和≤90°转角；3/4/5走廊抽样，30架次按走廊均分，机型均匀抽样。每条走廊第一架t=0，其后iid均匀60–90s间隔；入场取下一个0.25s tick，不动态筛掉冲突。最后航点激活且进入25m半径才到达；1200s仿真年龄超时单独记失败；保留待入场人口、入场原始风险、终点状态和删除前区间暴露。分布、到达、时钟歧义和原文页码对应见REPRODUCTION与配置。NR仍无策略动作、观测/奖励或PPO。

**实际执行（主线程串行）。**

1. [系统/场景/性能初次回归](../runs/20260905T043517Z-paper-environment-tests-c89c1a06/log.txt)：42项通过，6.785s；60秒/128MiB预算。
2. [12机型原生probe](../runs/20260905T043537Z-paper-performance-probe-1319e270/artifacts/performance_probe.json)：22个检查全通过，64s仿真、约6.92s job墙钟。检查全部12机型的创建、最大TAS、爬降、±3.5加减速度、实例/数组、删除/重置/重建、未知机型拒绝。正VS命令下降时仍正确取下降上限；30秒/128MiB预算。
3. [单场景吞吐与入/退出](../runs/20260905T043717Z-paper-nr-one-e9f9026b/artifacts/result.json)：seed51001、30/30到达；约8.82s job。90秒/128MiB预算。
4. [首轮8场景](../runs/20260905T043810Z-paper-nr-eight-91c0d774/artifacts/result.json)：240/240完成；约23.41s job。随后增加同/跨走廊分类、入场事件分类及走廊偏离诊断，未修改动力学或采样种子。
5. [最终回归](../runs/20260905T044023Z-paper-environment-final-tests-d38dfe5b/log.txt)：44项通过、无跳过、6.826s；加入近闭合路线提前完成的回归和线段距离测试。60秒/128MiB预算。
6. [最终8场景](../runs/20260905T044040Z-paper-nr-final-44ce1be4/artifacts/result.json)：约24.22s job，rollout合计16.79s，峰值进程RSS约327.7MiB；90秒/256MiB预算。源码、场景、每机终止记录及所有冲突事件已保存；未保存全量逐步轨迹，后续对偏离机定向补录。

上述复跑不增加独立场景数：独立开发seed固定为51001–51008。它们恰好抽到7个四走廊、1个五走廊，未在本批原生NR覆盖三走廊；生成器单测覆盖3/4/5。不据此声称完整分布评价或持出测试。

| Seed | 走廊 | 到达/计划 | LoWC事件 | NMAC事件 | 超走廊宽度架次 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 51001 | 4 | 30/30 | 68 | 38 | 1 |
| 51002 | 4 | 30/30 | 64 | 38 | 0 |
| 51003 | 4 | 30/30 | 52 | 33 | 0 |
| 51004 | 5 | 30/30 | 39 | 26 | 1 |
| 51005 | 4 | 30/30 | 21 | 16 | 1 |
| 51006 | 4 | 30/30 | 35 | 27 | 0 |
| 51007 | 4 | 30/30 | 55 | 42 | 2 |
| 51008 | 4 | 30/30 | 63 | 33 | 2 |

**计数与解释。** 累计38.9025 flight-hours；LoWC397个连续pair事件、27557.25无向pair-s（有向55114.5），NMAC253个、6285.25无向pair-s（有向12570.5）；对应无向708.37/161.56 s/flight-hour。LoWC同走廊224、跨走廊173；NMAC同走廊205、跨走廊48。LoWC/NMAC入场起始事件17/3，已包括在上述总数中。两级阈值嵌套，事件数不可相加当作独立总冲突。无LoWC场景比例0%，无NMAC场景比例0%；无LoWC架次16/240（6.67%），无NMAC架次35/240（14.58%）。0超时、0缺失，最大在途30；高度误差0，名义TAS误差<0.000094m/s。系统检查通过只证明已列出的行为/计数，不能替代下面的走廊诊断。

**发现的问题。** 7/240架次超过76.2m走廊半宽，均为Amzn，累计64.25 aircraft-s；最大363.146m，来自seed51005/F025。原生默认fly-by和25°bank在80.665m/s时对应约1.42km转弯半径，是有源码支持且量级一致的解释；没有逐步转弯轨迹，尚不称为已定位根因。下批最小诊断见上方交接。不能以240/240完成掩盖宽度不满足，也不能为提高完成率/冲突指标筛掉这些飞机或seed。当前冲突样本不稀缺；策略可解性、奖励信号和泛化仍需后续动作与训练验证。

**交接与审查。** builder `paper_performance`（线程`01a06fd4-c44b-7d30-b8d9-0a035348e960`）只写性能表/适配/单测/probe；builder `paper_scenarios`（`01a06fd5-301e-7d13-b6e8-a83e05c40e50`）只写生成器/NR配置/单测，均约6分钟上限、禁止实验/清理/嵌套委派并明确共享工作区。输入为本任务、REPRODUCTION和已装BlueSky源码；输出为约定API和待主线程执行的检查。只读scholar `nr_semantics_review`（`01a06fd5-67e9-7a72-972b-6c0268f54068`）按3分钟原文核验、2分钟NR复核、1分钟偏离解释，未执行项目负载。模型元数据实际核对为gpt-6-astra/xhigh。其原文核验区分走廊起点与动态入场、指出wall-time歧义；代码复核指出近闭合路线提前到达，主线程修复并回归。最后偏离解释仍标为解释；所有负载和结果汇总由主线程执行。

复用命令（本次预算已结束，下一批按新执行块运行）：

```bash
python3 -B tools/lab.py run --label paper-nr --stage dev \
  --config configs/nr_pilot.json --seconds 90 --disk-mib 256 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  -- "$PWD/.venv/bin/python" -B -m nr_pilot --config configs/nr_pilot.json
```

## 2026-09-05：未来10h双线安排提案（未启动）

**用户请求与建议。** 用户准备授权自主10h，先要求比较既定执行延迟路线与深入复现TR-C2026扰动/非合作路线。建议两条都推进，约80%主要投入名义baseline、20%用于原文机制核验和可独立验证的接口；比例是优先级而非第二个并行训练预算。两线共享BlueSky环境、观测、动作、奖励和PPO。原文也先有名义baseline再评价/训练扰动变体，因此共同前置工作优先。稳定PLAN中的科学顺序不变，本提案等待用户后续启动指令，不把讨论当作10h开始。

**本轮核验。** 已将原PDFpp.8–11、17–25的机制、训练规模、未披露参数和执行延迟关系写入REPRODUCTION。定位/通信/传感器扰动和非合作交通属于原文，风与湍流不属于其实际实验；原文p.25提到延迟执行导致指令过时。FULL/NC额外100k训练episodes意味着后续比较需要追加训练量匹配。作者机构论文条目已访问，有限搜索未核实公开代码/权重；没有联系作者。本轮没有运行项目测试、安装环境或开始实验。

**建议墙钟安排。** 以下时间窗用于调整投入，前置问题未解决时不因到点而跳到后续科学结论。

| 从后续启动起 | 主线程重点 | 可检查产物 |
| --- | --- | --- |
| 0–1.5h | 保存路线定向追踪Amzn偏离、核对入场冲突；补三走廊原生开发覆盖；检查并锁定PyTorch、实际沙箱CUDA/CPU及可用备份 | 根因轨迹与明确重建选择；小诊断结果、真实吞吐与环境锁定记录 |
| 1.5–4h | 接通60动作、5s决策、机动完成锁/目标路线、7/10维观测与CPA、原奖励、逐机终止 | 同链路NR/固定动作/随机策略诊断；构造可解脱场景验证动作有物理效果，同时保留随机人口失败 |
| 4–8h | 共享attention actor-critic与PPO、逐机GAE、保存/恢复；先短pilot测速度，再在预算内训练并用独立开发场景配对NR | checkpoint、学习曲线、动作/奖励分布、安全与任务效率；优先一个训练种子跑通，有余量再独立复跑 |
| 8–9.5h | 有可信名义改善时冻结checkpoint，先做小延迟扫描；时间允许再分开做定位/通信扰动诊断。尚无改善则集中定位物理/动作/观测/奖励/PPO瓶颈 | 与NR和零延迟一致口径的开发对照，或具体无改善证据；不预设必须得到退化/提升 |
| 9.5–10h | 停止新训练、保存恢复点、整理可复现命令/配置/图表/失败与下一步、Git交付和既有渠道通知 | 可继续的研究记录和产物清单；未完成项明确列出 |

**配套复现工作。** 在主线实现或训练时，以有边界的Astra只读调查/独立文件实现推进：把原文扰动定义变成参数表、补齐触发/持续/恢复/重叠的明确选择；隔离物理真值与受损观测，使用独立随机流，确保安全指标仍按真值计算。优先可校验的位置误差与通信中断；传感器三类、静态/动态非合作UAS和CAT/GA先形成准确规范，再按余量实现。零扰动/零延迟须退化到同一个名义接口。此块不承诺训完原文6个专门模型加FULL/NC，更不将单次注入测试称为抗扰策略复现。

**判断与资源。** 不预先承诺10h内250k episodes或baseline收敛。当前NR的8场景rollout约16.79s只说明无策略采样吞吐，不能外推PPO。先做5–15分钟pilot，依据实测决定后续episode数与单job时限；所有项目负载由主线程经lab串行监督，最多一个GPU训练，保存可恢复checkpoint。建议本块新增运行产物软预算≤10GiB，依赖/缓存另计且安装前估算；只读df当前约146GiB可用，不能当作总量硬配额。原有产物不自动清理。重要长训练前需要独立持久副本；若尚无已授权目的地，先做短且可重建的开发工作，不把同盘复制当独立备份。

可信改善至少要在预先固定的开发集合上比较同初态NR，联合检查LoWC/NMAC事件与暴露、flight-hour、完成/失败、航程/延误、控制和越界；不能只看回报，不能通过失败提前退出降低风险分子。模型选择限于开发集，最终测试集保持未看。单训练种子仍只作初步趋势。后续对原文抗扰训练与我们的延迟方法使用同等训练预算，并分开延迟、观测扰动及二者叠加。

后续用户启动时记录实际UTC起止、预算和子agent停止条件；相关实现、诊断与有界修复连续执行，不逐个小步骤再次询问。预留收尾时间，不启动跨过截止的负载。10h内若基线未有效，交付实现/失败诊断和扰动规范，不宣称完成有效基线或推进未经验证的改进模型。

**只读交接回执。** scholar `perturbation_scope`：问题为原文扰动复现的具体范围及其与执行延迟的关系；只读范围为本地原PDF与活动研究记录，禁止代码修改、实验、通知和嵌套委派；来源重点pp.7–11、17–25；输出机制/参数/未知项、可并行准备与训练依赖；≤4分钟。线程`01a06fe6-e5e9-7c62-8dfc-34d89ba6fa0e`实际turn_context核对为gpt-6-astra/xhigh。返回上述来源支持的清单，无修改或测试；下一步为主线程整合计划，本轮已完成。

## 2026-09-05：用户要求的GPU/CUDA连接检查

用户确认既定主线优先、深入扰动复现按进度安排，并要求先检查GPU和CUDA。本轮仅做连接/计算诊断，不启动10h、不安装PyTorch、不训练。主线程本地完成，没有新委派。

**定位与证据。** 普通受限命令中nvidia-smi退出9，/dev下没有nvidia设备节点；但/proc与/sys可见已加载驱动580.173.02，PCI可见NVIDIA显卡，nvcc报告12.8/V12.8.61。随后经自动审查获准的主机只读nvidia-smi查询成功：NVIDIA GeForce RTX4090，驱动580.173.02，显存24564MiB，空闲3492MiB。故初次错误是当前执行环境没有暴露设备，不能推导为主机驱动损坏。

主线程串行调用三次launcher，固定源码快照和Bubblewrap隔离始终保留：

1. [普通受限GPU启动](../runs/20260905T050428Z-gpu-visibility-1e787327/status.json)：20s/16MiB预算，supervisor检测不到GPU设备而拒绝，pid为空，负载未开始。这是保留的失败。
2. [获准主机环境内的lab查询](../runs/20260905T050512Z-gpu-host-sandbox-d2575898/log.txt)：20s/16MiB预算，沙箱内nvidia-smi成功，型号/驱动/显存与上方一致；约0.057s job。
3. [CUDA实际计算](../runs/20260905T050609Z-cuda-kernel-probe-20d365fd/log.txt)：45s/32MiB输出/8192MiB每进程虚拟地址空间预算，约0.911s job，编译并执行[极小核函数](../tools/cuda_probe.cu)。cudaMalloc分配16KiB输出，GPU计算4096个3*i+7，显式同步并回传与CPU逐项比较，mismatches=0；这16KiB不包括CUDA上下文本身开销。device_count=1，compute capability8.9，driver API报告13000，Runtime报告12080（实际使用12.8）；分配前可用显存3251044352 bytes。未测持续负载、PPO吞吐或模型容量。

启动方式是执行工具的require_escalated获准主机权限，再运行原有lab --gpu；不是sudo、修改设备权限或取消沙箱。没有更改驱动/Toolkit、lab隔离实现，也没有终止或调整任何已有GPU负载。当前卡已有较多显存占用，后续训练需按实际可用量小批量测试，不能预设24GiB全部可用。

项目依赖元数据只读检查确认BlueSky1.1.1和numpy2.4.6，未发现torch；CUDA原生核函数通过不等于PyTorch/attention反向传播通过。下一步在正式研究准备中通过uv固定兼容PyTorch wheel，并沿同一获准lab路径验证前向、反向和保存恢复；完整10h仍等待用户启动。复用命令见ENVIRONMENT，本轮代码直接以实际CUDA探针验证，未新增镜像式单测或运行无关回归。

## 2026-09-05：用户要求追查GPU占用

05:08–05:09 UTC经获准主机权限只读查询nvidia-smi和指定PID的/proc进程元数据，未运行实验、读取外项目源码/日志或进程环境变量，未终止/调整负载。GPU利用率99%，总占用20556MiB、空闲3492MiB；两个主要进程均属magic，入口scripts/reinforcement_learning/rl_games/train.py，工作目录/home/magic/cxy/IsaacLab_DexAssemble，解释器/home/magic/miniconda3/envs/env_isaaclab/bin/python3.11，由tmux下bash启动。

| PID | GPU占用 | 任务参数 | 并行环境 | 最大迭代参数 | 已运行 |
| --- | --- | --- | --- | --- | --- |
| 3059945 | 9831MiB | Isaac-DexAssembly-GeometryDRSharpaNutFixedTacmapAlignedFlushRobotBaseTest1-Direct-v0 | 1024 | 1000 | 20h22m59s |
| 3062529 | 9771MiB | Isaac-DexAssembly-GeometryDRSharpaNutFixedTacmapAlignedFlushRobotBaseTest2-Direct-v0 | 1024 | 1000 | 20h21m58s |

二者均headless；只输出明确白名单参数，未展开其他任意参数。max_iterations是启动参数，不代表已经完成的进度；没有读取训练日志，因此不能判断还需多久。其余列出的Xorg、GNOME、TeamViewer和其他图形进程合计约770MiB。以上是实时查询快照，不是持续监控结论；后续研究前重新确认可用量。低空项目当前仍未开始PPO或10h研究块。

## 2026-09-05：保留其他GPU进程时的可行性评估

用户询问不停止进程是否能继续项目。05:12 UTC只读检查/proc/meminfo、/proc/cpuinfo、sched_getaffinity和loadavg：AMD Ryzen9 7950X（16核），当前可用32逻辑线程；1/5/15分钟load=2.056/2.121/2.176；MemTotal=97950664KiB，MemAvailable=64591668KiB（约61.6GiB）。普通受限会话的ps只列本命名空间，未将其当作全机CPU进程清单；上述资源数值为瞬时/滑动窗口观测，非预留配额。

判断：已有CPU BlueSky实际运行证据加上当前CPU/内存余量，足以支持继续环境、动作、观测、奖励与小闭环开发；PPO小训练优先验证CPU方案。GPU仍可访问，但05:08–05:09利用率99%且仅约3.4GiB空闲，因此不能承诺共享训练速度或显存始终足够，也不能承诺对其他训练完全无影响。保留其他进程，不干预其任务。

后续沿用单launcher、从当前线程设置和短pilot起步，以实际PPO网络比较CPU/GPU前向、反向和完整rollout吞吐，记录显存峰值；小CUDA整数探针不回答这些问题。计时参考[PyTorch官方benchmark工具](https://docs.pytorch.org/docs/stable/benchmark_utils.html)，不用异步提交耗时冒充已完成计算时间；此文档查阅不代表已安装其当前版本。若共享GPU收益不足则明确选择CPU并记录，不静默回退；大规模训练另按实测安排。本轮没有训练、安装或启动10h研究块。

## 2026-09-05：已授权10h执行块（进行中）

用户明确“开始10h计划”，并强调另外两个GPU进程属于其他人的实验，禁止停止或影响。主线程于05:15:13 UTC开始，截止15:15:13 UTC；14:45:13起预留记录/保存/交付。既定baseline-first与配套原文复现适用，未授权扩大到全规模原文复现或读取最终测试。输出软目标≤10GiB，依赖/缓存另计并先估算；不递归清理任何原有产物。单launcher串行，初期单线程CPU、nice15/idle I/O优先级，运行预算先分钟级。按用户最新约束取消共享GPU对照，CPU wheel与无--gpu沙箱保证不向该卡提交计算；不kill/renice/修改另外两个任务或其工作区。共享主机的CPU/磁盘影响不能绝对保证为零，持续检查主机余量，紧张则减少/暂停自己的负载。

第一批独立交接（均禁止嵌套委派、实验/清理/通知；builder明确共享工作区并保留他人修改）：

- scholar `route_fidelity`：问题为原文走廊约束、未披露转弯语义与可辩护跟踪选择；只读原PDFpp7、14–17和原生BlueSky源码、活动记录；交付事实/缺失参数/候选方案及横移动作风险；≤15min。线程01a06ffe-a137-7c21-842d-f914a73e1517，实际元数据gpt-6-astra/xhigh。
- scholar `observation_reward_spec`：问题为Table2精确特征、CPA、Eq19–22与GAE终止语义；只读原PDFpp5–7、11–16和活动记录；交付逐维可实现规范、未披露项与重建选择；≤15min。线程01a06ffe-d14e-7a43-a134-df5b7a6db9d8，实际元数据gpt-6-astra/xhigh。
- builder `route_trace`：问题为保存seed51005/F025/C04的定向原生轨迹；仅写src/route_probe.py、configs/route_probe.json及必要test_route_probe.py；读取保存场景与nr/performance/原生源码；交付逐步位置/航点/目标和实际朝向/bank/速度/转弯距离/偏离与峰值记录；≤15min，主线程执行。线程01a06fff-1ea2-7d22-9fb1-912236dee3b5，实际元数据gpt-6-astra/xhigh。

主线程负责CPU依赖、资源与备份准备、native环境/动作集成、所有负载执行和结果记录。重要训练的独立备份仍需落实；优先使用已有获准私有GitHub分支保存可恢复的小型研究产物，具体备份内容/大小和远端结果在有实际checkpoint后记录，不把同盘复制冒充独立备份。

### 05:23–06:16 UTC：物理定位、CPU依赖与学习链路

**实际依赖。** uv固定torch2.9.1+cpu，官方175.9MiB CPU wheel，无CUDA依赖。pytorch-cpu显式索引只绑定torch，其余依赖保持PyPI；第一次非显式索引解析使其他包选到该索引，保存中间锁到.cache/uv-lock-before-explicit-20260905.toml后，从HEAD恢复仅本次生成的uv.lock并重新解析，确认无其他索引迁移、原packaging26.3保留。sync时仍只允许已核验的zmq元包构建；全部运行间隙安装，未修改正在运行的解释器。命令、官方来源与恢复方法更新ENVIRONMENT。

**原生转弯实证。** [route-probe-native](../runs/20260905T052301Z-route-probe-native-0c160aa5/artifacts/result.json)单独重放seed51005/F025/C04（时间平移至0），精确再现最大363.146188m、越界14.75s、飞行108.25s。峰值age82.5s，实际heading76.224°/目标108.267°，bank25°、heading rate3.248°/s、半径1422.908m；保存每0.25s CSV。原文未披露bank、路径拐角连接/捕获控制。中心90°理想圆弧的解析偏离R(1-cos45°)显示高速原生圆弧并不自动满足76.2m半宽；内侧满幅偏移再加有限半径更不能假设包含在走廊内。计算只是几何解释，不是实机保证或作者参数。

**已实现。** paper_actions保留绝对60动作、5s接口、native加减速/爬降、45°捕获与miter平移路线；分量锁与容差明确为未披露处的重建选择，未增加速度governor或走廊过滤。paper_observation实现7/10维、当前航迹CPA、水平6000ft邻居、Eq19–22及归一化裁剪计数。paper_environment统一NR与策略的入场/物理步进/终止/真值风险，保留失败及越界；paper_rollout提供开发诊断。shared_ppo为当前邻居单query attention、共享128/128 Tanh、CPU PPO，宽度/归一化/Adam等配置明确。paper_train已实现逐ID轨迹、完整episode GAE、原子checkpoint/RNG恢复及配对开发评价，当前只作literal环境学习诊断。

**Controller实际验证（均单launcher、nice15/idle IO、CPU14单线程，无GPU）。**

| Run | 结果与边界 |
| --- | --- |
| [paper-learning-core-tests](../runs/20260905T053547Z-paper-learning-core-tests-e5baefc8/log.txt) | 81项中2个断言失败：横移符号误写、浮点近零要求绝对精确；保存失败 |
| [paper-learning-core-fixed](../runs/20260905T054007Z-paper-learning-core-fixed-3c1a280f/log.txt) | 修正断言后81项通过，实际执行torch反向及参数更新；不是训练有效性 |
| [paper-env-crossing](../runs/20260905T054030Z-paper-env-crossing-e956cf79/artifacts/result.json) | 构造M100双机交叉：NR/名义2/2、LoWC39.75/NMAC12.25 pair-s；固定上下高度2/2且两项0，无越界，只证明此构造可解脱 |
| [paper-env-population](../runs/20260905T054206Z-paper-env-population-be961306/artifacts/result.json) | seed51001 NR与名义物理指标完全一致且复现旧NR：30/30、LoWC3817.5/NMAC815.5 pair-s、5.166389 flight-hours；random21/30、9超时、27越界，不能据NMAC降低称改善 |
| [paper-lane-m100](../runs/20260905T054532Z-paper-lane-m100-a625f240/artifacts/result.json) | 2km直线右横移1/1完成，但最大90.010m、越界107.25s；末端lane误差0.151m，初始超调不是零误差跟踪 |
| [paper-lane-amzn](../runs/20260905T054829Z-paper-lane-amzn-2f2dd042/artifacts/result.json) | 同类直线右横移1/1完成、最大76.006m、不越界；不同机型不能仅按速度推断失败 |
| [paper-train-tests](../runs/20260905T061237Z-paper-train-tests-e263dd3d/log.txt) | 新8项收集/ID/GAE、实际weights-only checkpoint恢复Adam与全部RNG、配对评价与选模测试通过，1.392s；两轮真实pilot随后启动 |

人口random的F007/Cranfield超时最大20234.45m、F021/M200超时4206.15m，均记录过capture_beyond_leg_end；原生提前切换名义腿和末端LNAV关闭是待核查机制，不能先归咎学习器。继续做7个预声明5NM/90°单机case与末端语义调查；在严重越界/失败未解释前不投入小时级训练或宣布有效baseline。

**追加交接与回执。** 以下线程实际turn_context两轮均核对gpt-6-astra/xhigh，fresh/bounded上下文；builder仅做实现与AST/JSON静态检查，所有测试/实验由主线程执行。

- route_trace（01a06fff-1ea2-7d22-9fb1-912236dee3b5）在完成定向追踪后续接动作：问题为60绝对动作/锁/native45°路线；独占paper_actions.py、同名配置/测试，来源PDFpp14–15及原生源码；交付3文件和语义限制，约20min，实现完成，无实验。
- paper_features（01a07006-e650-73e1-a02b-b7f3fcc07a40）：问题为7/10维和原奖励；独占paper_observation.py/配置/测试；来源observation_reward_spec与PDF，输出纯函数与18项测试，约25min，实现完成。06:12续接7case route_action_study.py/配置/必要同名测试，约20min；来源environment/action回调、原生跟踪发现；不得改主环境或执行负载，当前进行中。
- shared_ppo（01a07007-ab17-7c11-9aae-99bc858955e9）：先独占shared_ppo.py/配置/测试实现共享attention PPO；后独占paper_train.py/paper_train_dev.json/test_paper_train.py实现收集/保存恢复与开发评价，约40min；来源原PDF/REPRODUCTION及当前API；均已交付，仅静态检查，无实验。主线程正在执行两轮literal pilot。
- learning_integration_review（01a07012-cad8-7d00-99ae-b13c0691e4bf）：只读环境/动作/观测/奖励/PPO集成，一轮约15min，指出env应整批验证mask后执行、裁剪计数须标范围，主线程已落实；后续只读trainer/GAE/checkpoint/resume审查进行中。
- route_fidelity原调查与续审返回：先保持literal参考、用几何与tracking而非奖励选择跟踪候选，不将内侧边界内缩静默当数值修正（会改变最小横向间隔）；本轮继续只读调查末端LNAV/25m到达与路线重建冲突，≤15min，无实验。

主线程下一步：完成并审查两轮训练/恢复诊断，保留原始路线问题，依定向几何结果实现明确、可比较的最小执行修正候选；重要checkpoint落实已有获准私有GitHub备份。当前10h仍在进行，截止不变。

### 06:13–06:18 UTC：首次真实PPO两轮pilot

[literal-ppo-pilot](../runs/20260905T061300Z-literal-ppo-pilot-ba54a8c6/artifacts/result.json)预算300s/256MiB/4096MiB，实际job67.98s、CLI66.17s；两轮训练seed610000/610001，4524/4247逐机转移、71/67个minibatch、各K=1，单轮7.730/7.589s，真实参数更新。初始和终末仅用预声明开发前2例53001/53002（3/4走廊），同初态NR60/60；初始sample42/60、两轮后40/60，越界48/54架次、最大74.06/75.17km。选模保留第0轮，说明best不等于达标。学习轮数太少且物理链路有缺陷，既不能论证原文方法失败，也不能用低NMAC率论证成功。

latest.pt 664620字节（内嵌第0轮best）、best.pt 198201字节；固定副本及开发/训练记录准备到checkpoints/literal-pilot-20260905，manifest含SHA/源run。当前只是同盘副本，尚未验证远端持久备份。随后启动跨进程续到3轮，与连续3轮比较，专门检验native重置/续训状态，不是扩大科学训练。

只读trainer审查回执：未发现阻断收集/GAE/恢复的实质bug，指出假环境恢复单测不能替代native跨进程证据，已据此安排上述对照；无文件改动或测试执行。route_fidelity中间调查将seed51001的9超时分成7个Mnet/Tecnalia低速真实任务超时（累计航程8.54–8.95km、尚差终点0.69–1.44km）与2个严重远飞；不能将全部超时归为末端LNAV问题。

追加builder route_trace（同Astra/xhigh线程）只写action_failure_probe.py/对应配置，重放原random人口并仅追踪F007/F021；依据现env/actions与原native WP代码，输出逐步LNAV/最后WP/终点距离/动作变更和旧结果复现检查，约20min静态实现、不运行实验，主线程独占验证。

### 06:17–06:21 UTC：native跨进程续训一致

[literal-ppo-resume](../runs/20260905T061702Z-literal-ppo-resume-78630a7b/artifacts/result.json)从两轮checkpoint续到总3轮；[continuous-three](../runs/20260905T061901Z-literal-ppo-continuous-three-fc6bdb9a/artifacts/result.json)从头连续3轮。两者均正常完成，分别约59.62/74.64s job，未同时运行。[独立checkpoint比较](../runs/20260905T062109Z-literal-native-resume-compare-fe7da12f/artifacts/result.json)实际weights-only加载两份产物，model、Adam、全部RNG、计数器、配置/源码身份、最佳checkpoint及科学开发评价逐项完全一致；仅排除评价phase及实际墙钟/RSS。比较脚本src/checkpoint_compare.py随源码保存。该证据覆盖真实BlueSky跨进程reset/续训，仍不证明策略有效。

准备将小型恢复点checkpoints/literal-pilot-20260905（2轮latest、0轮best、3轮resumed及配置/指标/比较证据）随当前已授权私有GitHub分支提交；原始runs不加入Git，重复continuous-three.pt不加入Git。06:16主机load约2.39、MemAvailable61.42GiB、磁盘可用145.08GiB；未向GPU提交计算或操作其他实验进程。

### 06:23–06:29 UTC：7单机路线结果与远端恢复点

[route-action-seven](../runs/20260905T062257Z-route-action-seven-725d5871/artifacts/result.json)实际7例全完成、无超时，11.45s job；这并不意味着走廊可行。两等长腿共5NM、赤道东向后北向90°左转，完整每0.25s原生轨迹保留。

| Case | 最大偏离m | 越界aircraft-s | 到达s |
| --- | ---: | ---: | ---: |
| Mavic中心 | 12.265 | 0 | 640 |
| Mavic内侧 | 88.473 | 311.25 | 631.75 |
| Mavic外侧 | 88.840 | 324.25 | 652.75 |
| Amzn中心 | 406.142 | 15.75 | 107.25 |
| Amzn内侧 | 474.220 | 23.75 | 105.5 |
| Amzn外侧 | 348.829 | 12 | 108.5 |
| Amzn原生转弯中才内移（t45s） | 406.142 | 15.75 | 107.25 |

所有发出横移动作的case最终锁完成；6/7发生不同程度越界。解析圆弧和原生轨迹一致支持有限turn envelope与边缘目标不可自动相容，不能只加训练来解释。为区分终点采样漏过与末端真实错过，主线程新增route_completion.py纯几何辅助，提供已声明25m端点邻域的段最近距离、以及候选有限出口截面交点（两种不同任务定义）；尚未接入环境或改变成功定义。[route-geometry-tests](../runs/20260905T062811Z-route-geometry-tests-76b091bd/log.txt)8项通过，覆盖精确5NM几何/内外符号及段漏采样、反向/远离出口与高度交点等；不构成执行修正的实测。

**远端备份落实。** 提交82fd7cf0a65c280b0a97bfdb51304c180b08aaa5已push到crafff/low-altitude的既有私有分支research/trc-baseline-system-20260905，随后git ls-remote核对同一完整SHA。包括CPU恢复点latest/best/resumed-three（总1550417字节模型）及源码/配置/原始小型指标；不是全runs/PDF的备份。后续严格恢复须使用该源码版本，后续物理修正不会静默加载不兼容checkpoint。

新增绘图交接paper_features（原线程Astra/xhigh）：只写route_study_plot.py，用上述7case精简坐标JSON生成标准matplotlib SVG/PNG/PDF，展示首段横移/转弯/偏离曲线与有限走廊边界；输入由controller从既存CSV逐样本准备，保存source SHA，不执行绘图或实验，约20min。

### 06:29–06:38 UTC：末端故障证实与显式语义候选

[action-failure-trace](../runs/20260905T062910Z-action-failure-trace-89478f96/artifacts/result.json)完整原30机random重放，各保存参考比较全通过（21到达/9超时、总4673次动作draw、最大20234.451491m），16.00s job。F007在t772.75s/age331.25s由close-and-away分支关闭LNAV，真实最终WP、距新端点68.2716m；F021在t910.75s/age626.75s由passed-leg分支关闭LNAV，距新端点38.3757m。两者均最后capture超出腿末端、lane锁尚未完成；两CSV均0个25m圆被相邻样本漏过的事件，故不是简单采样漏判。关闭后分别记录869.0/573.5s LNAV-off飞行，解释此前远飞尾段。其他7个慢机超时仍单列。

主线程新增可选swept_endpoint/finite_exit及route_exhausted_without_arrival终止（原sampled_endpoint默认保留）、final-leg capture guard、altitude完成需VS<=.05候选，分别在独立配置保存；并在aggregate保留新增失败数。finite_exit使用固定名义出口有限宽高、跨步交点和最终腿进度，不将native passed或无限平面当成功。只读审查指出单腿出生时最终航点已激活必须在首动作插CAP前记账，已在register后latch；还需相关回归与实际变体诊断。[execution-semantics-tests](../runs/20260905T063731Z-execution-semantics-tests-1c323a20/log.txt)修正该进度记账前99项通过、8.219s；不将单测作为变体有效性证据。

追加review交接learning_integration_review：只读上述helpers/env/actions/trainer聚合与3个环境配置，定位误成功/删除/未来信息/mask不一致，≤15min；已有首个进度记账问题回执，主线程落实。route_trace追加vertical_lock_probe实现交接（上次slot满未送达，后续重试成功）：只写独立probe/配置，8个5NM直线Mnet/M100/Cranfield/Amzn×两种完成条件、5s边界反复31/43、每.25s altitude/VS/lock/指令记录，≤20min，无实验/清理/嵌套；主线程独占运行。

### 06:39–06:58 UTC：执行候选、原文来源与高度根因

默认literal完整random重放在 [literal-refactor-replay](../runs/20260905T064232Z-literal-refactor-replay-426c8656/artifacts/result.json)仍通过全部原参考比较，16.06s job；新选项未改变默认物理结果。[路线图](../reports/route-execution-20260905/README.md)已生成并目视检查，输入保留全部原轨迹采样及SHA。

终止候选先使用同一开发seed51001、各30机NR/random：[swept](../runs/20260905T064533Z-swept-terminal-population-d2937dfe/artifacts/result.json) random为21完成/7超时/2导航耗尽；[finite-exit](../runs/20260905T065436Z-finite-exit-population-fixed-3f46a720/artifacts/result.json)为17/7/6；[finite-exit加执行完成修正](../runs/20260905T065556Z-execution-population-5f21eaa9/artifacts/result.json)为19/7/4。三者NR均30/30；有限出口多保留末端约51.75累计aircraft-s（5.180764h对5.166389h）。有限宽高出口会拒绝接近端点但从走廊边界外退出的飞机，不应为提高完成率把它计成功。提前删除失败尾段改变风险分母与后续population/RNG，不能将这些同seed随机运行当完全相同逐机动作的因果对照。

有限出口第一次运行 [ca493efe](../runs/20260905T064718Z-finite-exit-population-ca493efe/log.txt)在结果保存时因NumPy bool不可JSON序列化失败，保留负记录；几何返回统一转原生数值/bool并补实际NumPy回归。[单腿入场进度回归](../runs/20260905T065343Z-terminal-regression-30f9db2a/log.txt)1项、[终止几何含序列化](../runs/20260905T065350Z-terminal-numpy-regression-6368c391/log.txt)5项通过。

[vertical-lock-eight](../runs/20260905T065518Z-vertical-lock-eight-fixed-a518ad01/artifacts/result.json)8例均完成，26.79s job。四机型literal/settled成对反复250/450ft，5s动作时序、原生垂直动力学保留。literal最大高于顶/低于底：Mnet0.321/4.404m、M1006.428/0.761m、Cranfield0/0、Amzn16.283/8.665m；对应settled四例均0越界、无残余VS反向指令。literal分别31/21/0/7次反向时仍有残余VS；Cranfield虽锁释放有3m/s残余，但到下个5s命令时已经稳定。这支持完成条件必须区分高度位置与垂直速度，不能只由锁释放瞬时VS推断实际反向风险。组合population也将高度越界从1063.25 aircraft-s变为0，但随机动作流同时改变，仅作一致性支持。首次probe [2043e497](../runs/20260905T065358Z-vertical-lock-eight-2043e497/log.txt)因原生az/swaltsel在首次physics前尚未创建而失败；主线程改为未定义时null，az列明确是原生命令而非实际有限差分加速度，不填伪零。

来源回执：route_fidelity与observation_reward_spec完成2024论文/配置、BlueSky fork及另外4个作者仓库的只读调查；固定链接和具体限制写入SOURCES/REPRODUCTION。没有2026完整导航源码或权重证据。旧OpenAP Amzn最大44m/s与Table3 196kt不同；旧普通flyby reached每次更新当前TAS对应转弯距离，1.1.1在航点激活时缓存。原文参数不被替换，另建分离敏感性。

新增交接paper_features：只写navigation_sensitivity.py、两配置及对应聚焦单测；问题为上述速度与flyby缓存各自影响；来源7例route_action_study和固定旧源SHA；输出baseline/仅Amzn44mps/仅刷新普通flyby三条独立轴，各串行新Python进程置于同一个outer lab（不嵌套agent），防止性能表单例串扰；≤25min静态实现、不实验。shared_ppo追加只写trainer/对应测试及execution训练配置，将两个明确scope贯穿产物且不放松resume身份；≤20min，不实验。observation_reward_spec追加只读2026 pp.8–11/17–25，对位置噪声与通信中断给出事实/歧义/最小规范和零扰动测试建议；≤20min，不修改/实验。各任务均保留他人文件、禁止嵌套委派。

06:58左右只读thread metadata再核对：paper_features第4turn、route_trace第4、route_fidelity第6、observation_reward_spec第3、shared_ppo第3、learning_integration_review第3，最新实际model均gpt-6-astra/effort xhigh；线程ID见前文。主线程继续本10h，优先收敛执行选择后有界扩大训练，不向GPU提交任务，不提前通知完成。

### 06:58–07:02 UTC：单轴敏感性与下一训练配置固定

[navigation-sensitivity-tests](../runs/20260905T065846Z-navigation-sensitivity-tests-4c498ab1/log.txt)6项通过；[三变体各7例](../runs/20260905T065855Z-navigation-sensitivity-seven-c4774d42/artifacts/result.json)73.27s job全部正常终止。baseline重复此前7例；仅Amzn公开44m/s时Mavic三例完全相同，Amzn中心/内/外/late最大偏离76.028/150.028/76.198/76.028m，越界0/9.75/0/0 aircraft-s；仅普通flyby刷新时Table3 Amzn四例的这些指标未变，Mavic内/外最大89.140/89.909m。实际refresh wrapper/native调用逐例相同且>0、finally恢复均通过，运行中无flyturn模式。来源差异对速度重要，但不消除完整边缘执行问题；不把该旧速度当2026勘误或修改主类型表。

[execution-swept人口](../runs/20260905T065658Z-execution-swept-population-1a632e1d/artifacts/result.json)random21完成/7慢机超时/2Cranfield导航耗尽，无高度越界、最大横向227.028m；guard仅防capture几何超过末端，不保证有限转弯能捕获。与finite_exit组合19/7/4相比，不因完成率稍高而选择较宽松任务定义。具体选择理由见DECISIONS，下一轮使用有限出口execution配置、全部12开发例、25轮/600s内预算；仍是有未解决横向约束的学习诊断。

trainer第二scope传播与严格恢复兼容的 [scope-tests](../runs/20260905T070132Z-paper-train-scope-tests-820f61ab/log.txt)10项通过，2.99s job。shared_ppo仅静态实现三owned文件，controller执行测试，原literal测试保留。read-only reviewer追加navigation单轴与终止候选审查，≤15min，禁止实验/修改，正在返回。07:00普通资源诊断load约3.07、MemAvailable61.35GiB、磁盘可用144.64GiB；受限/proc看不到原两个PID，不据此推断外部训练已退出，CPU-only承诺不变。

### 07:02起：25轮训练进行中及配套来源补齐

主线程启动 [execution-ppo-25](../runs/20260905T070231Z-execution-ppo-25-834d0f28/status.json)，25轮目标、600s内预算/660s外监督、256MiB输出/4096MiB每进程地址上限；继续nice15/ionice3/CPU14。初始全12开发例实际169.28s：NR359/360（seed53008有1导航耗尽），sample239/360（68导航耗尽、53超时）；横向越界16/341架次，最大599.314/618.641m，均0高度越界。首个单seed NR30/30不能推广全开发集无失败。正在学习，不将初始结果当最终结论。

独立reviewer返回：没有阻断当前小PPO诊断的实现问题，静态公式与实际hook/速度核对均通过；刷新无效仅限7个恒速case，不能排除策略加减速时旧缓存影响。保留同seed动作流与失败暴露解释边界，无需据swept完成率较高改选训练定义。无修改/测试。

observation_reward_spec只读规范返回，事实/歧义与具体选择写入 [PERTURBATIONS](../paper/PERTURBATIONS.md)：二维噪声定义、p与events/h差异、blackout重叠与缺失ownship控制均不冒称作者代码。追加paper_features实现交接：仅observation_perturbation.py/配置/测试，纯独立感知层，零扰动不消费RNG、同刻cache、JSON恢复、事件/可见性分离；不改正在训练的源模块，不执行测试/实验/清理，≤30min，返回后controller验证。真实aircraft-s与缺失决策PPO尚不由离散接口伪造。

route_fidelity追加15min只读特刊核验：读PLAN/SOURCES与出版社/客座编辑官方入口，不修改、不实验、不联系他人。返回大学官网明确链接的编辑现行主页仍征稿、截止2026-12-30；出版社征稿列表搜索索引一致，CFP正文/作者指南403。已更新PLAN/SOURCES/NOW，访问日期与deadline分开、投稿系统未核实。当前10h截止15:15:13 UTC不变。

### 07:09–07:12 UTC：首批execution模型与继续预算

[execution-ppo-25结果](../runs/20260905T070231Z-execution-ppo-25-834d0f28/artifacts/result.json)实际15轮，CLI394.76s/job396.60s；因保守的15s/episode初始估计、1.5安全因子及12-case末次评价预留，在轮间停止，非资源中断或25轮完成。末次开发94.45s，sample244/360（63导航耗尽、53超时）、342横向越界、最大618.641m、0高度越界，65.434167flight-hours；LoWC26029.25/NMAC4773.25无向pair-s，即397.793/72.947s per flight-hour。初始239/360、NMAC63.799/h；完成略升而冲突指标较初始恶化，不能判为有效改善。NR359/360、54.908333h、LoWC35604.75/NMAC8838.5无向pair-s（648.440/160.968每flight-hour）。best仍不达标。

源/诊断图提交ea37154366ecc9bba058de35a1ae1e338b9e8380已push，git ls-remote同SHA；新恢复点checkpoints/execution-pilot-20260905保存15轮latest/best及完整小型开发/学习记录，约1.65MB，正在准备独立远端备份。随后计划沿同配置恢复到总100轮、内1800s/外1860s，CPU资源限制不变；实际8s左右/episode支持该有界扩大，25轮评价间隔保留。训练有效性与横向可行性仍分开，尚不开始延迟效果实验。

### 07:12–07:49 UTC：恢复点、中断恢复与扰动接口

15轮恢复点已随私有分支提交6c008e49282849b4791d09f5af3cf8a417989bd4并push，远端SHA核对一致。[100轮目标负载](../runs/20260905T071311Z-execution-ppo-100-620c1921/status.json)于07:13:11启动，实际最后完整训练65轮，最新日志07:26:06；外层session72625返回143，旧launcher状态遗留running且无result.json。**发送者和原因未知，不能据此宣称固定工具时限或100轮完成。** 主线程未发信号；随后确认launcher锁空闲，经获准主机只读精确项目进程审计无仍存负载，才开始新job。原始状态不修改，另存[controller_observation](../runs/20260905T071311Z-execution-ppo-100-620c1921/controller_observation.json)。

该段初始15轮开发科学汇总与上段完全一致。25轮sample248/360（59导航耗尽、53超时），NMAC64.1499、LoWC407.4062无向pair-s/flight-hour；50轮234/360（73导航耗尽、53超时），NMAC66.2211、LoWC378.5755。最好选模25轮，仍未达95%完成门槛；65轮无开发评价。[65轮完整性](../runs/20260905T073653Z-checkpoint-65-integrity-ba7452f1/artifacts/result.json)实际沙箱weights-only读取通过：35325个参数全部有限、completed/next seed index均65、所有科学源码hash一致、内嵌best25。独立小副本见[恢复说明](../checkpoints/execution-recovery-65-20260905/README.md)，待本批push后再续65→100，保持科学源码ea37154兼容，内900/外960s且持续轮询。较短分段是预防措施，不是退出原因诊断。

全开发NR唯一失败另以[seed53008定向重放](../runs/20260905T071053Z-development-nr-endpoint-68b4ebc4/artifacts/result.json)确认：F029/Amzn无策略命令，age114.25s、t473s导航耗尽，终点距离94.016m、最大偏离599.314m、越界61s。这是当前路线执行问题，不能归因学习。

**扰动实测。** [纯层15测试](../runs/20260905T073355Z-observation-perturbation-tests-8c19ee06/log.txt)通过，包含零RNG消耗、共享噪声/可见性、严格同刻缓存、重叠/到期、JSON恢复与无效输入原子性。[5例native诊断](../runs/20260905T073432Z-perturbation-native-probe-b8a3c19f/artifacts/result.json)9.84s通过：plain与zero的编码/掩码、动作、逐物理轨迹和科学汇总完全一致；100%通信中断时0次策略推理、46个逐机保持决策、226.5真实不可用aircraft-s，NR轨迹/终态/真值风险完全一致，LoWC39.75/NMAC12.25无向pair-s仍记录。plain/zero/position100%各52次推理、LoWC26.5/NMAC4.0；该未训练小模型在此例动作未变，不能称定位鲁棒性。模型只初始化一次，未训练、未加载checkpoint，不改变现有PPO或GAE语义。

**新增交接与实际回执。** 各任务仍禁止嵌套委派；builder不执行负载/清理/通知且保留他人修改。07:48只读实际thread元数据：paper_features第7turn、route_trace第6、route_fidelity第8、observation_reward_spec第3、shared_ppo第5、learning_integration_review第5，最新均gpt-6-astra/xhigh，线程ID见前文。

- shared_ppo：只写learning_curve_plot.py，读取开发JSON格式和报告约束，交付6面板未平滑静态曲线模块及严格输入核验，≤20min，已完成静态检查未渲染；随后仅写tools/lab.py、test_lab.py，问题为可捕获SIGTERM留下状态，交付flag handler、安全边界、只回收自有Popen组、恢复旧handler与3回归，≤20min。主线程[lab-sigterm-tests](../runs/20260905T074648Z-lab-sigterm-tests-befad413/log.txt)全部26项通过，10.74s，包括仅对随机临时fixture自己启动的supervisor发SIGTERM、子进程不再延迟写入和锁可重用。不处理SIGKILL或保证任意外部终止能清理。
- learning_integration_review：只读上述两个launcher变更文件/差异与中断旁注，问题为自有清理、异步flag和恢复handler是否正确；≤10min，独立review已返回无阻断问题，未运行测试或发信号；真实fixture readiness只来自父payload，不单独证明SIGKILL升级分支，当前亦不作该断言。
- paper_features：此前纯扰动层完成，当前仅写exposure_audit.py、对应配置/聚焦测试；问题为原文potential/LoWC/NMAC时长与小时分母；读取原PDF pp.11/17–18及15轮备份首行NR，输出原生6000ft潜在pair-s、完整/占用airspace-time、aircraft-time、真实构造/推理次数、12例及5走廊子集加权汇总，≤25min，controller执行。要求只旁路观测、不改物理/训练，风险与原参考精确核对。
- route_trace：仅写perturbation_probe.py/配置，问题为零扰动原生不变和缺失ownship的明确保持目标；来源纯层规范/现env API，交付上述5case与真值不可用时长积分，≤25min，静态完成，controller已执行。
- route_fidelity：只读原PDF图10/指标与保存NR JSON，≤15min，返回下列口径事实/歧义，不改文件、不实验。下一步是原生potential及双分母审计，不能直接校准到论文数值。

**指标来源核验。** PDF p.17称累计各受控agent约1400 flight-hours，p.11按time×ownship×intruder累加；图10标注potential/LoWC/NMAC每小时NR为34257/11791.77/3193.22，训练为498/6.88/0.59。但印出的极限式没有明确累计aircraft-time除数，图轴与标注斜率不能唯一反解；没有原始曲线数据，不能猜一个修正系数。正文6000ft与Fig4的6080ft亦不一致，当前采用正文。34.42%/9.32%等是成对暴露比，不是独立相遇事件概率；多邻居pair-s/flight-hour可以大于3600。

首行12例NR累计197670 aircraft-s、17265完整airspace-s，平均同时在场11.449。LoWC/NMAC有向71209.5/17677pair-s，除aircraft-hours为1296.880/321.937，除airspace-hours为14848.202/3685.908。现有4个5走廊case分别为951.031/229.710和11163.735/2696.469。后者靠近论文标注只构成待核验分母假说，不证明作者使用airspace-hours，不据此调场景或宣称复现吻合。后续所有比较保留原始分子、两类小时分母及完成/失败/越界。

本10h仍进行中，截止15:15:13 UTC、14:45收尾不变；继续CPU14/单线程/nice15/idle IO，不向共享GPU提交任务，不干预其他人的进程。

07:50 [首张学习曲线](../reports/execution-learning-20260905/README.md)已通过lab实际渲染（3.28s）并目视核对，保留0/15/25/50轮原始未平滑开发汇总；重复15轮aggregate精确一致后合并一次，没有伪造65轮评价。当前形态显示完成率/越界无稳定改善，低于NR的冲突暴露在第0轮就存在，不能当训练增益。

07:53 本批提交7e38f8f74a854d04938109aebe4c6d5e4cabd7c9已push且远端SHA核对一致，65轮模型得到私有GitHub独立副本。新源码/文档diff检查通过；Matplotlib原始SVG含其生成的行末空白，保留原图以匹配产物hash，不称全文件零空白警告。随后以原科学源码/配置启动65→100段（900s内、960s外），CPU资源边界不变。route_trace追加只读新训练日志/0–50开发动作与失败统计调查，≤15min，不重复全面GAE审查；仅建议有辨识力的下一诊断，不因曲线噪声直接调参。

07:56 route_trace只读新日志调查返回：0/15/25/50轮每case超时数均不变，总53；完成变化全部对应导航耗尽68/63/59/73。开发平均请求速度比例0.8400/0.8419/0.8402/0.8413，训练65轮四速度动作占比23.89/25.43/26.15/24.53%。Mnet/Tecnalia名义8.6427m/s，5NM名义约1071s，均匀四档均值.8375对应约1279s，超过1200s；按直线平均至少.8929才能完成。这是低速超时机制假说，现JSONL不含逐机类型，不能把53个全部指为这两类。

65轮258667样本/4077minibatch，动作计数/逐轮访问数一致，回报误差约7e-12。条件熵2.075–2.167与跨状态动作直方图熵4.061不同，不能用前者除log60认定塌缩；当前缺valid_count。minibatch更新前KL约2.0e-6–2.26e-5、clip fraction0，不等于更新后全批KL。25→50完成少14但总回报改善135.61，原奖励无直接越界/失败惩罚，可能存在提前结束负奖励的激励，但尚未证实策略利用失败。

追加route_trace仅写policy_diagnostic.py/对应配置/聚焦测试：对明确checkpoint及同轮reference的原12开发例只重放sample，单次真实forward hook记录合法动作数、条件熵/归一熵、分量概率、类型、逐机奖励三项与最终失败；严格原源码身份、原种子与完整sample摘要逐项一致，模型前后不变，NR仅引用原记录。≤30min静态实现，controller待当前训练结束串行运行≤240s，不改变当前训练。另observation_reward_spec只读补其余传感器/非合作机制参数与缺项，≤20min，不重复已完成位置/通信审计，不接近邻创新或实验。

08:02 observation_reward_spec只读其余机制返回，已将具体页码、字段裁剪顺序、完整邻机行排列不改变attention、NC角色/风险分母及CAT-GA参数缺项写入REPRODUCTION。没有新增实现/实验；源事实与候选选择分开，下一步优先完成名义学习与当前诊断，复杂NC不抢先替换有效baseline目标。


### 08:03起：总100轮完成及进一步瓶颈诊断

[65→100续训](../runs/20260905T075230Z-execution-ppo-resume-100-7df1b66d/artifacts/result.json)正常完成35轮，CLI648.985s/job650.817s，episode_target_reached，总100、best仍25；全程持续轮询、CPU14低优先级。恢复65轮评价237/360、75轮239/360、100轮236/360，均53超时；100轮71导航耗尽、340横向越界、0高度越界，NMAC56.1609/LoWC366.5960无向pair-s/flight-hour，总回报−2000.4135。完成/越界仍不可信，风险减少不能独立作为baseline有效；本次正常退出也不证明之前外层143的来源。

100轮小恢复副本[execution-100](../checkpoints/execution-100-20260905/README.md)含latest/best、完整小开发/训练/result与hash，约1.77MB，正在私有远端备份。保持ea37154科学源码身份，下一目标150，内1200/外1260s、25轮评价和5轮原子保存不变；不打开held-out或提前做延迟结论。

[exposure-audit-tests](../runs/20260905T080411Z-exposure-audit-tests-5232b1b4/log.txt)12项通过；root完成所交三文件窄静态review后，启动180s native原12例NR审计，不与训练并行。builder未执行负载。约07:53资源只读load2.78/2.48/2.52、MemAvailable60.85GiB、磁盘144.18GiB；本块当时39个run目录逻辑合计0.611GiB（含source/artifacts，依赖另计），仍在10GiB软目标内，不因此扩张共享CPU/GPU使用。

08:06 [exposure-audit-nr](../runs/20260905T080439Z-exposure-audit-nr-8d220f91/artifacts/result.json)88.46s正常完成，12例全部检查通过，LoWC/NMAC原分子、事件数及每例全部科学NR摘要精确复现；只有runtime/RSS排除。6000ft潜在230361.75无向pair-s、1516连续事件；LoWC35604.75/536，NMAC8838.5/362。累计197670 aircraft-s、17265 airspace-s，二者小时口径及有向乘2明确。全12 LoWC/potential15.456%、NMAC/potential3.837%，原有4个5走廊例11.988%/2.896%；与论文34.42%/9.32%差异不能用共同时间分母或共同pair因子抹平，不据较近的两项airspace小时数宣称辨认作者分母。

构造观测3827帧、44421 ownship/103090 intruder entries，返回reset/decision3471帧、39557/92202 entries，NR实际推断0。终止预览构造、原生暴露、政策输入三个数量不可互换。小完整报告及命令另存[exposure审计](../reports/exposure-audit-20260905/README.md)，逐步CSV留原run。route_fidelity追加只读新结果/原pp11/17–18，≤10min，核对potential精确定义与共同缩放不能解释的差异，不调参/实验/联系作者。
