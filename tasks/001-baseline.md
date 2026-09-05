# 001 — 原文式无延迟、无扰动基线

状态：`doing`（12类性能、场景、动作/观测/奖励和共享PPO已接通；正在验证小训练及路线执行缺陷）。负责人：主线程。目标顺序由用户确认：先有效基线，再冻结后加延迟。当前尚无有效新基线证据。

最新范围：本10h科研与交付已结束，授权窗口为2026-09-05 05:15:13–15:15:13 UTC；14:45:13起收尾，后续研究需新授权。用户要求不停止/不影响其他人的两个GPU实验，当前块使用CPU且不提交GPU负载；执行预算与交接见末节。此前15分钟预算均已结束。

当前可恢复结果：新refresh累计850轮、保存best775，仍无有效baseline。最终850sample233/360完成、340架次横向越界；best775为260/360完成、336越界。35点曲线、48例解码重放、F0203188行判据诊断均已完成；当前只做收尾复核和私有备份。入口：[NOW](../paper/NOW.md)、[850恢复点](../checkpoints/refresh-850-20260905/README.md)、[F020机制与下一配对方案](../reports/lane-completion-f020-20260905/README.md)。实际运行与交接在本文件末尾14:22–14:43节；后续训练需新的授权块。

此前04:29执行块已结束：用户再次明确“开始”，2026-09-05 04:29:42 UTC起，预算至04:44:42 UTC、≤15分钟/≤2GiB输出；12类性能、论文式场景生成/入场退出、小批无避让诊断，不训练。6个串行job最后于04:41:05 UTC结束，约11分23秒内完成；保存80,354,700字节产物（约76.6MiB），随后仅记录和Git交付。两个具名Astra/xhigh builder分别只写paper_performance及机型配置/测试、paper_scenarios及NR配置/测试；只读scholar核对原PDF场景语义。主线程写NR执行器/事件统计、集成并独占实验执行。

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

08:09 100轮恢复点与NR暴露审计随bd1ae0c99771ee7dd31f7a439235275afcb28152推送且远端SHA一致；doctor与本批diff检查通过。随后启动100→150（内1200/外1260s），继续同一配置、源码和CPU14限制，当前唯一launcher负载。

08:10获准主机权限只读nvidia-smi复查：原两个GPU PID3059945/3062529均仍列在compute-apps，显存9891/9771MiB；RTX4090总用20616MiB、余3432MiB、利用率98%。未读取外部项目文件/日志/环境，未发信号、调整进程或向GPU提交计算。该快照支持二者仍运行，不构成对其吞吐完全无影响的证明；本项目继续单核低优先级CPU。

08:17 policy_diagnostic builder已交付3个owned文件（56列流式CSV、4个纯夹具、同轮checkpoint/reference严格核验）；静态AST/JSON通过，尚未运行。独立reviewer窄审无阻断：C-order 4/5/3与名义37正确，条件熵/掩码归一熵/混合熵/经验熵分开，原forward/passive hook不追加随机draw，奖励按实际sampled ID记账并与逐机return对齐。归一熵排除valid=1且保留计数；分组仍是决策加权关联，不是时间占用或因果失败利用。当前训练结束后controller先4fixture再全12例100轮native精确对照，≤240s。

route_fidelity针对新exposure返回：共同乘数若对齐论文，potential/LoWC/NMAC分别需4.0827/9.0924/9.9188，不是同一个单位转换；返回邻机entries92202×5s=461010，与真实有向potential460723.5仅差0.062%，不支持简单5s/0.25s解释。p11没有明确CPA、接近方向、最近邻上限或同走廊排除；不补隐含筛选。已从固定run复制24个scenario/events来源及SHA至报告pair_input，root写同/跨走廊纯重聚合脚本，逐例断言事件时长/计数还原原暴露；仅AST完成，待训练间隙通过lab执行，不修改物理或训练源码。

08:18实际thread元数据再次核对均gpt-6-astra/xhigh：paper_features7turn、route_trace8、learning_integration_review7、observation_reward_spec5、shared_ppo5、route_fidelity9。追加shared_ppo只写ppo_gradient_probe.py/对应配置/聚焦测试，问题为既有低KL/频繁裁剪是否伴随共享梯度冲突；基于严格100轮副本和下一training seed610100，独立收集一完整batch，分开actor/加权critic共享层梯度范数/夹角与全batch KL(old||new)，原update只作用于可丢弃模型/Adam，绝不保存晋升或改变主训练。≤25min静态实现，controller后续≤120s串行执行；此测量不自动支持调参或因果归因。


### 08:22起：总150轮与策略诊断

[100→150](../runs/20260905T080929Z-execution-ppo-150-493a113d/artifacts/result.json)完成50轮，CLI774.236s/job776.085s，正常exit，best仍25。125轮sample227/360（53超时、80导航耗尽、347越界）、NMAC66.597/h；150轮230/360（53超时、77导航耗尽、344越界）、NMAC63.0828/LoWC365.7151无向pair-s/flight-hour、return−2081.0070。100轮初始恢复评价保持同科学汇总。当前没有足够任务改善，继续同配置学习且同时诊断，不据噪声直接改奖励或学习率。

[150轮副本](../checkpoints/execution-150-20260905/README.md)约1.802MB，尚待本批远端备份。计划150→250内1800/外1860s；已实测50轮约13min，扩大为100轮仍单CPU/单launcher、每5轮原子保存并持续轮询。先执行当前有界诊断，不并行实验。

[policy-diagnostic-tests](../runs/20260905T082304Z-policy-diagnostic-tests-32ea4b80/log.txt)4项通过，实际测试1.43s job。controller随后启动指定100轮checkpoint+同轮reference全12原生重放，≤240s/128MiB/4096MiB；独立review结论已记前文。梯度探针将保留完整collection与original update字段，便于同已完成150段首行真实episode101/seed610100比较，检查诊断不改变原采样/update语义。

08:26 [policy-diagnostic-100](../runs/20260905T082427Z-policy-diagnostic-100-96e8dc93/artifacts/result.json)110.66s正常完成，所有12例科学sample摘要/动作直方图与原100轮精确一致、model hash不变、0梯度、hook已恢复。3770真实forward、47167逐机采样决策，CSV25694555 bytes留本地。条件熵均值2.125525、H/log(valid_count)=.991783；合法数4/20/60分别25581/17892/1688次，其余8/12/40共2006次。策略仍近似合法支持上的均匀分布，不能把约2.1熵认作塌缩。

全部53超时确认为29 Mnet+24 Tecnalia，各自0到达；两类策略期望速度比.840643/.842913（全局.842507），与前述1200s可达性机制相符。到达/超时/导航耗尽组平均age569.70/1200/529.85s、return−4.583/−10.050/−5.441；耗尽组平均回报反而比到达组差，这些非匹配分组不证明主动利用失败。安全总项−1948.1603、未缩放eff−36031.65×.008、arrival236与总−2000.4135逐机对账。完整结果/身份小报告见[策略诊断](../reports/policy-diagnostic-100-20260905/README.md)；5.11MB JSON含完整分组，原25.69MB CSV/逐机原始文件仍本地。

[事件重聚合](../runs/20260905T082648Z-exposure-pair-decomposition-aa29f123/artifacts/pair_decomposition.json)0.066s通过每case原始暴露/连续事件计数还原断言；同走廊占potential/LoWC/NMAC的60.91%/81.40%/92.90%，现有5走廊子集44.61%/74.89%/89.59%。记录共同航路主导当前NMAC，但标签自身不证明每例追赶/几何交叉。脚本、24来源SHA输入及结果补进[NR报告](../reports/exposure-audit-20260905/README.md)。这些诊断不改变现有PPO源码、任务定义或训练随机流。

08:32 150轮恢复点、策略诊断和同/跨走廊事件分解随4c178fbab6bdb052a9c6f1ce50badae94e9ccf86推送私有分支，远端SHA一致；本批doctor/diff检查通过。随后启动150→250（内1800/外1860s），当前唯一launcher负载，仍CPU14/单线程/nice15/idle IO/4096MiB虚拟地址上限，每5轮保存，持续轮询；10h截止15:15:13、14:45收尾不变。

08:50左右250目标仍进行，175/200轮sample完成233/241（均53超时），NMAC64.824/67.699每flight-hour，尚无稳定任务改善。梯度探针3文件静态完成，独立review无阻断；待当前负载结束串行验证，再与原episode101（4889样本/77minibatch/1epoch）科学摘要及全部ppo字段比较，排除runtime/RSS/collection_wall_seconds。另observation_reward_spec只读自身7维对下一转弯的可观察性及原文敏捷假设，≤15min；不得把瞬时观测混叠直接声称无可行控制或论文创新，不改当前训练。


### 08:55–09:07 UTC：250轮正常完成、梯度原生精确对照

[150→250](../runs/20260905T083233Z-execution-ppo-250-dfe44090/artifacts/result.json)完成100轮，CLI1393.066s/job1394.878s，08:55:48正常退出，best仍25。175/200/225/250轮sample完成233/241/242/244，NMAC64.824/67.699/76.938/73.208每flight-hour。250轮53超时、63导航耗尽、347横向越界、0高度越界，64.992847flight-hours；LoWC25030.75/NMAC4758无向pair-s，LoWC385.130842/h，return−2300.19648。仍未建立有效baseline，不能按较低于NR的风险单项冻结。小副本[250恢复点](../checkpoints/execution-250-20260905/README.md)5份数据2011252字节，记录原run/源码身份与SHA，准备私有远端备份；下一段同科学配置计划250→400，内2400/外2460s，仍每25评价/每5保存，CPU14单线程低优先级，10h截止不变。

[梯度5测试](../runs/20260905T085659Z-ppo-gradient-tests-3d751a9e/log.txt)通过，1.366s测试/2.890s job。[native探针](../runs/20260905T085732Z-ppo-gradient-probe-dd663a67/artifacts/gradient_probe.json)17.229s内部/19.005s监督正常完成。严格100轮checkpoint恢复下一training seed610100，4889样本、77minibatch、1epoch；controller与已完成150段首行episode101比较，除wall_seconds/collection_wall_seconds/process_peak_rss_mib外全部episode科学字段及整个ppo dict完全相同，包括动作直方图。比较源hash/行号和完整结果另存[梯度报告](../reports/ppo-gradient-100-20260905/README.md)。

原权重全rollout共享actor/weighted-critic norm .0247298/1.0266925，cos+.050667；原首64样本为.1536491/1.2471115，cos−.079015。两个静态组合均触发global .5裁剪。原update在可丢弃model/Adam副本后，全batch KL(old||new)均值3.60475e−5、max.000132381；同固定returns的explained variance .330459→.332767。参考model/Adam/checkpoint未改变，更新副本未保存。大范数与近正交只描述原权重两个测量，不等于77步Adam方向或已证实critic妨碍学习，不据此直接改参。独立reviewer此前静态无阻断；追加≤15min只读实测解释核对、最多2个有依据后续，不修改/实验/嵌套。

observation_reward_spec只读镜像可观察性回执：5NM首东4630m、后南/北4630m，首段中点同状态且未原生换航点，own7=[.25,.25,nominal_speed/(196kt),.5,.5,0,0]，Mavic速度维1/7、Amzn .8；无邻机、60动作全合法，任何同权重策略分布相同。底层_route知道完整未来航路，这不证明没有共同可行控制。先前Mavic inner/outer完成动作90/92.75s时已78.199m>76.2，早于原生首航点换315.75/326.25s；需区分首段capture容差与转弯。原文pp14–17敏捷few-seconds未披露bank/join/tolerance，不擅自赋瞬时航向或旧44m/s。

追加builder route_trace：仅写route_observation_probe.py/对应配置/聚焦测试，不改core训练源码/既有配置；2例Mavic中心+24例Amzn左右转×4速度×3lane，名义高度、先NR到首段中点的首个5s边界再重复固定合法请求，保存触发时真实own7/mask/固定未训练模型输出、capture/转弯phase、越界/终态/原生导航摘要。可另列24例Amzn仅普通flyby刷新轴，总50例；保留Table3速度/宽76.2/bank25/1200s。≤30min静态，无实验/清理/信号/GPU/通知/嵌套且保留他人工作；controller后续≤180s串行验证。有限动作响应表不是所有时变策略可解性证明。

0–250学习图输入已从五份备份development.jsonl及各manifest重组，13个原开发评价点；重复15/100/150全sample和NR aggregate逐项相同才合并，所有源SHA/行号保留。65点评价来自后续恢复初评，原中断run仍无65评价；旧图覆盖范围不冒充最新。


09:08提交6c9b40f8a8d4202633b34512f5b8c299a5a57ea3已推送私有分支且ls-remote同SHA，250恢复点、原生梯度报告及0–250图已备份；图生成job3.30s，PNG已目视核对，doctor无问题。diff检查排除原样保存的生成SVG（含绘图库尾空白），其余通过。随后[250→400](../runs/20260905T090835Z-execution-ppo-400-95effea7/status.json)于09:08:35开始，内2400/外2460s，当前唯一CPU14 launcher，不向GPU提交负载。

梯度独立review仅算术/只读，未改或跑实验：两处g_actor·(g_actor+g_critic)均正，普通梯度下降下合成方向仍降低actor局部目标，不能推广Adam。entropy/actor norm仅.0239%/.00350%，不支持高熵由熵项压倒actor。追加shared_ppo仅写独立probe/test，≤25min静态，在同一可丢弃原update测首个真实Adam参数位移/与actor和critic的投影及固定first64目标变化；保持整77步原update科学参考精确比较，不重复归一first64或改变主线。

辅助机制交接paper_features：仅新encoded_sensor_faults.py/对应配置/测试，≤25min静态；基于已核原pp10–11与REPRODUCTION，实现归一化之后的确定性显式fault plan，缺失有限2、异常由调用者给有限越[0,1]值、至少2指定scalar的显式permutation。无事件率/异常分布猜测，无随机采样/训练/NC/GAE接入；保留shape/row ID/mask/真值且提前全量验证、零fault逐bit不变，记录计划项与实际数值变更数量。完整邻机行置换不能充作有害扰动。禁止实验/清理/信号/GPU/通知/嵌套且保留别人修改；controller待主训练间隙独占验证。这是论文机制接口补齐，不是抗扰训练结果。

09:12只读相应会话turn_context实际元数据复核：route_trace第9turn gpt-6-astra/xhigh、shared_ppo第8turn gpt-6-astra/xhigh、paper_features第8turn gpt-6-astra/xhigh、learning_integration_review第9turn gpt-6-astra/xhigh。仅读取指定研究子线程的模型字段，不复制认证配置。


### 09:26 UTC：400轮段中间值与独立诊断准备

275轮sample240/360，53超时、67导航耗尽、342横向越界，NMAC73.114952/h；300轮244/360、53超时、63导航耗尽、345横向越界，NMAC69.751791/h。恢复250初评的整个sample/NR aggregate与保存参考精确相同，仍无有效baseline。此处不以100轮的高熵外推300轮策略分布。

镜像50case实现/AST/JSON已交付，尚未执行3项fixture或native。controller核对配置/测试，独立reviewer只读核心未发现阻断；pre_turn_controls_exact跨native/refresh的24项仅作描述未进入总flag，后续若作同状态刷新效应解释须逐项确认。刷新从t=0启用；同arm左右turn输入/真实状态相等已有总检查。CAP不会推进名义阶段，真实下一航点2首次映射后锁存；最终任务状态由原环境记录，固定响应不能证明不存在可行时变策略。

Adam step1新增仅probe/test已静态交付，controller与独立reviewer未见阻断：额外可丢弃原update只被动拷贝首步前后参数，完整77步metrics/final模型/Adam/shuffle须与未观察路径精确相同，再测固定64目标/KL；全rollout优势不在64上重归一。8项测试/native尚待单launcher间隙；首步投影不是critic因果贡献，64样本1步KL不可直接与4889样本77步KL推断累积率。

encoded_sensor_faults三文件及16fixture完成，仅AST/JSON；controller只读API核对为先全量校验后独立deepcopy，异常在目标dtype中仍须有限且越[0,1]，显式scalar permutation读原clean snapshot，重叠计划拒绝，完整邻机行置换单列为attention集合不变。追加paper_features仅新encoded_sensor_probe.py/config，≤20min静态；原2-M100 crossing上plain/empty/disabled/missing/abnormal/falsified共6例同未训练模型，零/关闭与plain编码-动作-物理-科学摘要精确比较，缺失own速度2、异常−.5、own速度与高度交换显式方案。被动hook核对真正输入、有限合法logits和值，保留真值风险，模型/源输入/mask不改；不加入事件率猜测或训练。实际待controller≤90s。

route_fidelity只读原p6§3.4/Algorithm1确认训练采样；p17§6.1frozenpolicy/MonteCarlo未规定test sample/argmax。现配置sample主指标/report_argmax=false，原_evaluate_case支持argmax，直接改配置resume应严格失败。追加route_trace仅新checkpoint_policy_modes.py/config/test，≤25min静态：对未来400副本做严格来源/配置核验，原12例sample实际重放与400科学参考精确比较，再独立argmax；另按原初始化seed/config构造0轮模型并复现已存0轮sample，随后argmax，共最多48例，NR仅引用精确同参考。总inner570/outer600s，0优化器/更新/保存/选优，不因argmax较好替换sample主指标。源论文的实际test选择仍未知，无法称argmax是作者要求或仅降低评价方差。

09:29获准主机权限只读nvidia-smi：原PID3059945/3062529仍分别9891/9771MiB，RTX4090总用20616、余3432MiB、GPU99%。未发信号或调整外部进程，也未向GPU提交计算；快照不证明外部吞吐零变化。CPU load约3.29/3.38/3.21、MemAvailable约60.81GiB、磁盘143.65GiB；本项目仍CPU14/单线程/nice15/idle IO/单launcher。


### 09:41–09:44 UTC：400轮恢复点与串行原生诊断

[250→400](../runs/20260905T090835Z-execution-ppo-400-95effea7/artifacts/result.json)正常完成150轮和终末全12评价，CLI1983.854s/job1985.640s，09:41:40结束，总400、best仍25。350/375/400完成231/235/236，均53超时；400轮71导航耗尽、343横向越界、0高度越界，64.58flight-hours；LoWC23348.5/NMAC4300.75无向pair-s，对应361.543822/66.595695每flight-hour，return−2132.924009。无持续任务/containment改善，不冻结或开始延迟效果结论。[400小副本](../checkpoints/execution-400-20260905/README.md)5数据文件2219645字节，逐字节复制核对及SHA；仅本地，待本批私有push。

[相关32测试](../runs/20260905T094320Z-current-diagnostic-tests-52e62e2a/log.txt)实际1.401s测试/2.991s job全通过：route3、Adam8、sensor16、mode5，未扩展到无关/legacy测试。controller静态review传感器6例driver未见阻断，零/关闭需要真实forward/编码/物理/科学摘要同plain，非零保留实际响应。独立reviewer四模式读源码无阻断：严格400来源配置、原0初始化顺序、真实48_evaluate_case调用、两个sample科学参考精确、NR只引用同记录；仍待原生证据。400初始化sample精确并不等于argmax是作者decoder或能替换主指标。

### 09:44–10:26 UTC：诊断闭环与有依据的新训练路线

**50例路线。** 首次[route-obs-fifty](../runs/20260905T094432Z-route-obs-fifty-3d08c02b/artifacts/result.json)13.894s全部在create之前失败，零有效轨迹。整数JSON origin使原生经度数组的原位浮点归一化报dtype错误；controller只修独立driver的构造坐标为float，补真实整数夹具。[4回归](../runs/20260905T094712Z-route-obs-coordinate-tests-edb3434d/log.txt)通过，随后[50例成功](../runs/20260905T094801Z-route-obs-fifty-float-25d5d741/artifacts/result.json)27.799s，50到达、无超时/耗尽，列明测量检查全通过，24个跨arm pre_turn_controls_exact全部True。含2Mavic名义镜像、24Amzn固定速度/lane/转向、24对应外部刷新；固定未训练模型只在触发时测输入/输出，不控制响应。原始48Amzn均有整步越界，Mavic均0、max12.265m；有限表不等于全部时变策略不可解。

半速Amzn北转中心最大偏离native916.518m/109.25outside-s，refresh102.676m/2s；外侧873.834m/101.75s→76.487578m/.25s。route_fidelity只读源码确认1.1.1 direct/advance缓存turndist，而traffic按当前TAS转弯；旧fork固定SHA849d76fd44880f8d17a69aefa0bd37208f2b2fbb reached每次刷新。当前半速实际40.3325m/s、R355.727m，旧缓存1422.908m；native首真实航点advance44.25s、refresh70.75s。1.05倍中心刷新反而380.924→452.119m，不能泛化为普遍改善，不能据旧fork改变Table3。完整证据/公式/镜像表及4轨迹另存[route-response报告](../reports/route-response-20260905/README.md)。实际4面板[渲染](../runs/20260905T100410Z-route-response-plot-f6b00a4a/status.json)3.979s，PNG目视通过。

**末步统计解释。** observation_reward_spec只读完整physics.csv：refresh北转s0l2与镜像南转s0l0各776行，前775均inside，只有最后一步末点outside。北转有效出口插值193.587162801s（193.50→193.75、fraction.3486512023），cross76.200000000791m在现1e−7容差内；末点沿出口前进6.567628m，侧偏仅多5.111mm，而有限折线距离76.487578包含纵向overrun。环境/action各计整个末步后才detect/delete，.25s不是出口前实测越界时长。保留原始值，不删整步，不推出连续安全或推广其他46例；若未来改截止需一致处理risk/time/reward/terminal state。该小末步现象无法解释400轮88173 aircraft-s越界。

**真实Adam首步。** [adam-step-probe](../runs/20260905T094635Z-ppo-adam-step-probe-bf41552b/artifacts/gradient_probe.json)19.267s，旧静态诊断全部字段、episode101科学摘要/整个ppo update均精确复现；额外observer-copy整77步最终model/Adam/shuffle/metrics也与原update相同。首步shared位移norm .000689369，actor g·Δ shared−1.58874e−6、weightedcritic +.000384402；固定64 actor目标.02151170→.02150103、weightedvalue .77312833→.77356440。历史Adam动量可使当前batch损失上升，不能判优化器bug或critic因果妨碍。64样本单步KL7.4595e−9与4889样本77步KL3.60475e−5测量窗口不同。新结果和精确对照追加原[梯度报告](../reports/ppo-gradient-100-20260905/README.md)，无权重晋升。

**传感器六例。** [encoded-sensor-native](../runs/20260905T094908Z-encoded-sensor-native-probe-784a2f6f/artifacts/result.json)10.420s全部通过，plain/empty/disabled编码、mask、动作、真实forward、sampling RNG、物理及科学摘要精确相同；非零显式missing/abnormal/falsified实际进入原策略，finite合法输出、source/truth/mask保持、模型0grad且hook恢复。各2/2到达，27forward；plain52逐机推理LoWC26.5/NMAC4，missing54推理32.5/0，abnormal52与falsified52的风险仍26.5/4。无事件率/训练/鲁棒性结论，详见[六例报告](../reports/encoded-sensors-20260905/README.md)及PERTURBATIONS。

**400与0两种部署。** [checkpoint-policy-modes](../runs/20260905T095203Z-checkpoint-policy-modes-400-d0460c2d/artifacts/result.json)366.420s，48次真实原_evaluate_case；400严格源码/配置/版本、原0初始化61001，两个sample所有percase/aggregate科学字段（含动作hist）精确复现，排除仅wall/RSS；NR两参考精确一致后引用，0额外NR运行/optimizer/update/checkpoint/selection。0sample239/360、400sample236/360；0argmax235/360(43超时82耗尽347width)、400argmax298/360(1超时61耗尽338width)，全0height。NMAC/LoWC每flight-hour分别63.799/366.513、66.596/361.544、152.175/715.463、149.196/607.074。单轨迹argmax任务改善值得保留，但sample主指标不变，仍无可信baseline。原论文p6训练sample明确，p17测试decoder未定。小完整资料另存[四模式报告](../reports/policy-modes-400-20260905/README.md)。

**新路线交接。** 暂停旧400→600；新证据支持将普通flyby当前状态刷新作为显式重建修正，验证后从原初始化新训。shared_ppo问题为当前TAS/qdr公式、step范围Proxy恢复与严格身份；仅写navigation_refresh.py、paper_environment.py、paper_train.py、两refresh configs、test_navigation_refresh.py，≤25min静态，保留他人修改且禁实验/GPU/信号/清理/通知/嵌套。交付默认off保持、on每步finally恢复/逐次人口数组、episode-local audit和8fixture，训练source hash新增adapter，旧checkpoint不迁移。[19相关测试](../runs/20260905T102203Z-navigation-refresh-core-tests-feb50c3b/log.txt)实际1.460s测试/2.992s job通过（8新+env+train）。

10:21追加shared_ppo仅新navigation_refresh_probe.py/config/必要纯测试，问题为新集成是否精确等于原default与已审计external refresh；来源原0训练参考、50例结果及当前adapter；交付原12 NR/sample、4fixed响应、30机NR/sample集成/外部对照，真实native计数/出生删除/reset/恢复与原物理rows，≤20min静态，controller≤420s串行执行。原始4refresh物理行已定向复制reports/route-response-20260905/refresh_four_physics.csv及manifest，2382行/1754325B，来源SHA保留。route_trace独立只读6变更文件/测试/BlueSky调用链，≤15min，返回无阻断原生验证问题；native零人口跳过不要求wrapper数等于全部物理步数。无实验/修改。10:26实际turn_context核对shared_ppo第10turn、route_trace第13均gpt-6-astra/xhigh。

[0–400图](../reports/execution-learning-20260905/README.md)19评价点及23来源记录，重复15/100/150/250 aggregate精确合并一次；实际[最新渲染](../runs/20260905T100952Z-execution-learning-figure-400-a22a1834/status.json)3.291s，三产物逐字节复制、PNG目视通过。NOW/DECISIONS/REPRODUCTION/PERTURBATIONS/LESSONS同步本批证据。主10h截止仍15:15:13 UTC，CPU14/单线程/低优先级、不向共享GPU提交任务或干预其他人，未提前发送完成通知。

10:29本批已提交并push `cc482f0350933382f493c2b9a515b17846b664ef`，ls-remote精确同SHA，含400恢复点与已验证诊断；新navigation core/config/test暂未入此提交，保留旧源码可恢复版本。doctor无问题；生成SVG原样尾空白与CSV标准CRLF明确处理，其余diff检查通过。10:30获准主机只读nvidia-smi仍见3059945/3062529显存9891/9771MiB，GPU98%、剩3432MiB；本项目无GPU计算/外部进程干预。load约2.24/2.23/2.32、MemAvailable61.26GiB、磁盘143.48GiB。

### 10:33–10:39 UTC：刷新原生验证通过并从头训练

[比较器3回归](../runs/20260905T103310Z-navigation-refresh-probe-tests-aee50278/log.txt)通过；controller窄读driver后运行[原生集成](../runs/20260905T103332Z-navigation-refresh-native-09ae0b8a/artifacts/result.json)，内部204.281s/监督205.973s正常完成，8项总检查全部通过。默认关闭原12例NR/sample的percase及aggregate全部精确复现（只排wall/RSS），无wrapper；开启四固定响应完整summary/phase/milestone及2382物理CSV行全部精确等于既有external刷新，仅arm标签显式不同。

30架次seed53001 NR/sample分别原生5451/5965 calls，与外部wrapper计数/调用流hash/完整物理流hash和每机摘要精确相同；人口大小分别1–19/1–20、各56次ID变化，reset初始计数0，方法和callback恢复。四单机native/wrapper761/775/430/412 calls，公开step contexts39/39/22/21均完整退出。模型无变化/梯度，无checkpoint加载/优化器/训练。NR30/30且0width；未训练sample22/30、3超时5耗尽、21width/4410.25 aircraft-s，仍不是有效基线。完整小结果另存[刷新验证报告](../reports/navigation-refresh-20260905/README.md)。

接着启动refresh-ppo-100：新配置current_state_refresh、从seed61001初始化和原training610000流起，100轮目标/内1800s/外1860s、256MiB输出/4096MiB每进程地址空间，CPU14单线程nice15/idle IO、每25全12开发评价/每5原子保存。严格新源码/配置，不恢复旧400模型，其余Table3/执行动作/观测/奖励/终止口径不变。10h截止和14:45收尾不变。

追加shared_ppo仅拥有checkpoint_policy_modes.py/原测试/新refresh100配置：问题为复用四模式诊断于新训练轮次，交付保留v1旧400、显式v2轮次派生固定四模式、generic reference参数、严格源码/配置/同轮参考、原0sample精确对照；≤20min静态，禁止运行实验/测试/GPU/信号/清理/通知/嵌套。追加observation_reward_spec只读原p6/Algorithm1/训练预算与当前循环，问题为250k episodes单位和更新节奏的可比性，交付有页码事实/歧义与有限预算解释，≤15min，不重审reward、不改参、不执行负载。两任务与controller串行原生训练独立。

10:40刷新源码/原生报告已随`535cad4cd6ebffe3f13034459ca59fc2669997e0` push且远端同SHA。实际新训练run为[runs/20260905T103813Z-refresh-ppo-100-4e103166](../runs/20260905T103813Z-refresh-ppo-100-4e103166/status.json)，10:38:13启动；初始全12评价187.787s，NR360/360、16width/217.5aircraft-s、max412.216m；sample246/360、53超时61耗尽、339width/51139.5aircraft-s、max465.519m，66.945833flight-hours，LoWC29185.25/NMAC5283.75 pair-s（435.953196/78.925748每flight-hour），return−2641.982548，0height。它是修正环境中的未训练起点，不能把与旧0轮的差异当学习收益。

10:43 observation_reward_spec只读预算回执：原p6/Algorithm1、p17和p18图轴确认250k为计划30机场景外层收集/更新循环，K1是一次样本遍历、B64为小批量大小。120s real-time截断与1200s每机及临时空人口仍有歧义，不能认定作者750万条完整flight。原文未明确名义预热/课程/advantage或return归一化/尾batch/clip；现选择继续明确标记，不因此调参。其只读核查251–400段150轮=606847transitions/9558Adam steps/842.5383flight-hours。controller随后定向读取6份现有备份日志并逐轮核对1–400连续无重复、seed610000起、每轮30planned/K1/sample_visits=samples=rollout_samples、minibatches=ceil(samples/64)：完整400为12000planned、1616717transitions、25463Adam steps、131621global decisions、2244.607361training flight-hours，日志collection+update3306.347s（不含评价等）。[预算JSON和源hash](../reports/training-budget-20260905/README.md)保留。无新实验/模型加载；不能用此小预算称论文规模训练失败，旧/新谱系轮数不相加。

四模式v2三owned文件已静态交付，controller窄读diff未见阻断；v1保持400，v2按正整数轮次派生固定四模式并严格同轮引用，generic/旧reference别名互斥。新增3fixture加旧5项，等待当前唯一训练job结束后再执行；当前未宣称测试/native已通过。

10:54新refresh25/50轮全12开发评价各105.41/105.39s；25完成240、53超时67耗尽、334width、outside46975s、NMAC88.123168/LoWC443.938988每flight-hour、return−2720.126896；50完成252、53超时55耗尽、336width、outside46619.5s、66.767014flight-hours、NMAC76.568498/LoWC423.420164、return−2435.113311。均0height；50较新0的246完成/78.925748NMAC有所改善，不把单点评价当稳定收敛。NOW压缩为当前运行/关键证据/约束与入口，完整历史仍在此任务与reports，doctor无断链。

10:48追加route_trace只读当前PLAN/SOURCES/REPRODUCTION及最多3篇一手近邻，问题为后续action-delay队列/预测过滤/多机CBF的具体假设和公平对照；≤20min定向浏览，不改变主线、无修改/实验/外部消息/嵌套。10:54回执已将ICLR2021 Random Delays/DCAC、TCST2023动态环境输入延迟CBF、T-RO2025 GCBF+的正文/官方代码入口、延迟/可见信息/保证假设与适用边界写入SOURCES。下一步候选明确零延迟是否有隐含一步、FIFO/覆盖/过期丢弃、初始保持和生效合法性，再比较同信息RNN/队列、当前/预测/误差余量过滤。此为后续来源核验，不宣告创新或转移连续控制定理到60离散动作/5s锁；当前继续无延迟基线。

10:59新refresh75轮评价107.325s：251/360、53超时56耗尽、337width/51753aircraft-s，NMAC84.147763/LoWC459.933228每flight-hour、return−2801.831348，0height；冲突回升，不把50单点判收敛。10:59只读实际turn_context复核shared_ppo第11turn、route_trace第14、observation_reward_spec第8均gpt-6-astra/xhigh。当前100段结束并备份/四模式核验后，预计按实测9.1s收集更新、105s每12例评价，可分段100→300（约45–50min）继续原配置；仍以结果/剩余墙钟决定，不跨14:45收尾预留和15:15硬截止。

### 11:04–11:07 UTC：新100轮完成与同谱系解码核验

[refresh-ppo-100](../runs/20260905T103813Z-refresh-ppo-100-4e103166/artifacts/result.json)正常完成100轮及五次完整开发评价，内部1541.836s/监督1543.665s，11:03:57结束，stop=episode_target_reached，best50。100sample240/360、53超时67耗尽、341width/47057.5aircraft-s、max465.519m、0height，66.782361flight-hours；LoWC27779/NMAC5214 pair-s即415.963131/78.074508每flight-hour，return−2380.929450，48101policy decisions。尚无持续有效基线；旧新执行谱系不合并。

[新100小副本](../checkpoints/refresh-100-20260905/README.md)5数据1988998字节，逐字节复制及SHA核对，含latest100/model/Adam/RNG/best50，源535cad4兼容，下一training seed610100。准备本批push后同配置100→300、内3300/外3360s（约45–50min），CPU边界/截止不变。

[通用四模式8测试](../runs/20260905T110514Z-checkpoint-policy-modes-v2-tests-3a87d157/log.txt)1.429s job全通过；controller窄读v2保留严格身份与同轮reference、v1兼容后，启动refresh-policy-modes-100，外600s/128MiB/4096MiB，两个reference均从新100副本development.jsonl分别明确选0/100，无模型更新或选优。实际结果待完成。renderer只增加可选单行title及源码hash，默认标题保持，纠正原docstring误称PDF；新0–100五点评价输入已准备，真实渲染待单launcher间隙，不新增镜像实现测试。

### 11:13–11:18 UTC：四模式完整结果与图表核对

[refresh-policy-modes-100](../runs/20260905T110609Z-refresh-policy-modes-100-13b09003/artifacts/result.json)内部409.995s/监督411.688s，48实际native全部通过，两sample逐case/aggregate（含动作hist）精确参考，严格源码/配置/软件和checkpoint内嵌参考通过、NR两个记录一致后引用，模型/输入不变、0optimizer/update/checkpoint writes/selection。新100argmax306/360、1超时53耗尽、334width/69514.25aircraft-s、55.655556flight-hours、NMAC162.229986/LoWC657.167099每flight-hour；新0argmax240/360、30超时90耗尽、342width/74250.5s、60.490694h、175.886557/765.716949。均0height；任务改善66架次不等于有效baseline，100argmax NMAC仍稍高于NR161.163924。完整小资料另存[同谱系四模式报告](../reports/policy-modes-refresh-100-20260905/README.md)。

新图首[render](../runs/20260905T111403Z-refresh-learning-figure-100-af4a767e/status.json)数值正确但总标题被subplot循环变量覆盖，controller目视发现；仅renderer将figure_title与子标题分开。新增1个owned随机临时夹具，实际导出默认/自定义两组六子图；首[fixture](../runs/20260905T111600Z-learning-title-regression-ea9c6c10/log.txt)误假定SVG文字存为path注释而失败，metadata标题已正确；改为解析真实SVG text节点后[回归](../runs/20260905T111652Z-learning-title-regression-svg-text-a1fa873b/log.txt)通过，4.081s job。该测试不改任何模型/指标。固定图[render](../runs/20260905T111615Z-refresh-learning-figure-100-title-fixed-0e6283ec/status.json)3.343s，SVG/PNG/PDF/metadata逐字节复制、PNG目视通过，见[新学习曲线](../reports/refresh-learning-20260905/README.md)。

另纠正此前README的PDF缺失推断：rg文件枚举遵从ignore而未列PDF，但输出常量和实际文件表明renderer一直生成PDF。400图PDF38072B已从明确源run复制，旧README已改回“PDF本地保留、不入Git”；不改原run。不是数据丢失或新增PDF功能。当前无launcher负载，准备将本批已验证新100/图/报告/工具与来源记录私有备份，再同源码535cad4恢复100→300。

11:19提交`1325351f961b53d6ef33edd7e1f7156caaf9ee38`已push且ls-remote同SHA，包含新100恢复点及已验证四模式/曲线/预算与三篇近邻来源；源码/文档检查通过，生成SVG尾空白按原样保留。随后[20260905T111952Z-refresh-ppo-300-bebea134](../runs/20260905T111952Z-refresh-ppo-300-bebea134/status.json)启动，原科学源码535cad4/同配置恢复100→300、内3300/外3360s、256MiB输出/4096MiB地址空间，CPU14/nice15/idle IO单线程，当前唯一job。持续轮询、原25轮评价/5轮保存和截止预留不变。

11:24 refresh100→300的恢复初评188.046s完成，整个NR/sample aggregate与新100副本末评精确一致；已续到107、最后seed610106，单轮9.268s，当前运行正常。这里只核对实际已完成的恢复评价，不预报300完成。

11:29新refresh125轮全12评价105.468s：252/360、53超时55耗尽、341width/46934.75aircraft-s、NMAC78.410800/LoWC418.881112每flight-hour、return−2585.656747，仍0height。完成与50同为252但NMAC高于50的76.568498，不作有效基线或稳定趋势判断。

11:32追加shared_ppo仅新lateral_exposure_audit.py/refresh100配置/聚焦测试，问题为新模型剩余越界主要机型与横移执行状态；只读当前535科学源码、100副本和原12sample参考，通过原_evaluate_case旁路真实physics callback聚合逐机/机型outside，按post-update lane_active和capture-active分区，保存首/峰/末样本及终态。所有科学摘要/动作hist须精确参考，不追加forward/RNG、不改物理/奖励/终止；以completed*dt给出有效出口后整末步贡献的保守上界，不擅自删步或作连续安全判定。输出紧凑聚合、无巨大全量CSV；内<=450/外480s、128MiB，当前300结束后controller单job执行。builder<=25min静态实现，不执行负载/清理/信号/GPU/通知/嵌套，保留他人修改。

11:33获准主机只读GPU审计：原PID3059945/3062529仍9891/9771MiB，GPU98%、used20616/free3432MiB；未向GPU计算或干预进程。主机load3.396/3.258/3.096、MemAvailable60.662GiB、磁盘143.343GiB。新refresh已150轮、正在全开发评价，当前唯一launcher仍100→300。

11:39追加observation_reward_spec只读理想90°几何推导，问题为Table3异质转弯半径与中心/内外极限lane可行性：名义入射ray x<=0,y0、出射x0,y>=0，半宽76.2，恒R切向1/4圆连接平行偏移；独立核对controller候选a=1−1/sqrt2及max-distance/端点/投影分支，不把候选当证据。来源50fixed刷新与旧Mavic响应、当前几何/机型；输出假设/证明或反例、原生偏差和有限航段边界。<=15min，无修改/实验/加载checkpoint/信号/GPU/通知/嵌套，不据理想模型宣称BlueSky连续安全、策略不可行或创新；当前baseline不改。

新refresh150/175全12评价105.936/105.669s。150：237/360、53超时70耗尽、343width/46835.75s、NMAC83.138492/LoWC432.774433每flight-hour、return−2617.307704。175：247/360、53超时60耗尽、340width/52226.75s、NMAC77.392891/LoWC416.148830、return−2462.515055。均0height，无稳定改善。11:41训练已180轮，唯一job继续100→300。

11:46新refresh200轮全12评价105.236s：259/360、53超时48耗尽、339width/49165.75aircraft-s、NMAC70.161553/LoWC418.773718每flight-hour、return−2444.970593，0height。为当前新谱系最高完成数、NMAC较初始下降；仍有显著失败/越界，不据单点晋升有效baseline，继续300目标。


### 11:55 UTC：越界归因与理想转弯独立核对

shared_ppo交付3个owned侧向暴露审计文件，AST/JSON/空白静态检查通过，未执行测试或native。controller已核对主流程，原_evaluate_case/sample/12case不变；随后交observation_reward_spec只读独立review，<=20min，范围新probe/config/tests及必要helper，重点精确重放、状态归因、逐机/机型对账和完成数乘dt上界。当前训练结束后才串行fixture/native，尚不宣称实测通过。

observation_reward_spec完成理想90度恒R推导，中心aR、内边w+aR、外边三分支及有限ray投影均独立确认，a=1-1/sqrt(2)。w76.2时中心R上限260.163073m、外边区间76.2–520.326147m；小Mavic R45.373348的外边圆角仍可越界。只适用已经捕获平行航线后的固定精确四分之一圆，不推广任意控制/连续安全/创新。完整假设、分支证明、三组原生差异和终点/协议边界写入[几何报告](../reports/turn-geometry-20260905/README.md)。

route_trace获新绘图handoff：仅turn_geometry_plot.py与可选聚焦fixture，<=20min静态；root拥有报告/input，输入已从原始保存case取得精确R/最大距离及源SHA。预期归一化公式曲线、两组理想路径与明确原生比较图，SVG/PNG/PDF/metadata；不执行render/实验，不拟合R，保留半速Amzn出口后76.487578及旧Mavic捕获协议限制。

新refresh225全12sample：244/360、53超时63耗尽、334width/46663.25s、NMAC69.552686/LoWC417.702788每flight-hour、return−2321.746016。完成数较200回落，当前仍无稳定有效基线。11:55训练已250，正在评价，300目标继续。

11:59仅读取上述3个研究线程自身turn_context：shared_ppo第12turn、route_trace第15turn、observation_reward_spec第10turn实际均gpt-6-astra/xhigh；未读认证信息。refresh250全12评价105.301s：254/360、53超时53耗尽、337width/45557.5s、NMAC77.771735/LoWC419.675732每flight-hour、return−2429.715931。11:58已260，仍同300job。

12:02独立lateral review未见正常采样/物理/归因静态阻断；指出实际forward rows未动态检查60000上限，以及JSON超限后重复保存同一大对象可能缺最终错误。交原builder仅probe/test窄修正，<=8min静态，动态计数异常仍恢复hook、保留最后有效partial并额外小failure旁注；controller稍后执行。全局RNG检查不单独证明私有sampling generator状态，须由原链路静态不访问及完整参考重放补证；分区为末态整步分类而非精确驻留时长。

12:03 refresh275全12sample：256/360、53超时51耗尽、337width/51819s、NMAC76.534319/LoWC401.160580每flight-hour、return−2331.521226。训练已282轮，继续300。下一续训将依据实际间隙后墙钟选择较长同配置段，减少重复恢复初评；仍不越过14:45收尾/15:15:13硬截止，不承诺未完成轮数。


### 12:08–12:10 UTC：新300正常完成与侧向暴露重放

[refresh100→300](../runs/20260905T111952Z-refresh-ppo-300-bebea134/status.json)12:08:14.961正常完成200新增轮、最终评价，CLI2900.276671s/job2902.142s、best200。300sample252/360、53超时55耗尽、335width/47705aircraft-s、0height、FH66.37930556；LoWC27059.25/NMAC4941.25无向pair-s，407.645874/74.439616每flight-hour，return−2278.502708。仍未有效，不冻结。

[300副本](../checkpoints/refresh-300-20260905/README.md)五数据文件2406481字节，与已完成源run逐字节相同，manifest SHA保存；latest含model/Adam/RNG/best200，原科学源码535cad4保持，下一训练seed610300。新学习图输入延伸13点0–300；恢复100重复NR/sample aggregate逐项一致才合并，每行源SHA/行号保留，等待单job渲染。

侧向审计两项review修正已交付，controller复核实际forward rows限额及输出超限小旁注/原partial不变。[6项fixture](../runs/20260905T120840Z-lateral-audit-tests-84a72342/log.txt)0.014s测试/1.431s job通过。随后[原生12例审计](../runs/20260905T120908Z-lateral-exposure-refresh-100-35f42563/status.json)启动，内450/外480s、128MiB/4096MiB、CPU14单线程，当前唯一job。12:10前4case已完成且逐项参考与逐机对账通过，不能提前称全部通过。


12:11[侧向审计](../reports/lateral-exposure-refresh-100-20260905/README.md)全12sample/360flight正常通过，内部131.646202s/job133.410s；961666飞机物理样本、3784forward/48101policyrows，全部13项顶层检查/逐case参考/逐机环境-controller三方对账精确，模型/输入不变、0grad、hooks恢复。结果3325366B及manifest已复制核对。47057.5s总outside中lane-active37757(80.24%)，capture-active710.5(1.51%)；四joint分支9300.5/0/37046.5/710.5s。M600+Mavic14561.25(30.94%)，Amzn817.25(1.74%)；53timeouts仍29Mnet+24Tecnalia。成功出口后整末步上界60s仅0.128%，保留raw、不减去/不作连续安全保证。原生当前状态分类不能直接证明永久锁或因果。

12:14交observation_reward_spec只读后续判别设计，<=20min、审计完整样本+paper_actions必要helper；最多一个小固定响应/旁路诊断，必须区分已存首峰末样本与未观察区间，核对实际完成条件，不改主训练或执行实验。

[几何4fixture](../runs/20260905T121137Z-turn-geometry-tests-7d3df300/log.txt)0.022s通过；[render](../runs/20260905T121145Z-turn-geometry-plot-940007d3/status.json)3.501s正常，四artifact逐字节/输出SHA核对、PNG目视通过，理想/实际差异和两类排除含义明确。新[0–300 render](../runs/20260905T121247Z-refresh-learning-figure-300-0254bc91/status.json)3.296s，四artifact复制核对/PNG目视通过；13actual评价、重复100 aggregate精确，报告更新。PDF本地、SVG/PNG/meta入Git。当前无launcher，准备本批备份；下一较长同配置目标300→850，按12:20前可用窗口内8100/外8160s，不跨14:45收尾或硬截止。

12:17提交f1cfb8574ea5c48956a5da4977ffb9d209a1de6b已私有push且ls-remote同SHA，保存300恢复点/完整侧向诊断/几何推导图与0–300图；doctor无问题、staged diff检查通过（原样生成SVG排除尾空白）。12:18:15启动[refresh300→850](../runs/20260905T121815Z-refresh-ppo-850-5786ced9/status.json)，内8100/外8160s、256MiB/4096MiB，原科学源码535cad4与原配置严格恢复。当前唯一launcher/CPU14单线程低优先级；目标850与实际完成分开，最迟监督截止14:34:15，保留最终验证和14:45收尾。原10h硬截止15:15:13不变，主线程不提前通知。

12:21 refresh300→850恢复初评188.696362s完成，整个NR/sample aggregate与300副本末评精确相同；开始新轮次，当前唯一job保持原资源/源码/配置与截止。


### 12:24–12:26 UTC：锁条件只读核查与一个后续判别诊断

observation_reward_spec只读原审计+paper_actions/config，未修改/测试/项目负载。seed53004 M600/F020存活797s、outside670.75s、lane-active631.25s、active普通航点630.25s，实际5commands/5captures、终态锁关闭，capture_beyond_leg_end_commands0，故不能称一直未完成。全360终态均满足commands-captures=int(lane_active)，仅计数一致证据。F020首706.5s距离76.644649m/target+76.2/计数1-0/普通N1/LNAVon；峰837.25s82.187604m/target−76.2/3-2；末1484s76.295421m/5-5/锁off/LNAVoff，终态导航耗尽，保留航点索引不等于LNAV还在运行。

释放需普通名义航点映射、有限平移航段0<=along<=length且length>0、相对目标侧向误差<=2m、实际track相对名义方向误差<=5度同时成立。CAP切换本身不释放。静态可构造边缘target76.2/actual77.2仍满足2m容差而严格越界；这是现有明确重建选择，未发现违反源码规则的bug。完成几何用固定原点当前段有符号横向坐标，暴露用当前位置投影完整有限折线最近距离，两者不能用centerline_distance−76.2简单替代。

仅交shared_ppo新lane_completion_probe.py/对应refresh100配置/聚焦fixture，<=25min静态、禁止负载与核心改动：原100模型/原case53004采样种子重放完整30traffic，仅F020保存完整3188期望物理记录（由实际存活/dt推导），直接读取原纯几何helper的lane_error/track_error/along/length/映射判据、代次与lock/CAP/LNAV/真实目标速度。原case科学摘要+actionhist及prior audit逐机终态精确核对，模型/输入/hooks不变；内<=60/外90s/32MiB，当前850结束后controller串行执行。预先区分侧误差尾段、有限段条件、所有判据真仍锁定的矛盾、已解锁但容差/几何仍越界；不直接推因果。

12:29获准只读GPU快照：原PID3059945/3062529仍9891/9771MiB，4090 used20616/free3432MiB，利用率100%；不计算/不信号/不调整进程。load3.446/3.275/3.158，MemAvailable60.701GiB，磁盘free143.257GiB。快照不证明他人吞吐绝对零影响；我们继续CPU14单线程低优先级。新refresh325评价103.810s：237/360、53超时70耗尽、340width/48334.5s、NMAC73.757107/LoWC416.124495每flight-hour、return−2498.740555，仍0height。

12:31追加observation_reward_spec只读下一冻结策略延迟协议审查，<=20min、<=1200词，读取现环境/actions/train接口及已存近邻源：区分generation/acceptance/application、零延迟精确基线、排队动作生效时已masked、FIFO与latest-overwrite分开、队列信息公平性及未来PPO请求动作logprob与实际执行动作区别。输出一个最小推荐诊断和未定项，不实现或启动延迟，不提保证。当前主线已350，继续850目标。

12:33 refresh350全12sample：242/360、53超时65耗尽、337width/50575.25s、NMAC77.594533/LoWC420.858109每flight-hour、return−2600.232214。后续原v2四模式配置仅新建checkpoint_policy_modes_refresh_850.json并显式设轮次850，未执行；须实际同轮checkpoint/reference产生后才用，若目标未达则使用明确实际轮次的新配置，不放宽验证器或重标文件。


### 12:38 UTC：下一冻结策略延迟协议的只读设计核对

observation_reward_spec静态读取环境/actions/train及已索引Ramstedt/Molnar原文，无修改/测试/模型加载/实验。建议把未来输出延迟置于“请求生成入队→执行器接收”之间，入队不改变accepted目标/锁/航线；保存绝对60类索引，交付被接受时才用当时位置/名义航段构造CAP。提前生成整条航线再延迟是不同协议。

记录t_generate（按当前观测/mask生成）、t_ready（最早可交付）、t_attempt（实际交付校验）、t_accept（接受，拒绝为空）、t_apply（有变化分量下发API，重复目标可能无新API）及原捕获完成时刻。完成机动不等于通信延迟结束。生成时合法不保证交付合法，先前在途动作可能开启锁，最后航段防回折mask也会随位置变化；推荐交付时沿原执行器整联合动作拒绝，保持先前目标、原生导航继续，不重采样/部分执行/自动等解锁重试。这是建议协议，尚未实现或冻结。

不能直接以出队子集调用现env.step：它要求当前全飞机批次，非法动作是异常，而延迟子集与拒绝应是正常事件；补伪“保持动作”会污染decision/acceptance计数。未来应分开请求生成与交付尝试，changed_instructions只计真正目标变化。FIFO逐机生成序交付，随机delay会有队首阻塞，同步多到期逐条重校验；latest-overwrite须进一步指定在收到新请求还是在生成新请求时丢旧。后者在delay>5s可能一直取消在途请求，不能隐去。Ramstedt原文§5按收到信息的生成时间辨新旧，不包含本项目分量锁拒绝规则：[原文](https://arxiv.org/html/2010.02966v3#S5)。

零delay应走原即时路径、保持观测/mask/排序采样/API/物理/终止顺序，不增加.25s或5s等待。验收需输入、请求、accepted目标、逐步物理、终止和科学摘要；额外观测调用也可能改变clipping计数。延迟RNG独立，不消耗策略sampling generator。冻结策略仍接收原7/10及真实执行器状态mask，在途意图不冒充accepted；这明确只延迟动作输出、不新增状态/锁反馈延迟。

未来memory/显式历史/预测比较应统一自己的请求/时间戳、接收拒绝ack是否可见及何时可见、delay是否事前已知；不能把模拟器未来随机到达时刻只给某方法。历史增强不保证消除所有部分可观测性。[Ramstedt§2.1](https://arxiv.org/html/2010.02966v3#S2.SS1)。未来延迟PPO仍记录生成请求及其原mask下logprob，不用后来执行/保持动作替代轨迹标签；当前下一冻结评价没有PPO更新。[Molnar定理3–4](https://tamasmolnar.com/publication/2023_Molnar-et-al_safety%20with%20input%20delay%20in%20dynamic%20environment_TCST.pdf)依赖输入历史/模型/初始等待段安全，不能给当前离散拒绝和BlueSky直接提供保证。

最小候选验收：有效名义模型冻结后，一个已有case配对0s/固定6s、逐机FIFO、到期整联合校验拒绝。6s只用于超过5s周期且非整数周期的机制测试，不是实测网络分布。建议事件序为前一步update/终止完→决策边界照常生成入队→下一物理步前交付到期；出生原名义初始化，终止清除该机队列，reset清空。检查24物理步延迟、未交付目标保持、拒绝与终止清理、零delay精确。固定等delay无乱序，不能判FIFO/覆盖优劣或形成退化曲线。delay分布/相关性/乱序/丢包/容量/覆盖时机/ack/边界顺序仍需明确；本10h不提前进入该实验阶段。

12:39 refresh375全12评价103.766778s：237/360、53超时70耗尽、344width/48392s、NMAC83.923429/LoWC425.328793每flight-hour、return−2738.278595，0height，出现回升，不据早期点外推改善。12:39训练已382，850目标继续。

12:42 shared_ppo交付lane_completion_probe/config/4fixture，仅AST/JSON/空白静态检查，无执行；交route_trace只读独立review，<=20min，范围新3文件和必要helper，核对实际判据、被动身份、生命周期/代次/类别与限额，不改核心。root读取集成接口，提示tail当前由capture计数增加（解锁）定义，须与CAP→普通航点的锁内跟踪阶段分开；完整CSV仍保留事件，不按术语推因果。

12:44 refresh400全12评价103.902165s：249/360、53超时58耗尽、338width/50640.5s、NMAC87.854245/LoWC432.393305每flight-hour、return−2557.674974，0height。新谱系400不是旧cached400，不混表或合并训练；当前已402，继续850目标。

12:48 laneprobe摘要窄修正完成，独立route_trace后续只读复核通过：保留release-tail并显式tail_start_event=lane_lock_release；新增每代lane_active且nominalmapping的实际tick/秒/越界/category/predicate汇总，不要求此前观察CAP，不由首末时间推连续。第五fixture覆盖该区别。两Python AST/JSON静态通过，无测试/模型/负载；待主线程850结束后执行5fixture+单case。无其它静态阻断，原sideflag仅along/track通过的限制保留。

12:50 refresh425全12评价103.807239s：248/360、53超时59耗尽、333width/42966.25s、NMAC71.844492/LoWC402.707686每flight-hour、return−2315.209318，0height。当前较低暴露点不等于稳定改善；已429继续850。

12:55 refresh450全12评价103.870827s：241/360、53超时66耗尽、338width/48772.5s、NMAC78.870587/LoWC435.674586每flight-hour、return−2614.469758，0height。与0相近，best仍200，当前继续850。

13:00仅读指定3个研究线程自身turn_context模型字段：shared_ppo第16turn、route_trace第18turn、observation_reward_spec第13turn实际均gpt-6-astra/xhigh。未复制认证配置。当前475评价中，唯一850job持续。

13:01 refresh475全12评价103.748723s：238/360、53超时69耗尽、336width/45102.5s、NMAC78.049901/LoWC433.527550每flight-hour、return−2562.454145，0height。当前已484，继续850。

13:06 refresh500全12评价103.266523s：244/360、53超时63耗尽、339width/41466s、NMAC75.037531/LoWC419.568960每flight-hour、return−2551.603753，0height。越界秒数下降但架次仍339，未有效；已505继续850。

13:11 refresh525全12评价104.633205s：249/360、53超时58耗尽、332width/42327.75s、NMAC74.767197/LoWC395.523218每flight-hour、return−2364.008343，0height。安全暴露变化仍不能弥补完成/containment，不冻结；850继续。

13:17 refresh550全12评价103.433280s：251/360、53超时56耗尽、334width/44643.75s、NMAC72.092093/LoWC409.751034每flight-hour、return−2353.937419，0height。仍无有效baseline，继续850；当前10h已8h、硬截止和收尾预留不变。

13:22 refresh575全12评价103.408472s：234/360、53超时73耗尽、341width/43378.75s、NMAC75.384188/LoWC390.486495每flight-hour、return−2195.317440，0height。LoWC较低而完成数回落，不能单项晋升，继续850。

13:28 refresh600全12评价103.436640s：243/360、53超时64耗尽、335width/45695s、NMAC74.344265/LoWC406.614333每flight-hour、return−2336.230563，0height。实际已601，继续850；仍无可信有效baseline。

13:29:58获准只读资源审计：原PID3059945/3062529仍9893/9771MiB，4090 used20618/free3430MiB、99%；未提交GPU或信号/调整其他进程。load3.329/3.521/3.431，MemAvailable60.652GiB、磁盘free143.250GiB。快照不证明外部吞吐绝对无变化；当前CPU14单线程nice15/idleIO唯一850job继续。

13:33 refresh625全12评价103.289707s：241/360、53超时66耗尽、339width/41323.5s、NMAC72.686927/LoWC395.637948每flight-hour、return−2320.639810，0height。当前已628，继续850；低outside秒数不等于低outside架次或完成改善。

13:39 refresh650全12评价102.895889s：245/360、53超时62耗尽、340width/42007s、NMAC71.377091/LoWC373.264432每flight-hour、return−2155.189827，0height。冲突暴露较低点仍无可靠完成/containment，已653继续850。

13:44 refresh675全12评价102.358593s：245/360、53超时62耗尽、334width/46993.25s、NMAC76.654116/LoWC393.731129每flight-hour、return−2407.864919，0height。持续波动，未达到冻结条件；850继续。

13:50 refresh700全12评价102.654906s：241/360、53超时66耗尽、336width/46038s、NMAC78.157014/LoWC405.494940每flight-hour、return−2623.194678，0height。仍近初始风险且任务失败，未有效；当前已703继续850。

13:55 refresh725全12评价102.995811s：247/360、53超时60耗尽、335width/45650.75s、NMAC68.553083/LoWC385.516615每flight-hour、return−2238.895624，0height。NMAC较低仍未达到任务/containment要求，继续850。

14:01 refresh750全12评价102.421270s：252/360、53超时55耗尽、336width/43197.5s、NMAC81.638022/LoWC395.775343每flight-hour、return−2453.180668，0height。NMAC回升，未有效；已751进入最后100目标轮。

14:07 refresh775全12评价101.932546s：260/360、53超时47耗尽、336width/43305.25s、NMAC72.489263/LoWC371.703895每flight-hour、return−2136.494436，0height。当前新谱系最高完成数，比200多1架，但仍未满足有效baseline或containment；最终best文件轮次待结束结果核对。已779继续850。

14:12 refresh800全12评价102.316941s：245/360、53超时62耗尽、336width/44538.5s、NMAC65.559434/LoWC353.234667每flight-hour、return−1980.971038，0height。新的较低风险点仍未同步达到任务/containment要求，不冻结或推显著性；已806，最后50目标轮继续。

14:17 refresh825全12评价102.752525s：255/360、53超时52耗尽、334width/42852.75s、NMAC73.710630/LoWC389.965168每flight-hour、return−2275.121892，0height。已827继续最后25目标轮，最终850及best文件尚待完成核对。


### 14:22–14:43 UTC：850完成、最终重放与锁条件诊断

[20260905T121815Z-refresh-ppo-850-5786ced9](../runs/20260905T121815Z-refresh-ppo-850-5786ced9/status.json)14:22:44.977正常完成300→850的550新增轮及最终全12评价，CLI7467.619323s/job7469.470164s、best775。最终850sample233/360、53timeout74route-exhaust、340width/47582.5aircraft-s、0height；FH64.25805556，LoWC23174/NMAC4144.25无向pair-s，360.6396085/64.4938594每flight-hour，return−2059.1171031，max偏离665.487062m。46284决策、54238changed、3463828.088m路径。best775为260/360、336width，仍非有效baseline；不冻结或开始延迟实验。

[850恢复点](../checkpoints/refresh-850-20260905/README.md)latest.pt/best.pt/result.json/development.jsonl/training.jsonl共3870813B，逐字节复制及manifest核对。最新模型/Adam/RNG/embeddedbest775完整，下一seed610850；旧400另系不迁移。新训练预算3日志1–850连续精确，3460683转移/样本访问、54484Adam步、279848全局决策、25500计划架次、4804.916250训练FH，收集更新日志7770.229432s（不含开发/setup/备份）。850仅原250k的0.34%；没有论文规模失败结论。

lane probe先前5fixture在[20260905T142256Z-lane-completion-tests-d45afac7](../runs/20260905T142256Z-lane-completion-tests-d45afac7/log.txt)通过，测试0.007s/job1.430226s。随后[20260905T142340Z-lane-completion-f020-a4377b08](../runs/20260905T142340Z-lane-completion-f020-a4377b08/status.json)18.836452s内部/20.509682s监督正常：完整原100sample/seed53004的30traffic，314forward/4067policyrows、F0203188行/1488768CSV字节；17检查全通过，原case科学摘要/actionhist、逐机终态/原审计样本与暴露精确，模型/输入不变0grad/updates/checkpoint写入/hooks恢复。全局RNG只读核对不单独证明私有generator，严格原sample参考重放提供进一步证据。

14:26交observation_reward_spec只读实际3188CSV/result/必要源码，<=15min：输出实测阶段/误差、机制解释与未定项及下一单一对照；禁止修改/实验/模型/GPU/信号/通知/嵌套。14:28交shared_ppo独占新lane_completion_plot.py，<=10min静态：完整样本四面板SVG/300dpiPNG/PDF/metadata、容差与采样边界明确；不执行render/tests，不改核心或他人文件。14:40原builder返回仅AST/空白检查通过；controller以唯一lab运行[20260905T144023Z-lane-completion-figure-3dc28cdd](../runs/20260905T144023Z-lane-completion-figure-3dc28cdd/status.json)，45s/32MiB/2048MiB、显式CSV/result输入，3.241902s正常。4artifact逐字节核对，PNG目视通过；PDF本地，CAP状态在CSV/阶段表、图中标代次与锁释放而非CAP事件。

独立解释实际只读重算确认F020第三代641.25s锁内，其中630.5s普通航点、628.25s仅侧误差失败（2513点全部越界），全部判据通过仍锁定0点；t1361.50侧误差−1.995739m时当步释放。第三代误差曾由−5.779288增至−5.995923m，再缓慢减小，实际航迹接近远端球面方位（重算最大差0.000033624度），支持远端归航残差慢收敛解释，未干预分离CAP/变速/几何因素。第四代60.25s侧误差失败但不越界；释放后总39.5s越界与2m容差并存，非永久锁或规则bug。详细实测数、证据和一个预声明500m引导航点配对建议在[F020报告](../reports/lane-completion-f020-20260905/README.md)，留给下一授权块，不本轮修改核心。

[最终四模式](../reports/policy-modes-refresh-850-20260905/README.md)run `20260905T142440Z-refresh-policy-modes-850-d139c180`14:31:27.894正常，CLI405.327780s/job407.029534s：0/850 × sample/argmax共48真实native例全通过，两sample逐case摘要/hist及aggregate精确；NR明确复用相同参考、0新增NR。850argmax295/360、0timeout65exhaust、321width，NMAC154.816/LoWC599.450每FH；initial0argmax240/360、30timeout90exhaust342width、175.887/765.717。无更新/模型选择/输入修改。sample为主，不能据argmax替作者声明部署方式。5数据副本/manifest及README保存。controller一处事后只读打印将modes列表误作字典报AttributeError，复制已完成；随后正确读取并写README，无实验失败或重跑。

[新曲线](../reports/refresh-learning-20260905/README.md)已扩展35实际0–850点，重复100/300的完整aggregate及全部NR精确一致，保留每行源SHA。render `20260905T143227Z-refresh-learning-figure-850-24c2df91`监督3.350697s，4artifact复制核对并目视通过，无新增科学实验；README更新最终850/best775/渲染来源。

14:41只读GPU快照原PID3059945/3062529仍9893/9771MiB，RTX4090 used20618/free3430MiB/99%；全块未提交GPU或信号/调整其进程，不能由快照宣称外部吞吐绝对零变化。当前无lab负载。14:40交route_trace最终只读review，<=12min/14:55前，明确850模型/四模式/曲线/F020/预算报告与源manifest，root同步文档；检查证据和解释、禁止实验/编辑/模型加载/信号/GPU/通知/嵌套，待返回。14:43已更新NOW为实际850终态，当前进入14:45收尾窗口，最终push与截止通知尚待实际完成。

14:46–14:48收尾只读检查：lab doctor入口/链接issues与local_missing均为空（系统Python3.10.12仅用于监督检查，不冒充项目3.11运行验证）；新probe/plot/test AST及两配置JSON通过。850恢复点/四模式/F020三manifest共12数据文件hash/字节数一致，两图全部6个输出hash/大小一致。事后通用绘图校验脚本首次假定outputs为字典，但learning元数据实际为列表，报AttributeError；按各自保存schema读取后全部通过，未修改元数据或重新渲染。37个明确科研文件暂存，两个明确小.pt强制加入，原始runs/PDF/私人目录未加入。staged diff检查通过，生成SVG原样保留尾空白。最终独立review及提交尚待。

14:49 route_trace最终只读review返回无阻断：两sample全12case/aggregate精确、NR相等、四模式数字、35点37源行、850轮预算、12清单数据及图hash均独立核对，F020逐代CSV统计和端点球面方位最大差0.000033623346度复算；两PNG目视。唯一CAP图文差异已修正，无模型反序列化/实验/编辑。controller已有doctor/manifest/staging检查通过，准备私有commit/push。另交observation_reward_spec只读将下一500m配对建议细化为实际航点映射/前缀重放/时间边界方案，<=10min且15:00前返回；明确只读必要source与保存CSV，不实现或增加本轮负载，不改容差/名义进度，不扩展诊断菜单。

14:50本批37文件提交`8e6e7cc921e34768b5d7ca9db5055217c5d1e892`已push当前私有研究分支，随后rev-parse与ls-remote精确相同。14:52仅核对三个指定子线程turn_context字段，shared_ppo第17、route_trace第19、observation_reward_spec第15turn实际均gpt-6-astra/xhigh，未读取认证或导出其它会话内容。

14:51只读盘点本块05:15:13后81份launcher status：75 succeeded、5 failed、1遗留running；5失败均已有原记录及相应修正/重跑，不删负结果。遗留running是07:13训练中断的原状态，不代表当前负载，已另存07:33:56 controller_observation：外层143、最后完整65轮、launcher锁确认空闲、当时精确主机只读审计无对应负载，原因/发送者未知。80份有结束时刻的监督时间合计19612.98863s，排除该中断缺失尾时和所有工具/读写/研究思考时间，不能当整个10h工作时长。完整有界区间未发现重叠；缺失区间不能由该简单盘点补成精确时间证据。最后lab任务14:40:26结束，当前无科研负载，进入只读方案整理/文档交付。

14:58 observation_reward_spec返回下一配对的只读实施细化，未编辑/加载模型/模拟：建议保存C02+F020单机固定响应（不复现30机风险/奖励），保留687实际入场/690首决策与0.25s偏移读取请求。188行前缀精确且两臂内态额外一致；t734记录后干预。500m固定投影guide映射nominal1，不增加原RouteGeometry顶点/名义进度或重置锁；原N1仍终点。发现addwpt内部direct的潜在共同干预，B只内部一次、A对N1匹配一次，A须精确保持原物理/目标/锁至1365s，否则归因无效。完整坐标、请求前缀、后段保持39选择、映射/事件顺序/停止条件已纳入F020 README和NOW。此为下一块待实施方案，不新增本块科研负载或现baseline配置变更。

15:03方案记录提交`836d4c8fb8b6c16dad078de5cef980a0596ed6b5`已push，rev-parse/ls-remote精确同SHA，工作区当时干净。只读共享非阻塞flock确认无lab独占负载持有者，立即释放探测锁；三个原生子agent均已完成，不中断/关闭他人实验。15:07:25结束前只读主机审计仍PID3059945/3062529、显存9893/9771MiB，RTX4090 used20618/free3430MiB/99%；load2.46/2.25/2.29、MemAvailable61.080GiB、项目盘空闲143.155GiB。无GPU计算或他人进程信号/调整；该快照不证明外部吞吐绝对无变化。所有科研负载最晚14:40:26结束，收尾只读研究方案已完成；本授权到15:15:13，不自动继续新实验。完成通知仅在实际截止后由主线程发送。

2026-09-05 15:15 UTC本块结束标记：本10h科研与交付完成，原任务001仍doing，因为有效baseline尚未获得。数据/模型提交8e6e7cc、方案836d4c8、资源收尾467bfd55c7e82733535d783431e5a12e881d9a9e均私有push并核对远端；没有活动lab负载或未结束子任务。下一步只保留F020配对机制诊断方案，不自动实施。主线程在实际15:15:13窗口到期后经已确认ntfy渠道发送完成通知，发送状态由通知工具本地去重记录保存。

2026-09-05 16:41 UTC用户追问航道中间节点，只读核对原文p7、paper_scenarios.generate_scenario、环境reset/ActionController._route及已保存`reports/policy-modes-refresh-850-20260905/scenarios.json`：原始路线0–3中间航点、总5NM、等长分段、各转角±90度以内，点实际注册到BlueSky；均匀抽样分布仍是明确重建选择。对保存JSON作标准库只读计数：12开发场景共48航道，0/1/2/3中间点分别11/12/14/11条。seed53004三条C01/C02/C03分别2/0/3中间点，F020属于C02，故该例CAP后下一点即远端终点，不代表生成器遗漏弯折航道。CAP为避让临时点，后续原名义航点按偏移路线保留。无新模拟、测试或模型加载。


### 2026-09-05 17:39 UTC：用户新授权解决无模型越界

范围NR执行定位/修正/小批验证；原10h已结束，不续PPO、不打开held-out，继续CPU14/1线程/nice15/idleIO/单lab。只读检索LESSONS中原Amzn半径、速度缓存、终止整步和Proxy教训，以及现报告/源码。保持Table3全部包络、bank25、航道500ft宽度、名义几何/入场/原始指标；候选自动弯前减速明确作为新执行重建，不伪称原文已披露。

原生Astra/xhigh、fresh bounded context新交接：nr_execution_design只读学者（问题最小物理合理NR修正；范围核心/BlueSky/原报告；输出半径/刹车/时序公式与兼容限制；<=15min，无修改/负载/嵌套）；nr_replay_probe builder（独占src/nr_containment_probe.py、对应配置和可选聚焦测试；问题原12saved scenarios无模型精确重放与逐机/段归因；输出有界CLI和检查，<=20min静态，无实验/清理/嵌套，须保留他人修改）。root独占新nominal_turn_speed.py及其配置/测试、集成/文档和所有launcher运行。两交接完整scope/source/output/stop已传，实际线程模型待元数据可用后核验。

只读源码关键点：原生reached先于本步速度更新，简单按目标半径+刹车距离减速可能晚于高速fly-by提前切换；制动触发需覆盖当前速度转弯提前量。ActionController._speed每物理步刷新CAS并关闭VNAVSPD，不能仅设置航点速度期待自动生效。候选独立NR-only子类保留原core/850身份，改变实际执行TAS目标而不改请求、真实位置或原生动力学；全部改变单列audit，未实测前不称已修复。

17:46–17:49进展：新增独立NR-only `NominalTurnEnvironment`，原共享环境/850源码不改。理想中心圆弧D=R(1−cos(θ/2))、T=Rtan(θ/2)，预声明分配0.5半宽、0.45相邻短段；当前及下一步可能加速的fly-by提前量与制动距离共同决定触发，原生按3.5m/s²及bank25执行。航点切换后保留低速，越过出弯切点且与后续目标方位对齐才释放。自动速度可低于策略最小离散档，明确是新重建选择，非作者2026设置或普遍安全保证。

[速度执行7项测试](../runs/20260905T174653Z-nr-turn-speed-tests-fb5e1d55/log.txt)及[被动审计7项测试](../runs/20260905T174711Z-nr-containment-tests-8e537551/log.txt)均由root经launcher通过；包括加速触发、跨航点保持/出弯释放、跨机/重置隔离、原始暴露归并/回调恢复/有界输出。只读独立nr_execution_review未发现阻断有限原生验证的缺陷，提示检查已激活但未释放限速，尚不能据fixture称原生修复成功。17:49仅核对三个指定线程自身session_meta/turn_context，nr_execution_design（01a072a2-6a1f-7e52-b529-f6483cb4cc20）、nr_replay_probe（01a072a3-cf24-7351-b88d-a0b94cdcc667）、nr_execution_review（01a072ab-26b2-75a3-bda2-565b638d31ee）各实际首turn均gpt-6-astra/xhigh，无认证读取或嵌套代理。

[原12case被动精确重放](../runs/20260905T174745Z-nr-reference-audit-37f08792/status.json)17:49:32正常结束，内部106.480645s/监督106.894981s；全部逐case科学摘要及aggregate与保存NR精确，回调/输入/样本归并检查通过。360/360到达、16架217.5aircraft-s横向越界、0高度，最大偏离412.215902m。360架覆盖全部12类型，31架Amzn中16架越界，其他329架在这批NR样本未越界；不能推为所有场景/策略下其他机型均安全。首次事后只读结果路径误写output而非artifacts，纠正后读取，无运行失败或数据改动。

17:53启动同12case候选配对，唯一lab `20260905T175338Z-nr-corner-speed-audit-dcf2ecd9`，外480s/64MiB/2048MiB、内450s，CPU14单线程。17:54再次交nr_replay_probe新独占`src/nr_turn_validation.py`及可选对应测试：全部12类型镜像90度中心弯、12类型直道原/新精确对照、Amzn最短论文段连续反向90度弯；输出物理实测速率/限速释放/原始containment及到达，<=12min仅静态编写，root运行。不得改前probe/core/config，保留他人修改，不做模拟/测试/清理/通知/嵌套。

17:55:39候选配对正常结束，内部120.071151s/监督120.454074s，792741个aircraft样本：360/360到达，0横向/高度越界、0超时/耗尽，最大距原中心折线38.670890m。41架被限速（20Amzn/15Cranfield/6Eh216），52次限速全部释放、0晚于名义航点切换才激活；全部360末态执行目标恢复巡航。FH55.051458333（+0.26624%、526.25aircraft-s），NMAC8924pair-s/162.102881每FH、LoWC36010.75/654.128902，均较原NR略升；不是避碰收益。原始暴露/退出整步不删减，全部场景/输入/回调/分层归并检查通过。[两结果逐字节备份](../reports/nr-containment-20260905/manifest.json)共9937192B，SHA核对，报告保留差异和NR-only复现边界。

17:56交nr_execution_review最终只读源码/两实际结果/14测试/manifest复核，<=10min，不做负载/修改/模型/信号/通知/嵌套，固定50验证另交builder不重复。17:59返回无阻断：逐项独立重算上述指标，原科学核心及共享输入一致，原NR科学摘要精确，52限速全部释放，319未覆盖飞机物理终态/航时/路径/暴露精确。NMAC+0.58261%、LoWC+0.86253%；微小跨层浮点重组差7e−15解释为加法顺序，原case聚合精确。提醒0晚激活标志不独立证明任意初态能在passage前达到cap，无连续/泛化/策略安全或作者身份结论。18:01只读两指定线程第二turn_context，builder与reviewer实际仍gpt-6-astra/xhigh；native read_thread未暴露模型字段，最终以匹配session元数据核验。

18:01固定验证静态检查发现巡航TAS往返CAS存在既有约9.34e−5m/s偏差（原Amzn参考80.664982261对请求80.664888889），向builder指出其1e−6巡航容差会产生已知假失败；改为明确独立1e−3巡航换算容差，Table3最大速度/加速度界不放宽。尚未运行固定native例，不是试验后调阈值。原生eps经安装源码核对为数组，采样按eps[0]读取。

18:02builder正式交付仅新driver/3helper tests，AST/空白检查，无负载。root唯一launcher [20260905T180219Z-nr-turn-validation-tests-acb3c913](../runs/20260905T180219Z-nr-turn-validation-tests-acb3c913/log.txt)3测试0.001s/监督0.120122s通过；随后[50原生固定验证](../runs/20260905T180229Z-nr-turn-native-validation-8d02777e/status.json)18:03:42.236正常完成，内部72.268830s/监督72.584296s、111168物理样本、38预声明fixture/50episode。24镜像90度中心弯覆盖全12类，最大距原折线36.991644m；12直道原/新共24次全科学摘要精确/新0覆盖；2 Amzn交替连续90度最短2315m段最大36.982423m。全50到达/0越界/物理Table3速度和加减速、bank25/高度/巡航恢复/回调/输入检查通过，最大实测绝对加减速度3.5m/s²，所需限速均激活并释放。直道最大到有限折线距离16.472943m是保留的出口后整步纵向距离，不称横向误差。

固定result/fixtures已逐字节复制加入[NR报告](../reports/nr-containment-20260905/README.md)manifest，4数据文件共10462657B。root只读核对两12case和固定50源快照中的环境/动作/性能/导航核心一致，完整代码未被本次NR改动；当前3模块/3测试静态与17fixture均通过，零模型/torch/GPU训练，无他人进程操作。独立review覆盖前述12case配对，固定50由root执行检查，不扩大review声明。18:04doctor入口/链接无issues/local_missing，diff空白检查通过；本次无失败lab负载，所有实验已结束。NOW缩为当前结果和稳定报告入口，详细历史保留本task；下一步统一策略执行语义，不自动续训。准备私有提交备份与主线程通知。

18:05–18:07收尾：6Python AST/2配置JSON、4备份原字节/SHA及6实际受测源文件与对应snapshot精确一致，doctor与staged diff检查通过。19个明确文件提交`62a2cb334c11762b0063271cb61c63aa430c9b5a`并push当前私有研究分支，18:06本地/远端SHA精确同值，工作区当时干净。三子任务已完成、全部lab最晚18:03:42结束。本次NR修正授权已完成；任务001仍doing，未达到有效MARL基线。当前仅同步NOW/本task收尾文档，主线程随后通过已授权`python3 -B tools/notify.py --complete`发送完成通知，实际发送/去重状态由工具记录，不据服务接受声称用户已收到。

2026-09-05用户后续追问NR不越界是否该由策略学习，并明确聚焦局部冲突消解。定向读取已保存[原文PDF](../resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf) pp.7–8、14–17，同时查询[期刊论文页](https://www.sciencedirect.com/science/article/pii/S0968090X26000306)确认研究定位。§4.1名义中心lane用于原计划跟踪，临时偏移用于冲突解脱；§4.2明确包括不受MARL控制、静态或动态意图UAS在内均假定遵守指定航道；§5.2.2战术调整保持计划路线并在航道内，§5.2.3以机间风险及指令效率为目标。因此本项目分工：名义跟踪为基础执行前提，解脱决策/执行保持航道合规为系统约束，NR本身不越界不是模型收益；零扰动/不越界不等于零机间冲突。PLAN与NOW补入此范围澄清；自动减速仍是未由作者核实的实现选择，当前NR-only成功不能代替所有策略共享执行链验证。只读原文/文档编辑，无新测试、模拟、模型加载或执行代码改动，不自动扩展成航迹控制研究。

2026-09-05用户追问是否“基础导航→动作控制器→基础导航”交接。只读核对`src/paper_actions.py` register/_speed/_altitude/_route/apply/update：所有阶段均通过同一BlueSky AP执行，ActionController修改目标和航点、监测机动；update完成条件只将altitude_active/lane_active置False，不重置target_speed/target_alt/target_lane或自动重建中心路线。故机动完成是保持新目标并解锁后续同类动作，回原名义目标需新指令；NR-only自动弯道限速的出弯恢复不等同于避让动作自动恢复。NOW记入这一区别；无代码修改、测试或实验，当前NR验证范围与未统一策略执行的限制不变。

2026-09-05 18:39用户问数学保证与导航来源。定向源码确认：PaperPerformance将Table3包络接入原生运动更新，ActionController通过selspdcmd/selaltcmd/route.addwpt/direct设置目标、CAP及偏移航点；navigation_refresh只在环境step期间包装原生reached，按实际TAS/bank/方位刷新普通fly-by提前距离，再调用原生判定一次并恢复；NR-only子类再覆盖速度指令为弯前减速/出弯恢复。不能将当前系统称为纯未修改BlueSky或全自研导航；原850使用refresh但未用NR-only限速。数学上可提出“全参考轨迹及已证明跟踪误差集合在走廊内”的充分条件；直航段可用绝对目标偏移加误差上界不超过半宽、垂向同理，来自三角不等式，尚未证明所需误差界。边界目标没有双向误差余量，不能因此无条件保证；也不能据此擅自收缩论文档位。保证还需安全可行初态、受限输入、航段/动作切换和采样/延迟条件。Ames2016及Singletary2020作者摘要入口已核对并记SOURCES，未逐条审定理或实施CBF。零新测试/实验/模型，只有说明和来源记录；现有17fixture/50native与原12NR结果仍为有限实验验证。

### 2026-09-05 18:44–19:00 UTC：数学推导交付

用户问“这个要做数学推导复杂吗？你能不能完成这个推导？”，授权完成具体推导与有界只读核验。此前10h已结束；本轮不实现安全过滤器、不启动模型/模拟/测试/GPU，不改变他人进程或当前执行代码。root独占新[CONTAINMENT_DERIVATION](../paper/CONTAINMENT_DERIVATION.md)及NOW/SOURCES/LESSONS/本任务文档；保留此前PLAN关于局部冲突解脱的范围澄清。

两次原生、显式gpt-6-astra/xhigh、fresh bounded context交接（均禁止嵌套）：

- `containment_proof_audit`：问题为当前执行链是否具备制动/转弯/误差推导所需的控制与更新语义；读写范围为只读src核心、配置、已安装BlueSky1.1.1与既有几何/NR报告；来源指向paper_actions/performance/environment/navigation_refresh/nominal_turn_speed及native traffic/AP/activewpdata；预期输出精确更新顺序、可证性质/反例和剩余证明义务；停止条件为一次有界源码审计，无修改、负载、通知或清理。线程`01a072e0-67ff-7983-adff-79ff28926ac1`，已从其匹配session元数据核验实际turn为gpt-6-astra/xhigh。
- `containment_proof_review`：问题为新推导的代数、可行域、几何、度量、混合切换及结论是否严格；范围仅只读新笔记、必要源码和两篇原文；来源为新笔记各式、nr_pilot/RouteGeometry/native traffic、Ames2016与Singletary2020正文；输出独立逐项通过/缺陷、反例和必要修订；停止条件≤12min、无修改/项目执行/嵌套。线程`01a072e7-9781-7d70-bdd6-e02251ab07b7`，实际首turn同样已由匹配元数据核验gpt-6-astra/xhigh。未依赖角色名称推断模型，也未读认证信息。

root完成六个带证明的条件命题：有界双积分位置集的不可控反例与精确制动可行域；全参考轨迹误差管充分条件；明确单模式Lipschitz动力学下的Gronwall误差界；已认证速度界下的相邻端点步内充分条件；可靠可达外包、末态安全续行集合K及非空认证动作条件下的组合归纳保证。另外给出五次平滑函数横移/升降/恢复构造与导数/理想控制界、一般内侧偏移圆弧最大距离和有限航段占用，以及直接对应native TAS分支的离散速度包络证明。简化模型和理想参考不是当前45°CAP/native导航的等价证明。

独立源码审计实际完成：核对physics .25s/decision5s与policy→native子步→动作进度/锁更新顺序；native TAS→heading→VS→位置以及航点通过的模式切换。TAS3.5m/s²不是可独立选择的横向/垂向制动权限，VS近目标直接赋值分支的阈值未乘dt，不能以1.524m/s²作全局有限差分界；高度捕获/舍入需离散证明。全部控制器目标、锁、航点/CAP映射和NR状态必须进入证明状态。审计独立推得内侧圆弧公式；未完成建议的原生垂向不变性证明。无代码修改或实验。

root还明确RouteGeometry固定原点投影与nr_pilot当前位置纬度评分的差别，推导对角矩阵尺度的上下界及保守证书半宽；实体航道宽度/动作/原始指标未修改。几何公式的精确值针对局部两航段，对完整折线仍是安全上界；若据此断言越界，须排除其它更近航段。内侧边界圆弧反例只否定该局部固定模板，不否定其它安全轨迹。完成谓词反例限定单东西直段、南向77.2m对目标76.2m，使两种度量一致；这是逻辑反例，无新native观测声明。

独立数学复核返回未发现阻断的代数缺陷，确认度量变换、双积分精确可行域、五次函数两个极值、理想切/法向加速度、半角圆弧、Gronwall与半步端点界、native TAS分支。提出三项明确化已全部集成：参考模型声明可独立选择切/法向控制；组合定理固定T>0/执行存在，认证下一状态集合由末态外包与可靠测量交集传播，再归纳属于K；完成反例明确东西直段。复核定向读两篇原文，确认其输入/正则性/增量稳定性/延迟和初始化条件；实际读到章节更新SOURCES。独立review与审计均只读，无测试/模型/实验。

结论：限定模型与条件的数学推导已完成首版，当前BlueSky全程安全证书仍未建立，缺实际跟踪界、混合切换可靠可达外包、可计算安全续行集合及始终可选的认证动作。NR360架/50固定验证不能替代这些假设，未将本推导称作创新或现60动作的保证。下一实证步骤仍为共享执行链的固定动作保持/过弯、横移/升降与恢复诊断，先取得有效无延迟baseline，再继续原研究路线；本轮不自动扩展实施。详细数学集中于专题笔记，避免在状态文件重建审批系统。

19:00收尾只读检查：`lab doctor`的issues/local_missing均空、`git diff --check`通过，仅文档6文件；检查不构成数学或科学验证，数学核验依据上述逐项证明与独立审阅。本轮零新lab负载，不重复执行既有17fixture/50native。两新子任务均已完成，准备按既有授权将这6份文档提交当前私有研究分支并核对远端；最终交付前由root调用既有`tools/notify.py --complete`，实际推送/通知结果以工具输出为准。
