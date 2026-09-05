# 001 — 原文式无延迟、无扰动基线

状态：`doing`（uv/Python基础完成，BlueSky尚未接通）。负责人：主线程；实现时再指定builder的具体文件。目标顺序由用户确认：先有效基线，再冻结后加延迟。尚未执行新模型训练。

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

- 问题：在已经固定的uv/Python环境上安装并核实BlueSky依赖，跑通无控制headless小样，不训练。
- 写入范围：pyproject/uv.lock、环境说明、src中的新环境适配、必要小测试及配置；不修改legacy。
- 依据：本任务、REPRODUCTION、原PDF。
- 输出：确切命令、版本、实际步速与语义差异；不可运行时给出具体原因。
- 停止条件：环境与一条无控制rollout真实可运行，或15分钟DEV预算到达；模型训练另按本任务顺序继续，不使用旧未验证入口。

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
