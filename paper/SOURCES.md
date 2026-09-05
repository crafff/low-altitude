# 来源索引

## 必用材料

| 来源 | 位置与状态 |
| --- | --- |
| Fremond 等 TR-C 2026 原文 | [本地PDF](../resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf)，30页，已核对原SHA；[DOI](https://doi.org/10.1016/j.trc.2026.105542) |
| Fremond 等 ICRAT2024 前序论文与公开参数 | [本地PDF](../resources/literature/local/fremond-et-al-2024-urban-corridor-tactical-conflict-resolution.pdf)，8页；[身份/下载来源](../resources/literature/icrat2024.json)；[官方会议目录](https://www.icrat.org/previous-conferences/11th-international-conference/papers/)。PDF参考文献27直接链接[作者数据仓库](https://github.com/RodolpheFmd/ICRAT2024/tree/26ed1a0d128944643017e7ba5e9511dfc273a9bd)。2026-09-05读正文及少量参数CSV，未见导航源码；配置与正文有版本差异，不覆盖2026规范。 |
| 本轮原文核对 | [REPRODUCTION](REPRODUCTION.md)，逐项页码与重建歧义；不是作者开源实现 |
| 旧LS006及近邻调查 | [LS006](../legacy/recovery-20260904/tracked/literature/LS006-fremond-trc-2026-fulltext-audit.md)，保留研究内容，旧admission状态失效 |
| 其他既有检索 | [旧literature目录](../legacy/recovery-20260904/tracked/literature/)，定向读取，避免重搜全部主题 |
| 旧失败基线 | [训练结果](../legacy/recovery-20260904/verified-fit002/result.json)、[评估汇总](../legacy/recovery-20260904/verified-heldout/result.json)、[缺失清单](../legacy/recovery-20260904/heldout-recovery-verification.json) |
| 投稿目标 | [TR-C低空特刊](https://www.sciencedirect.com/special-issue/333589/intelligent-and-safe-operations-of-low-altitude-aerial-transportation-systems)；2026-09-05读[KU Leuven官方教师入口](https://feb.kuleuven.be/public/u0004371/)明确链接的[Roel Leus现行主页](https://sites.google.com/view/roel-leus)，联合客座编辑仍欢迎投稿并列截止2026-12-30。[ScienceDirect征稿列表](https://www.sciencedirect.com/browse/calls-for-papers?subject=computer-science)搜索索引也给出同日及编辑名单，索引标上周抓取，今日列表正文未读成功。特刊正文及[作者指南](https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies/about/guide-for-authors)仍403，具体时区、完整CFP、投稿系统选项及要求未知。早前[Elsevier期刊介绍](https://shop.elsevier.com/journals/transportation-research-part-c-emerging-technologies/0968-090X)只支持期刊名称/一般范围。 |

## 系统一手依据

- 2026-09-05定向检查作者公开旧实现：[BlueSky master](https://github.com/RodolpheFmd/bluesky/tree/849d76fd44880f8d17a69aefa0bd37208f2b2fbb)与[rotor参数patch](https://github.com/RodolpheFmd/bluesky/commit/2bba625488e70b983256a11d6df8c31ebd290a0c)、[DASC22](https://github.com/RodolpheFmd/DASC22_application/tree/8a613d34e01f8e4d631065ed4fbaa15e9db3e35d)、[通用PyTorch DRL](https://github.com/RodolpheFmd/Modelling-PyTorch-DRL/tree/d51db989df932d962e66dfe7ff76db2cc950370a)、[Research-Data](https://github.com/RodolpheFmd/Research-Data/tree/c9625e277855d6087faf1190a4a72b1659977fda)、[CBF/MVP示例](https://github.com/RodolpheFmd/Conflict-Detection-Resolution-methods/tree/a7c330aab37a0ae44512ce7b1c2b085498ffe974)。实际只读指定源文件/完整目录元数据，未运行程序或下载大训练数据；版本差异和不适用边界见REPRODUCTION末节。旧fork Amzn44m/s不能替代2026 Table3的196kt；普通flyby距离旧版每次重算与1.1.1缓存差异已由[速度变化及原生集成验证](../reports/navigation-refresh-20260905/README.md)支持显式current_state_refresh新训练谱系，仍不证明作者2026使用旧版或已满足走廊约束。

- [Codex完成通知](https://learn.chatgpt.com/docs/config-file/config-advanced#notifications)与[客户端通知差异](https://learn.chatgpt.com/docs/notifications)：2026-09-05实际打开官方页面，核对notify外部程序的agent-turn-complete/cwd/thread-id/turn-id字段、用户级配置位置，以及IDE依赖连接主机通知。另读[Hooks](https://learn.chatgpt.com/docs/hooks)以区分Stop/SubagentStop及信任机制，本轮直接复用现有notify，不增加hook信任设置。[ntfy官方发布接口](https://docs.ntfy.sh/publish/)核对POST和标题/优先级字段；服务接受不证明设备显示。实际实现/测试/本机路由见任务001与SYSTEM。
- [BlueSky官方仓库](https://github.com/TUDelft-CNS-ATM/bluesky)与[PyPI 1.1.1](https://pypi.org/project/bluesky-simulator/1.1.1/)：2026-09-05核实发布元数据并安装CPython3.11 Linux wheel。API以已安装1.1.1的`bluesky/__init__.py`、`core/{base,entity,simtime}.py`、`traffic/{traffic,route,autopilot}.py`、`traffic/performance/perfbase.py`为准，上游master可能不同。已验证detached headless步进与原生路线；没有据此指定原文的BlueSky版本或OpenAP模型。1.1.1强制依赖OpenAP是软件包事实，论文未指定它。
- [`zmq==0.0.0`官方源码分发](https://files.pythonhosted.org/packages/6e/78/833b2808793c1619835edb1a4e17a023d5d625f4f97ff25ffff986d1f472/zmq-0.0.0.tar.gz)：2026-09-05内存读取归档和setup.py，确认仅依赖pyzmq的元包；哈希见任务001。允许本次特定构建，不代表泛化信任任意源码安装。
- [OpenAI Codex配置](https://learn.chatgpt.com/docs/config-file/config-reference#configtoml)：2026-09-05重新读取官方配置参考，核对默认子agent模型/effort、角色config_file及每技能禁用设置；现有键有对应依据，无需增加配置层。Astra选择来自用户要求，不根据速度或成本自动降级。
- [Bubblewrap](https://github.com/containers/bubblewrap)：已读取，明确它是沙箱构造工具而非现成安全策略；本项目自己定义只读输入、可写输出、私有进程/临时目录与无网络边界。本机0.6.1启动和嵌套构造验证通过。
- [uv项目管理](https://docs.astral.sh/uv/guides/projects/)与[Python版本](https://docs.astral.sh/uv/concepts/python-versions/)：2026-09-05读取官方文档，并以本机uv 0.8.22的help核对命令选项；采用项目独立的托管Python、.venv、pyproject和uv.lock。具体命令见ENVIRONMENT，实际隔离验证见任务001。
- 历史系统曾参考AIDE、AI Scientist v2等；本轮没有重新验证这些框架，也不安装它们。采用原生Codex、简洁文件记忆和有界迭代，不声称复现其科研性能。

## 执行延迟与安全过滤近邻（2026-09-05定向核验）

原生Astra investigator只读以下三篇正文相关章节与官方代码/项目入口，不运行其代码。目的为后续公平对照和假设审查，不改变先取得有效名义baseline的顺序，也不以模块重合否决或宣告创新。

- Ramstedt、Bouteiller等，**Reinforcement Learning with Random Delays**，ICLR2021：[正文v3](https://arxiv.org/html/2010.02966v3)、[作者代码](https://github.com/rmst/rlrd)。显式增广延迟观测、近K个已发送动作及可测延迟；K覆盖最大合计延迟。随机通信案例保留最近生成的信息、丢弃晚到过期动作；DCAC基于SAC重采样尚未影响物理结果的历史动作以改善信用分配。理论不提供避碰保证；实测延迟MuJoCo及飞行机器人Wi-Fi时延。代码包装器额外含一步动作延迟，min_action_delay=0不能直接作为真正零执行延迟。后续须区分FIFO/覆盖语义，并在相同信息与训练预算下比较RNN与显式动作历史/时间戳；我们是否能测得同样的延迟状态仍待确定。
- Molnar、Kiss、Ames、Orosz，**Safety-Critical Control With Input Delay in Dynamic Environment**，IEEE TCST2023：[作者正式论文PDF](https://tamasmolnar.com/publication/2023_Molnar-et-al_safety%20with%20input%20delay%20in%20dynamic%20environment_TCST.pdf)、[IEEE刊期目录](https://ieeecss.org/sites/ieeecss/files/documents/pcd/css_publications_content_digest_072023.pdf)。控制仿射连续系统、已知常值输入延迟；用当前状态/输入历史预测生效时刻自身状态，同时预测动态环境，再施加环境CBF约束。定理3–4还要求初始等待区间已安全、约束可行；误差版本依赖预测误差与Lipschitz界。验证ACC/Segway数值案例，未确认对应官方代码。当前状态过滤、预测过滤、加误差余量可形成后续对照，但随机乱序、5s保持、60类离散动作、机动锁及异构BlueSky包络不在现成定理内。
- Zhang、So、Garg、Fan，**GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multiagent Control**，IEEE T-RO2025：[官方项目](https://mit-realm.github.io/gcbfplus/)、[正文v4](https://arxiv.org/html/2401.14554v4)、[官方JAX代码](https://github.com/MIT-REALM/gcbfplus)。同质连续多机系统、局部邻居与输入约束；图CBF、有限时域轨迹标注及CBF-QP教师训练分布式策略，无执行延迟队列。理论要求函数真实满足GCBF条件、邻域变换的光滑性和安全初始集；作者承认学得网络难获全域形式验证。包含二维/三维模拟与Crazyflie实机；附录time delay为相对到达耗时。后续可借鉴同输入界解析CBF-QP对照及安全/完成/死锁/不可行记录；把它改为冻结PPO过滤器须标为改编，异构/延迟假设另证。

新来源或来源改变判断时在本页补充链接、核验日期、实际读到的内容和未解决项；详细调查留在任务或专题笔记，不重复登记每次搜索。来源失效时明确说明，不编造读过的全文。
