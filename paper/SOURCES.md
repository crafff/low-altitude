# 来源索引

## 必用材料

| 来源 | 位置与状态 |
| --- | --- |
| Fremond 等 TR-C 2026 原文 | [本地PDF](../resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf)，30页，已核对原SHA；[DOI](https://doi.org/10.1016/j.trc.2026.105542) |
| 本轮原文核对 | [REPRODUCTION](REPRODUCTION.md)，逐项页码与重建歧义；不是作者开源实现 |
| 旧LS006及近邻调查 | [LS006](../legacy/recovery-20260904/tracked/literature/LS006-fremond-trc-2026-fulltext-audit.md)，保留研究内容，旧admission状态失效 |
| 其他既有检索 | [旧literature目录](../legacy/recovery-20260904/tracked/literature/)，定向读取，避免重搜全部主题 |
| 旧失败基线 | [训练结果](../legacy/recovery-20260904/verified-fit002/result.json)、[评估汇总](../legacy/recovery-20260904/verified-heldout/result.json)、[缺失清单](../legacy/recovery-20260904/heldout-recovery-verification.json) |
| 投稿目标 | [TR-C低空特刊](https://www.sciencedirect.com/special-issue/333589/intelligent-and-safe-operations-of-low-altitude-aerial-transportation-systems)，2026-09-05正文及[作者指南](https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies/about/guide-for-authors)仍访问失败，不能核实截止日期/征稿状态/投稿要求；已读[Elsevier期刊介绍](https://shop.elsevier.com/journals/transportation-research-part-c-emerging-technologies/0968-090X)，确认期刊名称、ISSN 0968-090X和一般范围，不能替代特刊要求 |

## 系统一手依据

- [Codex完成通知](https://learn.chatgpt.com/docs/config-file/config-advanced#notifications)与[客户端通知差异](https://learn.chatgpt.com/docs/notifications)：2026-09-05实际打开官方页面，核对notify外部程序的agent-turn-complete/cwd/thread-id/turn-id字段、用户级配置位置，以及IDE依赖连接主机通知。另读[Hooks](https://learn.chatgpt.com/docs/hooks)以区分Stop/SubagentStop及信任机制，本轮直接复用现有notify，不增加hook信任设置。[ntfy官方发布接口](https://docs.ntfy.sh/publish/)核对POST和标题/优先级字段；服务接受不证明设备显示。实际实现/测试/本机路由见任务001与SYSTEM。
- [BlueSky官方仓库](https://github.com/TUDelft-CNS-ATM/bluesky)与[PyPI 1.1.1](https://pypi.org/project/bluesky-simulator/1.1.1/)：2026-09-05核实发布元数据并安装CPython3.11 Linux wheel。API以已安装1.1.1的`bluesky/__init__.py`、`core/{base,entity,simtime}.py`、`traffic/{traffic,route,autopilot}.py`、`traffic/performance/perfbase.py`为准，上游master可能不同。已验证detached headless步进与原生路线；没有据此指定原文的BlueSky版本或OpenAP模型。1.1.1强制依赖OpenAP是软件包事实，论文未指定它。
- [`zmq==0.0.0`官方源码分发](https://files.pythonhosted.org/packages/6e/78/833b2808793c1619835edb1a4e17a023d5d625f4f97ff25ffff986d1f472/zmq-0.0.0.tar.gz)：2026-09-05内存读取归档和setup.py，确认仅依赖pyzmq的元包；哈希见任务001。允许本次特定构建，不代表泛化信任任意源码安装。
- [OpenAI Codex配置](https://learn.chatgpt.com/docs/config-file/config-reference#configtoml)：2026-09-05重新读取官方配置参考，核对默认子agent模型/effort、角色config_file及每技能禁用设置；现有键有对应依据，无需增加配置层。Astra选择来自用户要求，不根据速度或成本自动降级。
- [Bubblewrap](https://github.com/containers/bubblewrap)：已读取，明确它是沙箱构造工具而非现成安全策略；本项目自己定义只读输入、可写输出、私有进程/临时目录与无网络边界。本机0.6.1启动和嵌套构造验证通过。
- [uv项目管理](https://docs.astral.sh/uv/guides/projects/)与[Python版本](https://docs.astral.sh/uv/concepts/python-versions/)：2026-09-05读取官方文档，并以本机uv 0.8.22的help核对命令选项；采用项目独立的托管Python、.venv、pyproject和uv.lock。具体命令见ENVIRONMENT，实际隔离验证见任务001。
- 历史系统曾参考AIDE、AI Scientist v2等；本轮没有重新验证这些框架，也不安装它们。采用原生Codex、简洁文件记忆和有界迭代，不声称复现其科研性能。

新来源或来源改变判断时在本页补充链接、核验日期、实际读到的内容和未解决项；详细调查留在任务或专题笔记，不重复登记每次搜索。来源失效时明确说明，不编造读过的全文。
