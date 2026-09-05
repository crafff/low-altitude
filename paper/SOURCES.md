# 来源索引

## 必用材料

| 来源 | 位置与状态 |
| --- | --- |
| Fremond 等 TR-C 2026 原文 | [本地PDF](../resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf)，30页，已核对原SHA；[DOI](https://doi.org/10.1016/j.trc.2026.105542) |
| 本轮原文核对 | [REPRODUCTION](REPRODUCTION.md)，逐项页码与重建歧义；不是作者开源实现 |
| 旧LS006及近邻调查 | [LS006](../legacy/recovery-20260904/tracked/literature/LS006-fremond-trc-2026-fulltext-audit.md)，保留研究内容，旧admission状态失效 |
| 其他既有检索 | [旧literature目录](../legacy/recovery-20260904/tracked/literature/)，定向读取，避免重搜全部主题 |
| 旧失败基线 | [训练结果](../legacy/recovery-20260904/verified-fit002/result.json)、[评估汇总](../legacy/recovery-20260904/verified-heldout/result.json)、[缺失清单](../legacy/recovery-20260904/heldout-recovery-verification.json) |
| 投稿目标 | [TR-C低空特刊](https://www.sciencedirect.com/special-issue/333589/intelligent-and-safe-operations-of-low-altitude-aerial-transportation-systems)，本轮访问失败，具体日期/要求待官方核对 |

## 系统一手依据

- [OpenAI Codex配置](https://learn.chatgpt.com/docs/config-file/config-reference#configtoml)：2026-09-05重新读取官方配置参考，核对默认子agent模型/effort、角色config_file及每技能禁用设置；现有键有对应依据，无需增加配置层。Astra选择来自用户要求，不根据速度或成本自动降级。
- [Bubblewrap](https://github.com/containers/bubblewrap)：已读取，明确它是沙箱构造工具而非现成安全策略；本项目自己定义只读输入、可写输出、私有进程/临时目录与无网络边界。本机0.6.1启动和嵌套构造验证通过。
- 历史系统曾参考AIDE、AI Scientist v2等；本轮没有重新验证这些框架，也不安装它们。采用原生Codex、简洁文件记忆和有界迭代，不声称复现其科研性能。

新来源或来源改变判断时在本页补充链接、核验日期、实际读到的内容和未解决项；详细调查留在任务或专题笔记，不重复登记每次搜索。来源失效时明确说明，不编造读过的全文。
