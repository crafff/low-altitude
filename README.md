# TR-C：先复现有效基线，再研究执行延迟

复现 Fremond 等（2026）的无延迟、无扰动 MARL 冲突解脱基线，再在同一个冻结模型上测量动作延迟造成的退化，验证时序记忆、预测、动作调度与安全过滤。

## 从这里开始

- [当前状态与下一步](paper/NOW.md)
- [论文路线](paper/PLAN.md) · [原文复现核对表](paper/REPRODUCTION.md)
- [协作与记忆](docs/SYSTEM.md) · [运行安全边界](docs/SAFETY.md)
- [原论文 PDF](resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf)
- [历史及恢复归档](legacy/README.md)

## 日常只需要几个命令

```bash
python3 -B tools/lab.py status
python3 -B tools/lab.py search "奖励"
python3 -B tools/lab.py doctor
```

执行项目测试（实际命令在独立沙箱中运行）：

```bash
python3 -B tools/lab.py run --label system-tests --stage system --seconds 90 --disk-mib 256 -- /usr/bin/python3 -B -m unittest discover -s tests -v
```

`status` 显示当前任务和最近运行，`search` 即时搜索活动文件，`doctor` 检查入口与本地链接并区分未随 Git 保存的本地材料。最新进度只维护在 NOW 和它链接的任务中。

每次执行自动留下 `runs/<id>/` 的命令、源码快照、配置、状态和日志；不需要手动建 Run 对象、绑定哈希或走审批链。具体研究命令见当前任务。

使用 Codex 原生 Astra 子 agent。协作约束以 [AGENTS.md](AGENTS.md) 为准，文档按内容变化维护，不逐轮重写所有文件。

Git 保存源码、配置和研究笔记。PDF、模型、运行产物和恢复归档仅在本地；新 clone 中这些链接可能缺失，获取方式见 [文献说明](resources/literature/README.md) 和 [归档说明](legacy/README.md)。Git push 不等于这些材料已有备份。
