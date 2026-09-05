# 当前状态

更新：2026-09-05 UTC。发布分支 `research/trc-baseline-system-20260905`。

## 当前任务与下一步

[001：无延迟、无扰动基线](../tasks/001-baseline.md)。下一次先落实干净的BlueSky执行环境与原文语义，跑一条分钟级无控制headless rollout，测步速和资源；首批≤15分钟、≤2GiB输出，不直接开始训练。

当前 `/usr/bin/python3` 实测为3.10.12；新的BlueSky/PyTorch环境和CUDA尚未验证。不要沿用旧环境路径或上轮3.11检查记录推断可用性。

## 最新可信结论

- 系统重建及精简复核完成，见 [任务000](../tasks/000-system.md)。日常只维护NOW与当前任务，其余文档按事件更新；移除无消费者的index，新增doctor入口/断链检查。
- 最终隔离回归 [system-simplify-final](../runs/20260905T022553Z-system-simplify-final-87328e2c/log.txt)：22项通过、无跳过。修正了runtime边界、源码遗漏与可执行位问题。只证明本轮系统行为，不代表GPU、BlueSky或基线有效。
- 旧模型没有改善冲突，新PPO/环境语义/完整评价尚未实现，本轮未训练新模型。旧结果与复现顺序见任务001；旧24个已看过场景仅可用于诊断。
- 原论文已恢复并核对身份，见 [文献记录](../resources/literature/manifest.json)。历史材料和恢复缺失情况见 [归档入口](../legacy/README.md)，不激活旧半成品。

## 剩余边界

独立介质备份尚未建立。Git保存源码、配置和笔记；本地PDF、模型、运行产物和恢复归档不随push保存。重要长训练前落实第二份持久副本。

旧完整Git历史保留在本地 `research/trc-reboot-20260904`；其中历史大文件被GitHub拒收，因此发布分支从当前系统快照开始，详见任务000。

接手读本页、任务001与 [PLAN](PLAN.md)，再按需查 [复现表](REPRODUCTION.md) 和SOURCES。提交与远端同步状态直接查Git，不在本页重复维护。
