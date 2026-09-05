# 当前状态

更新：2026-09-05 UTC。发布分支 `research/trc-baseline-system-20260905`。

## 当前任务与下一步

[001：无延迟、无扰动基线](../tasks/001-baseline.md)。BlueSky无控制headless诊断已接通；下一步实现原文Table3的12类无人机性能及纸面场景/入场/退出，再验证固定动作的速度、高度和横移后果。观测、奖励与共享PPO随后接入同一物理链路，尚不具备训练入口。

项目Python 3.11.13及BlueSky 1.1.1由uv独立管理并在实际沙箱中验证，配置/锁文件随Git保存，见 [环境说明](../docs/ENVIRONMENT.md)。系统启动器仍可用3.10.12；PyTorch尚未安装，CUDA尚未验证。

## 最新可信结论

- 系统重建及精简复核完成，见 [任务000](../tasks/000-system.md)。日常只维护NOW与当前任务，其余文档按事件更新；移除无消费者的index，新增doctor入口/断链检查；旧技能整套归档，退出自动发现路径。
- 最新原生诊断 [bluesky-nr-final](../runs/20260905T030644Z-bluesky-nr-final-b9a03656/artifacts/result.json)：单机转弯1/1完成；同高交叉2/2完成，LoWC/NMAC无向pair-seconds为28/8.75；相差200ft对照2/2完成、暴露为0。交叉场景reset后轨迹完全重现。使用继承原生PerfBase的诊断类，无机型包络限制，不是论文12类机型或有效MARL基线。
- 最新隔离回归 [bluesky-metrics-tests](../runs/20260905T030608Z-bluesky-metrics-tests-6f6949ec/log.txt)：25项通过、无跳过。Python实际身份见 [uv-environment](../runs/20260905T024306Z-uv-environment-49a6e0d4/artifacts/environment.json)。仿真可运行、指标单测、GPU可用及策略有效是不同结论。
- 旧模型没有改善冲突，新PPO/环境语义/完整评价尚未实现，本轮未训练新模型。旧结果与复现顺序见任务001；旧24个已看过场景仅可用于诊断。
- 原论文已恢复并核对身份，见 [文献记录](../resources/literature/manifest.json)。历史材料和恢复缺失情况见 [归档入口](../legacy/README.md)，不激活旧半成品。
- 目的、参照论文、四条改进方向和目标期刊集中在PLAN，原文参数/页码在REPRODUCTION。已重新核实TR-C期刊名称及一般范围；特刊正文和作者指南仍访问失败，截止日期/当前开放状态/细则未核实。

## 剩余边界

独立介质备份尚未建立。Git保存源码、配置和笔记；本地PDF、模型、运行产物和恢复归档不随push保存。重要长训练前落实第二份持久副本。

旧完整Git历史保留在本地 `research/trc-reboot-20260904`；其中历史大文件被GitHub拒收，因此发布分支从当前系统快照开始，详见任务000。

接手读本页、任务001与 [PLAN](PLAN.md)，再按需查 [复现表](REPRODUCTION.md) 和SOURCES。提交与远端同步状态直接查Git，不在本页重复维护。
