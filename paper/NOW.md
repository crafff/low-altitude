# 当前状态

更新：2026-09-05 UTC。发布分支 `research/trc-baseline-system-20260905`。

## 当前任务与下一步

[001：无延迟、无扰动基线](../tasks/001-baseline.md)。12类机型包络、场景/入场退出和小批NR已实现。下一步先用保存的seed51005/F025路线追踪Amzn转弯偏离，固定合理的路线跟随语义，并核对入场即冲突；随后验证动作，再接观测、奖励与小训练。当前所有已观察场景仅作开发诊断。

项目Python 3.11.13及BlueSky 1.1.1由uv独立管理并在实际沙箱中验证，配置/锁文件随Git保存，见 [环境说明](../docs/ENVIRONMENT.md)。系统启动器仍可用3.10.12；PyTorch尚未安装，CUDA尚未验证。

## 最新可信结论

- [新NR诊断](../runs/20260905T044040Z-paper-nr-final-44ce1be4/artifacts/result.json)：8个新开发场景、240/240架次完成、0超时；LoWC/NMAC连续采样事件397/253，暴露27557.25/6285.25无向pair-s，累计38.9025 flight-hours。所有场景均有冲突；其中17/3个事件入场即出现。冲突信号不稀缺，尚不证明策略可学会解脱。
- 发现训练前需处理的跟踪问题：7/240架次（全部Amzn）超过走廊半宽76.2m，最大偏离363.15m，累计64.25 aircraft-s。原生fly-by和25°bank的有限转弯半径是有源码支持的解释，尚需逐步轨迹定位；到达率100%不证明走廊约束满足。最终NR的all_checks_passed仅指其列明的运行/计数检查。
- [12机型原生包络诊断](../runs/20260905T043537Z-paper-performance-probe-1319e270/artifacts/performance_probe.json)22个检查通过，实际测到最大TAS、爬升/下降及±3.5m/s²；原生bank、垂直加速度和高度捕获未改。[最新回归](../runs/20260905T044023Z-paper-environment-final-tests-d38dfe5b/log.txt)44项通过。尚无PPO训练或GPU实测。
- 主线程通知已接入[随Git维护的脚本](../tools/notify.py)，沿用已有ntfy订阅，用户已确认收到；子agent过滤、去重与重试测试通过。入口/恢复见[系统说明](../docs/SYSTEM.md)。
- 系统重建及精简复核完成，见 [任务000](../tasks/000-system.md)。日常只维护NOW与当前任务，其余文档按事件更新；移除无消费者的index，新增doctor入口/断链检查；旧技能整套归档，退出自动发现路径。
- 早前[3种构造诊断](../runs/20260905T030644Z-bluesky-nr-final-b9a03656/artifacts/result.json)保留作回归，不计为新增独立训练/测试场景。Python实际身份见[uv-environment](../runs/20260905T024306Z-uv-environment-49a6e0d4/artifacts/environment.json)。
- 旧模型没有改善冲突，新PPO/环境语义/完整评价尚未实现，本轮未训练新模型。旧结果与复现顺序见任务001；旧24个已看过场景仅可用于诊断。
- 原论文已恢复并核对身份，见 [文献记录](../resources/literature/manifest.json)。历史材料和恢复缺失情况见 [归档入口](../legacy/README.md)，不激活旧半成品。
- 目的、参照论文、四条改进方向和目标期刊集中在PLAN，原文参数/页码在REPRODUCTION。已重新核实TR-C期刊名称及一般范围；特刊正文和作者指南仍访问失败，截止日期/当前开放状态/细则未核实。

## 剩余边界

独立介质备份尚未建立。Git保存源码、配置和笔记；本地PDF、模型、运行产物和恢复归档不随push保存。重要长训练前落实第二份持久副本。

旧完整Git历史保留在本地 `research/trc-reboot-20260904`；其中历史大文件被GitHub拒收，因此发布分支从当前系统快照开始，详见任务000。

接手读本页、任务001与 [PLAN](PLAN.md)，再按需查 [复现表](REPRODUCTION.md) 和SOURCES。提交与远端同步状态直接查Git，不在本页重复维护。
