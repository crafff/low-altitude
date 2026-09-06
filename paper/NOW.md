# 当前状态

更新：2026-09-06 03:10 UTC。分支`research/trc-baseline-system-20260905`。

[001：无延迟、无扰动基线](../tasks/001-baseline.md)进行中；先有效paper-like BlueSky MARL，再冻结测延迟，最后研究改进。[目标/论文/期刊](PLAN.md)。当前出现初步学习收益，尚未建立稳定有效基线。

## 最新：2400s新谱系0→256完成

[报告/核算/曲线](../reports/timeout2400-train256-20260906/README.md)。用户授权的首段已完成，未自动继续1024。独立[训练配置](../configs/paper_train_training2400.json)采用2400s保护/0.1mm出口容差，从头seed950001、新训练场景9600000起；四CPU采样/PPO单线程，无GPU。原1200配置/旧模型不改，无新观测、奖励、过滤器或PPO调参。

原12DEV/360架：0/64/128/256到达349/353/350/354，均0超时；NMAC86.918/94.283/85.362/77.419秒/FH，LoWC465.444/447.179/435.957/425.449。最终比初始到达+5、NMAC率−10.93%、LoWC率−8.59%，累计暴露亦下降；64先变差，尚不能称收敛。四候选均过95%完成筛选，best=256。

NR恒为360/360、NMAC162.10；未训练模型已低于NR，全部差距不是学习收益。旧冻结256在新2400评价为348/360、NMAC83.58；与本轮训练场景种子不同，不构成纯时限消融。原始越界仍保留，用户要求搁置全程包含研究未变。

累计256场景/7680计划架次、1156809样本、18104 Adam步；平均13.03s/四场景。两训练和独立归约均成功，监督合计1682.88s（约28分钟），恢复64科学结果精确、14模块来源不变、best内嵌/独立状态核对通过。所有负载03:07:46 UTC结束，其他两实验03:09核对仍为原启动实例。

## 恢复点与下一步

新[latest256](../reports/timeout2400-train256-20260906/checkpoints/latest.pt)及[best256](../reports/timeout2400-train256-20260906/checkpoints/best.pt)已保存；在同一新配置下可严格恢复。建议下一段保持设置继续至1024，检查趋势，再扩大训练种子和未见场景评价，不提前进入延迟结论。未继承旧10h授权。

[出口/时限校准](../reports/terminal-calibration-20260906/README.md) · [旧1200模型](../reports/parallel-continue256-20260906/README.md) · [并行接入](../reports/parallel-pilot-20260905/README.md) · [环境](../docs/ENVIRONMENT.md) · [REPRODUCTION](REPRODUCTION.md) · [LESSONS](LESSONS.md)。详细运行/备份证据在任务。
