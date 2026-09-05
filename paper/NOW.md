# 当前状态

更新：2026-09-05 18:07 UTC。分支 `research/trc-baseline-system-20260905`。

## 当前任务

[001：无延迟、无扰动基线](../tasks/001-baseline.md)仍进行中。目标是先得到有效paper-like MARL基线，再冻结模型测执行延迟，之后研究改进；[目的、TR-C期刊/特刊和路线](PLAN.md)。**现有850轮模型仍无可信有效baseline，不冻结、不开始延迟收益实验。**

用户新授权“先解决一下无模型状态越界”。本次仅NR执行修正与有限开发验证；此前05:15:13–15:15:13 UTC的10h已结束并发送通知，不继承其预算或自动续PPO。未打开held-out。

## 无模型越界：修正与验证完成

[完整报告与原始数据](../reports/nr-containment-20260905/README.md)：原12开发场景、48航道、360架；原参考逐case/aggregate精确重放。原16架越界均是Amzn（共31架），不推成永久安全机型分类。

| 同12开发场景 | 原NR | 新NR弯道速度执行 |
| --- | ---: | ---: |
| 到达／计划 | 360／360 | 360／360 |
| 横向越界架次／aircraft-s | 16／217.5 | 0／0 |
| 最大偏离中心折线 m | 412.216 | 38.671 |
| 高度越界／超时／导航耗尽 | 0／0／0 | 0／0／0 |
| 总flight-hours | 54.905278 | 55.051458 |
| NMAC 无向pair-s / flight-hour | 161.163924 | 162.102881 |

新增[显式NR-only配置](../configs/paper_environment_nr_turn_speed.json)与[执行器](../src/nominal_turn_speed.py)：按弯角/宽度/原生bank计算限速，提前制动、跨航点保持，出弯恢复巡航。保留Table3物理包络、bank25、500ft全宽、原始航道/入场和完整暴露/终止口径；不直接改真实速度或位置。41架发生自动限速，52次弯道限速全部释放，全部360架末态执行目标恢复名义值。319架无覆盖飞机物理终态/航时/路径/暴露与参考精确。

代价是总航时增加0.26624%，冲突暴露略升；没有避碰收益声明。此连续限速可低于策略最低离散速度，**不是已核实作者2026实现，也不是数学安全过滤器**。原共享PPO核心/850身份保持，本类拒绝策略动作；旧850不能直接与新NR声称同执行链公平比较。当前结论限于NR开发集0.25s采样，内侧边缘lane和横移动作仍需独立解决。

已通过17项聚焦fixture、独立只读配对复核和额外50次原生验证：12类×左右90度中心弯共24次、12类直道原/新共24次且12对科学摘要精确、Amzn连续反向90度弯2次。全部到达、0原始越界、限速释放并恢复实际巡航，速度/加减速/bank/高度界通过；实测最大加减速3.5m/s²。4份结果/fixture逐字节备份共10462657B，manifest保留源run/SHA。详细交接、实际Astra/xhigh核验与run结果均在任务末节。所有lab负载最晚18:03:42结束，三个子任务均已完成；代码/证据已私有push并核对远端，主线程在最终交付前调用既有完成通知入口，服务返回由工具去重状态记录。

## 后续顺序与保留证据

本次NR修正已完成。后续策略执行应统一自动驾驶规则，验证速度请求/横移动作/观测/效率奖励语义，再决定新训练；不把修复NR等同于MARL基线有效。原[F020临时引导航点配对方案](../reports/lane-completion-f020-20260905/README.md)保留，尚未实施。本次不自动扩展到该后续工作。

此前新refresh谱系已完成850轮（原文250k预算的0.34%），best775到达260/360、越界336；最终850sample到达233/360、越界340、47582.5aircraft-s，argmax到达295/360、越界321。较低冲突率伴随任务失败，未晋升有效baseline。[850模型/Adam/RNG](../checkpoints/refresh-850-20260905/README.md) · [35点学习曲线](../reports/refresh-learning-20260905/README.md) · [最终四模式](../reports/policy-modes-refresh-850-20260905/README.md) · [预算](../reports/training-budget-20260905/README.md)。旧400独立保留，不与新谱系相加。

[导航刷新验证](../reports/navigation-refresh-20260905/README.md)、[50固定动作响应](../reports/route-response-20260905/README.md)、[理想弯道几何](../reports/turn-geometry-20260905/README.md)解释执行背景。扰动/延迟线仅有[源规范](PERTURBATIONS.md)与[六例传感器编码](../reports/encoded-sensors-20260905/README.md)，无扰动PPO或抗扰收益结果。

## 运行与备份边界

所有新测试/实验由root经lab/Bubblewrap固定只读快照，CPU14、单线程、nice15、idle IO，一次一个job，不向GPU提交计算，不向他人的两个实验进程发信号或调整资源。此前15:07只读快照PID3059945/3062529仍在；不能由快照证明外部吞吐绝对零变化。

uv独立管理Python3.11.13、BlueSky1.1.1、torch2.9.1+cpu；[环境](../docs/ENVIRONMENT.md)与[系统/通知](../docs/SYSTEM.md)。CUDA先前小核通过仅证明可用性。旧143中断原因/发送者仍未知。

本次NR代码/验证/证据19文件已随`62a2cb334c11762b0063271cb61c63aa430c9b5a`私有push，18:06 rev-parse与ls-remote精确相同；本收尾状态随文档提交保存。此前850数据/模型及报告已私有备份。原始runs、本地PDF及历史恢复档未全部异地备份，不能称项目全量备份。详细历史只保留于[任务001](../tasks/001-baseline.md)。

## 按需入口

[REPRODUCTION：原文/重建差异](REPRODUCTION.md) · [LESSONS：经验](LESSONS.md) · [SOURCES：来源](SOURCES.md) · [DECISIONS：选择](DECISIONS.md)。旧审批链不再启用。
