# 当前状态

更新：2026-09-06 06:20 UTC。分支`research/trc-baseline-system-20260905`。

[001：无延迟、无扰动基线](../tasks/001-baseline.md)进行中。[目标/论文/期刊](PLAN.md)。当前：三个训练随机种子已完成，严重冲突泛化收益仍不稳定，不进入延迟结论。

## 最新：两个独立种子并行重复完成

[报告/协议/曲线/独立核算](../reports/independent-seeds-20260906/README.md)。新增seed950002/950003完整随机流程（初始化/场景/动作/shuffle均独立），各1024场景、原超参和2400环境不变。用户要求并行后，单lab内两个trainer各四核（8–11与12–15），无GPU。原自有初始化作业中断记录及0训练恢复点保留，其他实验未操作。

原DEV选best分别seed2=1024、seed3=896，latest均1024。固定已见24场景面板相对各自未训练：seed1 NMAC+11.18%/LoWC−2.67%/到达+7；seed2 −13.71%/−10.77%/+5；seed3 −4.23%/−6.37%/−5。冻结到达分别706/704/697（分母720，均0超时），NMAC89.34/73.36/83.73秒每FH。两降一升，按seed等权平均NMAC−2.25%、LoWC−6.60%，只是三seed描述性结果，不称稳定泛化或收敛。

所有策略比NR低，但未训练本身已好于NR，不能把全部差距算学习。seed3冲突下降同时到达少5。固定24未参与调参或模型选择，今后若用于改进诊断需要另外的全新确认集。

## 执行、模型与下一步

两个训练06:06:53结束，监督77.38min；并行外部评价8.48min，独立核算06:17:43通过。包含切换并行时仅自有作业的中断55.31秒，合计监督5211.11s（86.85min），未超原13440s。40份12399686字节备份，场景/样本/PPO步数、配置来源、模型选择和best恢复状态核对。全部负载已结束，06:18保护两实验仍原进程，未干预。

[seed2 latest](../reports/independent-seeds-20260906/checkpoints/seed2/latest.pt) · [seed2 best](../reports/independent-seeds-20260906/checkpoints/seed2/best.pt) · [seed3 latest](../reports/independent-seeds-20260906/checkpoints/seed3/latest.pt) · [seed3 best](../reports/independent-seeds-20260906/checkpoints/seed3/best.pt)。旧seed1保留。

下一步先基于现有训练/DEV记录分析严重冲突改善不稳定的原因（奖励/NMAC对应、动作和失败分布），再决定一个预先固定的小对照，不直接挑最好seed扩训。原始越界指标保留，用户搁置过滤器前置不变。没有继承旧10h。

[首轮新24](../reports/frozen-new24-20260906/README.md) · [原seed训练](../reports/timeout2400-continue1024-20260906/README.md) · [环境](../docs/ENVIRONMENT.md) · [原文核对](REPRODUCTION.md) · [LESSONS](LESSONS.md)。详情在任务。
