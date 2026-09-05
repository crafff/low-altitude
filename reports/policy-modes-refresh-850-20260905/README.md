# 新refresh850与原始模型的解码对照

原12开发场景上四模式共48次真实BlueSky评价全部通过，两sample逐case科学摘要/动作直方图及aggregate精确复现相应保存参考；只排除wall/RSS。模型/输入不变、0模型选择/更新/checkpoint写入。NR为已核对相同的保存参考，明确0次新增NR运行。sample仍为主指标，argmax是原60类合法联合动作的最大值，没有逐分量拼接。

| 模式（360计划架次） | 完成 | 超时 | 导航耗尽 | 越界架次 | NMAC pair-s/FH | LoWC pair-s/FH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| episode850_sample | 233 | 53 | 74 | 340 | 64.494 | 360.640 |
| episode850_argmax | 295 | 0 | 65 | 321 | 154.816 | 599.450 |
| initial0_sample | 246 | 53 | 61 | 339 | 78.926 | 435.953 |
| initial0_argmax | 240 | 30 | 90 | 342 | 175.887 | 765.717 |

NR参考360/360完成、16越界，NMAC161.164/LoWC648.535无向pair-s每flight-hour。全部模式0高度越界。四模式各自保留安全、任务失败、原始整步越界和flight-hours分母，不能只以较低风险率晋升模型。当前仍无可信有效baseline或held-out证据。

[完整结果](result.json) · [输入身份](input.json) · [文件SHA](manifest.json)。运行`runs/20260905T142440Z-refresh-policy-modes-850-d139c180`，内部405.328s/监督407.030s正常完成；5数据文件逐字节复制核对，原每case子目录仍在本地run。严格v2配置/原源码535cad4与软件/模型身份校验保持；测试已由前述8项v2fixture覆盖，此轮实际同850输入原生运行进一步验证。

复现使用`configs/checkpoint_policy_modes_refresh_850.json`，显式输入850 latest.pt及其development.jsonl、refresh100的原0 development.jsonl；调用`-m checkpoint_policy_modes --config ... --checkpoint ... --reference-trained ... --reference-0 ...`，一项lab job，CPU14单线程nice15/idleIO，内570s/外600s、128MiB/4096MiB。原文尚未明确实际部署解码方式，此诊断不替作者补充声明。
