# 原生路线执行诊断

[图 SVG](route_execution.svg) · [PNG](route_execution.png) · [图输入](figure-input.json) · [绘图身份记录](plot_metadata.json)

两条4630m等长航段构成5NM、90°左转；比较Mavic与Amzn的中心、起点内移、起点外移，以及Amzn在原生航点切换后的首个5s决策时刻内移。原始每0.25s轨迹来自 `runs/20260905T062257Z-route-action-seven-725d5871`，绘图来自 `runs/20260905T063934Z-route-study-figure-c67c9f69`。精简JSON保留全部9419个采样点、源CSV SHA及明确坐标换算，图不是示意飞行轨迹。

7例均到达，6例超过76.2m走廊半宽。Mavic中心最大偏离12.265m，边缘目标为88.473/88.840m；Amzn中心406.142m，内/外为474.220/348.829m。这说明完成任务和满足走廊约束须分别评价。该图不是已训练策略或算法效果，不能证明论文全部未知导航设置下都不可行。

走廊采用到有限中心折线的距离管道，因此转角外边界为圆弧、端点为圆帽；原文没有明确转角走廊形状，当前定义保留为重建选择。原生25°bank与Table3速度不变，内/外移的有限曲率轨迹不能自动保证边界。

复绘（主线程无其他launcher运行时）：

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label route-study-figure --stage dev --seconds 30 --disk-mib 32 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  --input reports/route-execution-20260905/figure-input.json \
  -- "$PWD/.venv/bin/python" -B -m route_study_plot \
  --input reports/route-execution-20260905/figure-input.json
```

输出位于新run的artifacts，不覆盖本图。PDF亦已生成并保留本地；Git中SVG/PNG供独立查看。
