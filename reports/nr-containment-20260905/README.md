# 无模型弯道越界修正

用户新授权先解决NR（No Resolution）越界；不续已结束的10h训练。原12个开发场景、48条航道、360计划架次配对结果：自动弯前减速使原始0.25s采样横向越界归零，全部到达。未打开held-out，不证明连续时间安全或避让策略有效。

| 原12开发场景 | 原NR | NR弯道速度执行 |
| --- | ---: | ---: |
| 完成／计划 | 360／360 | 360／360 |
| 超时／导航耗尽 | 0／0 | 0／0 |
| 横向越界架次 | 16 | 0 |
| 横向越界 aircraft-s | 217.5 | 0 |
| 最大距原中心折线 m | 412.215902 | 38.670890 |
| 高度越界 aircraft-s | 0 | 0 |
| 总 flight-hours | 54.905278 | 55.051458 |
| 总路径 m | 3323507.460312 | 3329784.800936 |
| NMAC 无向 pair-s | 8848.75 | 8924.00 |
| NMAC 无向 pair-s / flight-hour | 161.163924 | 162.102881 |
| LoWC 无向 pair-s / flight-hour | 648.535103 | 654.128902 |

总飞行时间增加0.26624%（526.25 aircraft-s），路径增加6277.340623m。冲突暴露略升；NR执行器没有交通感知或避碰机制，不能称冲突解脱提升。速率由总pair-s除以总flight-hours，不平均各场景速率。

## 归因与具体修改

原参考的31架Amzn中16架发生越界；其他11类共329架在这批NR样本未越界。这是固定开发样本的观察，不能推成永久安全机型分类。原参考精确重放后保存每机/原生当前航段的暴露和首/峰/末越界状态，可在[reference.json](reference.json)查询；航段分类是动作更新后的当前名义索引，不冒充最近几何段或因果标签。

新增[独立NR执行器](../../src/nominal_turn_speed.py)与[显式配置](../../configs/paper_environment_nr_turn_speed.json)，保留论文Table3速度/加减速包络、原生25°bank、500ft全宽、原始航点和计划入场、实际物理及原始暴露/终止记账。名义巡航请求保持原值，执行器仅向原生自动驾驶仪发送选定速度，原生逐步减速和转弯，不直接改实际速度或位置。

理想中心圆弧给出 `D=R(1−cos(θ/2))`、`T=Rtan(θ/2)` 和 `v²=g tan(bank) R`。预先固定用半宽的50%分配理想偏离、相邻短段的45%分配切线长度，保留几何与离散执行余量；90°弯约24.39m/s限速。触发同时考虑制动距离和**当前/下一步可能加速后的原生fly-by提前量**，避免高速状态先切换航点再减速。限速跨航点激活保留，越过出弯切点且航迹对齐后恢复名义巡航，连续弯取仍有效限速的最小值。

同12case实测41架被限速（20 Amzn、15 Cranfield、6 Eh216），52次弯道限速全部释放，0次在名义航点已切换后才开始限速；原始计划/入场和输入不变。其余飞机无速度覆盖。该触发标志不单独证明任意初态有充足制动距离。

**复现边界：** 原文没有明确披露此自动限速，且速度可低于策略最低离散档。它是本项目明确新增的NR沿航道执行选择，不是已核实作者2026实现，也不是后续论文的数学安全过滤器。原共享PPO环境/850轮模型未被静默更换；本类拒绝非None动作。在使用策略与新NR比较之前，须统一并验证执行语义，不能把旧850和新NR称为同执行链公平比较。内侧边缘航道和横移动作仍未由此修复。

## 验证与来源

- [原NR重放](../../runs/20260905T174745Z-nr-reference-audit-37f08792/status.json)：12个科学摘要和aggregate精确匹配原保存参考；790636个aircraft physics samples，内部106.480645s。
- [修正配对](../../runs/20260905T175338Z-nr-corner-speed-audit-dcf2ecd9/status.json)：相同场景和输入，792741样本，内部120.071151s；[完整结果](corner-speed.json)保留原始逐机终态、每段采样归并、限速全过程摘要、输入/source SHA与配对指标。
- [速度执行7测试](../../runs/20260905T174653Z-nr-turn-speed-tests-fb5e1d55/log.txt)与[被动诊断7测试](../../runs/20260905T174711Z-nr-containment-tests-8e537551/log.txt)通过，包括加速触发、保持/释放、状态隔离、原始计数与回调恢复。系统测试不充当科研结果。
- [固定验证3项helper测试](../../runs/20260905T180219Z-nr-turn-validation-tests-acb3c913/log.txt)通过，验证声明几何、同机连续加速度计算及未释放/旧机型/直道覆盖失败判据；本次合计17项聚焦测试。

[额外50次原生验证](../../runs/20260905T180229Z-nr-turn-native-validation-8d02777e/status.json)18:03:42正常结束，内部72.268830s/监督72.584296s，111168物理样本，全部检查通过。保存[38个预声明fixture](native-fixtures.json)与[完整结果](native-validation.json)，每例复用F001/C01后重置：

| 固定原生验证 | 次数 | 最大距原折线 m | 结果 |
| --- | ---: | ---: | --- |
| 全12类 × 左/右90°中心弯，2×2.5NM | 24 | 36.991644 | 全到达、0原始越界、所需限速均激活并释放 |
| 全12类5NM直道，原/新各一次 | 24 | 16.472943 | 12对科学摘要精确，新执行0速度覆盖 |
| Amzn连续+90/−90/+90°及镜像，4×1.25NM | 2 | 36.982423 | 全到达、0原始越界、全部限速释放 |

50例均符合Table3速度/加减速界、原25°bank及原高度界，最终真实巡航恢复；实测最大绝对加减速度3.5m/s²。通过连续同机前后TAS和实际时间差计算，不仅检查配置。航向增量反推的最大等效bank为25°（约1.8e−15rad舍入差），没有额外滚转动力学的声明。原生TAS↔CAS往返的已知约9.34e−5m/s偏差在运行前单列巡航比较容差0.001m/s，最大速度/加速度界另用1e−6容差。直道的最大距有限折线包含原出口后仍计入的末步纵向距离，不是16.47m横向偏离；原始指标未修改。

独立只读review核对实际12case配对、源/输入身份和17:46/17:47测试日志，无阻断；319未覆盖飞机物理终态/航时/路径/暴露精确，全部360末态执行目标恢复名义值。独立review没有运行项目代码，额外50例由root检查执行，其原始结果不冒称已由该次review复核。

上述原生任务均由root经lab/Bubblewrap固定只读源码运行，CPU14、单线程、nice15、idle IO，一次一个任务，零模型/torch加载、零训练更新、零GPU计算。终止整步暴露未删减；距离口径仍是局部投影下到原有限折线的距离。被动审计与原始逐机/场景指标精确归并，输入不变、回调恢复；所有检查通过和零越界是分别记录的结论。

[manifest](manifest.json)保存逐字节副本的源run路径、大小和SHA。原场景/NR输入沿用[已备份四模式报告](../policy-modes-refresh-850-20260905/README.md)，原训练和负结果保留。

重放已修正NR（以下启动器依赖项目uv环境）：

```sh
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label nr-corner-speed-audit --stage dev \
  --config configs/nr_containment_probe.json \
  --input reports/policy-modes-refresh-850-20260905/scenarios.json \
  --input reports/policy-modes-refresh-850-20260905/reused_nr.json \
  --seconds 480 --disk-mib 64 --memory-mib 2048 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  -- "$PWD/.venv/bin/python" -B -m nr_containment_probe \
  --config configs/nr_containment_probe.json \
  --environment configs/paper_environment_nr_turn_speed.json --label nr-corner-speed
```

原始NR精确重放时移除末行`--environment`参数并加`--require-reference`，配置默认仍指向原参考环境。
