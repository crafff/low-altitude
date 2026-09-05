# 观测扰动的最小重建规范

2026-09-05原文核对；独立[observation_perturbation感知层](../src/observation_perturbation.py)及[原生诊断driver](../src/perturbation_probe.py)已实现验证，尚未接入训练、无抗扰效果结论。主线仍先取得有效名义baseline。来源为[2026原PDF](../resources/literature/local/fremond-et-al-2026-resilient-marl-urban-air-conflict-resolution.pdf) pp.6、8–10、19、24，以及[2024原文](../resources/literature/local/fremond-et-al-2024-urban-corridor-tactical-conflict-resolution.pdf) pp.5–6和已存[公开配置](../resources/literature/author-configs-icrat2024/manifest.json)。一般范围见REPRODUCTION，此页记录明确的实现选择。

## 原文事实与歧义

2026 pp.9–10在固定高度虚拟移动水平感知位置，同一飞机的误差须一致影响其他飞机对它的相对观测；物理位置不动。Eq.11写sigma9.63/4.59m，Table1另写9.33/4.59；2024表为9.330/4.592，CSV的65%模型为9.2298948691985。因此采用2026 Eq.11，记录矛盾，不能说已恢复作者精确参数。

通信中断5/10/15s、概率25/50/25%，中断飞机对solver不可见且从相对观测中移除（p.10）。p.9逐步独立的说法只能用于发起事件，持续blackout自身并非iid。ownship不可见时是否停止推断、重触发是否延长、恢复用新数据还是缓存、6000ft筛选用真值还是受损位置，原文未充分说明。Algorithm1的奖励仍写正常观测，支持安全与奖励沿真值计算；缺失飞机不能免除物理风险处罚。

位置误差一维正态置信区间与二维位置向量之间缺乏协方差定义。p.10列出的受损字段包括LoWC/NMAC阈值，但Eq.21仅依赖未被扰动的垂直间隔。当前选择保持这两个阈值，用受损水平位置重算距离/CPA/航点距离及邻居筛选，保持已选[0,1]归一化。

## 明确的接口选择

- 每5s按稳定flight ID采样一次none/position/communication分类，概率之和≤1。关闭一种机制不放大另一种；两个概率为零不消费随机数。感知、场景与策略随机流独立，可保存恢复。同一时刻重复编码复用同一快照，禁止重采样。
- 位置采用东/北两轴独立N(0,sigma²)，所有观察者共用该飞机位移；不截断±9m，不由噪声位置差分生成速度噪声。9.63/4.59m是每轴标准差，二维半径9m内的概率约35.4%/85.4%，不能称二维65%/95%精度。
- 通信先触发、再形成可见集合：until=max(old_until,t+D)，available=(t>=until)。恢复使用当前新状态，不补发历史动作。连续黑屏可因多事件重叠超过15s；p=1可持续不可见。
- 诊断driver将不可见ownship从policy批次移除、继续执行末次接受的目标。这是“solver不可见”的明确重建解释，不是作者已确认实现。缺失决策不是终止；进入扰动PPO之前必须另行定义跨间隔奖励和折扣，不能伪造正常5s actor样本。
- 模块返回完整sensed states、可见ID、事件与until；实际BlueSky运动、到达/退出、奖励、风险、flight-hours均独立用真值。离散快照层只记录事件与可见性计数，不虚构实际aircraft-seconds，后者由物理步进按存活飞机与持续区间积分。

## 频率不能与持续占用混用

每架每5s独立触发概率p时，完整暴露窗口下期望发起率为720p次/flight-hour。这是推导，不是恢复的作者代码。2026 Table4位置6.48/h、专门通信2.16/h、FULL/NC通信5.76/h对应候选p=.009/.003/.008；2024正文/CSV的专门位置.01、通信.03与其TableII不能用同一5s换算统一。

须分别报告eligible aircraft-decisions、发起/重触发次数、连续中断段、不可见aircraft-seconds和真实flight-hours。按上述规则，p=.003的通信触发为期望2.16/h，但长期不可见比例约0.599%，不是0.3%。边界出生/退出和有限实验长度仍需实测计数。

最小校验包括：零扰动原观测/轨迹完全一致，sigma0退化；同一位移在所有观察者一致且真值/垂直阈值不变；同刻缓存和输入排序；5/10/15到期/重叠/reset；JSON状态恢复重复同一序列；通信不可见飞机仍计物理风险。接口校验不证明抗扰策略或真实通信可靠性。

07:34主线程的[15项纯层测试](../runs/20260905T073355Z-observation-perturbation-tests-8c19ee06/log.txt)和[5例native诊断](../runs/20260905T073432Z-perturbation-native-probe-b8a3c19f/artifacts/result.json)通过：零扰动编码、掩码、动作、物理轨迹及科学汇总与plain完全一致；100%通信中断0次推理、46次逐机保持决策，226.5真实不可用aircraft-s与物理暴露一致，保留LoWC39.75/NMAC12.25 pair-s。此为同一未训练模型的接口检查；不证明噪声下性能、经验发生率或抗扰训练收益。

## 归一化后的传感器字段缺陷

09:49已验证独立[encoded_sensor_faults](../src/encoded_sensor_faults.py)及[六例原生报告](../reports/encoded-sensors-20260905/README.md)。接口接受明确scalar slots和确定性FaultPlan：缺失填有限2（当前[0,1]约定的重建选择）；异常由调用者给出目标dtype下仍有限且在[0,1]外的值；虚假信息对至少两个指定scalar作非恒等双射，统一从未修改的clean快照读取。全量验证后才复制写入，拒绝重复/重叠slot，保留shape、dtype、ID、mask和源输入。记录计划项、应用项、数值/bit改变数量；完整邻机行置换另标attention集合不变，不当有害扰动证据。

16项fixture通过。实际6例同一未训练模型中plain、空计划、显式关闭逐bit保持编码/动作/采样RNG/物理及科学汇总；非零缺失own speed→2、异常→−.5、speed/alt交换均实际到达原策略forward，合法logits/value有限，原始真值保持独立、全部2/2到达。缺失例LoWC/NMAC32.5/0，其他例26.5/4 pair-s；这只是单一构造例的响应，不能解释为缺失有益或鲁棒性效果。未加入随机事件过程、发生率、训练、通信缺失决策GAE或非合作交通。
