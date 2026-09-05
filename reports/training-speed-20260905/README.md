# CPU / 多核 / 共享GPU训练速度诊断

2026-09-05完成；有界吞吐诊断，未开始收敛训练或持有测试集评价。当前主训练入口保持CPU单环境。

当前链路用CPU单线程更快；有明显收益的是独立BlueSky环境进程。建议后续用4个CPU环境进程采样、CPU单线程更新。并行训练需要显式固定每批采样后的更新节奏并验证学习效果，不能把本次采样3.5倍直接写成训练3.5倍。

## 相同PPO与完整单场景

真实30机场景seed920001、同一初始未训练模型，3726样本、59个minibatch、batch64/epoch1。网络35325参数，最忙一次推理29架；own7/intruder10/60动作、共享导航/奖励不变。每个完整场景30架都记录终止，其中21到达，不能当有效基线。

| 构建 / 设备 | 计算线程 | PPO更新中位数（秒） | 29架推理与采样（毫秒） | 完整采样＋更新（秒） | 样本/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU wheel | 1 | 0.149 | 0.703 | 8.507 | 438.0 |
| CPU wheel | 2 | 0.152 | 0.740 | 8.639 | 431.3 |
| CPU wheel | 4 | 0.166 | 0.781 | 9.181 | 405.8 |
| CUDA wheel / CPU | 1 | 0.154 | 0.734 | 8.699 | 428.3 |
| CUDA wheel / GPU | 1 | 0.661 | 4.286 | 11.198 | 332.7 |

更新每臂三次重复，均从同一CPU初始权重、空Adam和相同CPU shuffle种子恢复；有warmup，计时包含完整collation、拷贝、校验、梯度、Adam新状态分配及CUDA同步，不含模型构造。推理先warmup5次、再30次平均，包含CPU采样与传输。完整采样＋更新每臂仅一次，不含BlueSky/模型冷启动，也不含开发评价、checkpoint或日志备份；这是小样本诊断，不是长期吞吐预测。各次原始时间见[summary.json](summary.json)。

CPU参考完整流程约8.51秒，其中环境step约8.22秒、更新约0.15秒。当前瓶颈在CPU仿真/环境。GPU基线保持原有限值检查和Python标量读回，未做CUDA异步优化；小模型及同步/传输开销是GPU无收益的合理解释，本次没有单独剥离各项成本。共享GPU利用率约98–99%，这些数字不代表独占4090。

PyTorch均2.9.1，额外CUDA构建在CPU运行的对照约8.70秒、与CPU构建参数/Adam输出精确一致。float32 IEEE、无AMP/TF32、Adam foreach=False/fused=False。CPU初始化、行动采样和shuffle RNG机制相同；GPU前向后整批logits/value回CPU，再按原方法采样。CPU1/2的固定更新与参考精确一致；CPU4最大参数差1.49e-8；GPU参数差4.84e-8、Adam树张量最大差1.49e-8，各汇总指标最大绝对差5.96e-8。通过的是这批输入的数值检查，未宣称所有输入/长程训练逐位等价。

各完整场景的“飞机ID/决策序号/动作/奖励/终止”哈希一致；该哈希不包含全部观测、value、log-prob或GAE。固定更新使用同一保存的CPU张量批次，因而不依赖这种较窄的轨迹哈希来保证更新输入相同。

## 多进程环境采样

固定初始策略，四个诊断场景seed920001–920004、15942样本。每个子进程独立BlueSky、一个计算线程、独立物理CPU核；spawn，CUDA不进入worker。场景按seed核对样本数及上述动作/奖励/终止哈希，三个worker数均一致。

| 环境进程 | 冷启动（秒） | 四场景采样（秒） | 样本/秒 | 采样加速 | worker峰值RSS之和（MiB） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.96 | 35.52 | 448.8 | 1.00× | 578 |
| 2 | 12.26 | 18.07 | 882.4 | 1.97× | 1146 |
| 4 | 12.81 | 10.14 | 1571.6 | 3.50× | 2232 |

采样计时从所有worker已建立环境后开始，到父进程收到全部摘要为止，包含reset和摘要IPC；冷启动单列。尚未测全批训练张量的进程间传输，也没有将多场景池化后做PPO更新。RSS为各worker历史峰值相加，不是精确同时占用，也未包含父进程。现有训练每个episode后更新一次；若改成4场景共用一次策略再更新，属于明确的训练调度变体。

## 资源与保护证据

- AMD7950X：16物理核/32逻辑核；单线程CPU14，2线程CPU14/15，4线程CPU12–15，pool各worker依次固定CPU12–15。nice15、idle I/O、BLAS与interop单线程；只有模型intra-op按臂改变。未占满主机全部核。
- RTX4090/驱动580.173.02，独立uv环境torch2.9.1+cu128；原CPU环境和锁未变。全部实验经原lab/Bubblewrap、运行时只读，9个job严格串行。GPU job的64GiB是每进程虚拟地址上限，自己的RSS有2GiB软停止检查；CPU每进程地址上限4GiB（CUDA-wheel CPU桥为8GiB）。
- 启动前实际GPU free3162MiB；运行24次遥测最小free2665MiB。我们的GPU进程PID2625830最高观测492MiB，torch峰值allocated21.23MiB/reserved24MiB；256MiB allocator限额不包含上下文/库。轮询余量不是为他人预留显存或无限期保护。
- 两个原有PID3059945/3062529在全部GPU遥测中均存在，分别保持10045/9887MiB；[测试前](host-before.json)与[测试后](host-after.json)的进程启动ticks相同，没有退出重启。未向它们发信号、改优先级或读写其工作区。未测其训练进度/延迟，不能据存活推导无短暂算力竞争。结束GPU used20886MiB/free3162MiB恢复测试前水平。
- watchdog监测失败、保护PID缺失、低余量或自己的RSS越限只退出本测速进程；错误日志写失败也保证执行自己的退出。CPU池清理仅持有自己创建的Process对象。无遗留训练负载。

## 验证与复现材料

最终29项聚焦测试通过（PPO16、训练入口10、测速3），包括CPU默认行为、多线程数值检查、模型/输入设备不匹配、CPU采样适配一致、保护PID缺失不初始化CUDA，以及日志失败仍停止自己的负载。实际CUDA前向/梯度/Adam和采样已在上述GPU job运行通过。两独立uv环境的locked/offline/check均报告无需改变；主要科学包禁源码构建。

准备中两次只读环境检查失败已保留解释：第一次遗漏项目UV_CACHE_DIR，访问全局缓存只读失败；一次用全局--no-build检查，触发zmq==0.0.0仅有sdist的已知限制。均未启动负载或更改主环境；改用项目缓存及既定按包禁构建选项后成功。未出现失败lab job。

独立审阅发现并修复watchdog写日志失败可能绕过退出、CUDA初始化后缺立即余量复查两项问题；对应回归与实际运行均由主线程执行。代理实际Astra/xhigh身份、交接和权限范围见[任务](../../tasks/001-baseline.md)。

[备份索引](backup-index.json)：54个唯一运行结果/日志/manifest/使用过的PPO与测速源码/固定夹具文件，共4731768字节，逐字节对源文件核对。完整source快照和BlueSky临时缓存仍在本机runs，未声称全量异地备份。固定夹具[fixture.pt](fixture.pt)是本项目自己产生的CPU张量，weights_only=True读取；其SHA记录在索引。

入口：[测速实现](../../src/training_speed_probe.py)、[PPO适配](../../src/shared_ppo.py)、[环境声明](../../runtime/cuda/pyproject.toml)、[锁](../../runtime/cuda/uv.lock)、[环境恢复说明](../../docs/ENVIRONMENT.md)。每个job的manifest保留原命令、预算、快照哈希及runtime；所有命令仍须经lab，GPU复跑前需重新核对当前授权保护PID及资源。

| Job | 用途 | 证据 |
| --- | --- | --- |
| 20260905T231700Z-ppo-device-tests-b8b2ad77 | PPO设备回归 | [manifest](runs/20260905T231700Z-ppo-device-tests-b8b2ad77/manifest.json) · [log](runs/20260905T231700Z-ppo-device-tests-b8b2ad77/log.txt) |
| 20260905T231729Z-speed-real-fixture-05a2d327 | 真实固定夹具 | [manifest](runs/20260905T231729Z-speed-real-fixture-05a2d327/manifest.json) · [log](runs/20260905T231729Z-speed-real-fixture-05a2d327/log.txt) |
| 20260905T231818Z-speed-cpu-arms-b6555087 | CPU1/2/4臂 | [manifest](runs/20260905T231818Z-speed-cpu-arms-b6555087/manifest.json) · [log](runs/20260905T231818Z-speed-cpu-arms-b6555087/log.txt) |
| 20260905T231931Z-speed-cuda-build-cpu-deb6b1c5 | CUDA构建的CPU桥 | [manifest](runs/20260905T231931Z-speed-cuda-build-cpu-deb6b1c5/manifest.json) · [log](runs/20260905T231931Z-speed-cuda-build-cpu-deb6b1c5/log.txt) |
| 20260905T232008Z-speed-shared-gpu-907da07e | 共享GPU完整臂 | [manifest](runs/20260905T232008Z-speed-shared-gpu-907da07e/manifest.json) · [log](runs/20260905T232008Z-speed-shared-gpu-907da07e/log.txt) |
| 20260905T232058Z-speed-pool-one-6ed430c2 | 1环境进程 | [manifest](runs/20260905T232058Z-speed-pool-one-6ed430c2/manifest.json) · [log](runs/20260905T232058Z-speed-pool-one-6ed430c2/log.txt) |
| 20260905T232215Z-speed-pool-two-c00546da | 2环境进程 | [manifest](runs/20260905T232215Z-speed-pool-two-c00546da/manifest.json) · [log](runs/20260905T232215Z-speed-pool-two-c00546da/log.txt) |
| 20260905T232328Z-speed-pool-four-06428e04 | 4环境进程 | [manifest](runs/20260905T232328Z-speed-pool-four-06428e04/manifest.json) · [log](runs/20260905T232328Z-speed-pool-four-06428e04/log.txt) |
| 20260905T232428Z-speed-final-tests-c4781917 | 最终29项回归 | [manifest](runs/20260905T232428Z-speed-final-tests-c4781917/manifest.json) · [log](runs/20260905T232428Z-speed-final-tests-c4781917/log.txt) |

官方计时和内存API依据见[ENVIRONMENT](../../docs/ENVIRONMENT.md)。本报告的速度与资源结论来自本地运行，不是论文实验结果。
