# 项目Python环境

Python版本由 [.python-version](../.python-version) 固定，依赖声明在 [pyproject.toml](../pyproject.toml)，精确解析结果在 [uv.lock](../uv.lock)。uv.lock由uv生成并随Git保存；只在调整依赖时更新。应用库是否已接通、CUDA是否可用，以 [当前任务](../paper/NOW.md) 为准。

## 建立或恢复环境

在项目根目录执行，目录变量保留在当前shell中供后续uv命令使用：

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/environments/python"
uv python install --no-bin
uv sync --locked --managed-python \
  --no-build-package bluesky-simulator --no-build-package numpy \
  --no-build-package scipy --no-build-package pandas \
  --no-build-package pyzmq --no-build-package matplotlib --no-build-package torch
```

`.venv/`保存项目虚拟环境，`environments/python/`保存本项目的uv托管解释器，`.cache/uv/`保存下载缓存；这些目录均不提交Git。`--no-bin`避免向用户全局bin安装可执行入口。系统Python可以继续启动仅依赖标准库的lab监督器，科研负载使用`.venv/bin/python`。

当前锁文件固定BlueSky 1.1.1。其依赖的`zmq==0.0.0`只有源码分发，完全禁止构建的`--no-build`会在新环境失败。已检查官方966字节源码包：只是依赖`pyzmq`的setuptools元包，无包代码；本次仅它发生构建，其余安装使用wheel。上述命令禁止主要科学库源码构建；若将来锁文件变化出现新的构建需求，先核对新来源。源码地址、哈希和实际安装记录见[任务001](../tasks/001-baseline.md)。

依赖确定后修改pyproject并执行`uv lock`，再`uv sync --locked`；核对锁文件变更后随代码提交。先用轮子包安装；需要源码构建时单独调查依赖，不直接运行旧项目安装脚本。不要在lab运行期间修改解释器或依赖目录。

## 在实际隔离环境运行

uv管理解释器与依赖，lab管理负载隔离、预算和记录。先在宿主准备依赖，然后将两个专用目录只读挂入；沙箱中不运行需要同步/下载依赖的`uv run`。

```bash
python3 -B tools/lab.py run --label system-tests --stage system \
  --seconds 90 --disk-mib 256 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  -- "$PWD/.venv/bin/python" -B -m unittest discover -s tests -v
```

两个目录都需要：venv中的Python可能链接到托管解释器，标准库也在后者中。launcher自动将三个版本/依赖文件收入只读源码快照，运行配置仍须记录BlueSky、PyTorch、CUDA及重建选择的实际版本；锁文件不证明GPU或模拟器已经可用。

依据：[uv项目管理](https://docs.astral.sh/uv/guides/projects/)与[Python版本管理](https://docs.astral.sh/uv/concepts/python-versions/)。环境安装/锁定属于准备工作；项目测试、导入模拟器和训练仍通过lab执行。

## GPU执行路径（2026-09-05实测）

普通Codex受限命令未暴露/dev/nvidia*，即使/proc能读到驱动版本也不能启动GPU负载。经执行工具的主机权限审查后启动原有lab --gpu，可在Bubblewrap内访问RTX4090；无需改驱动、设备权限或取消隔离。需要主机执行权限时由执行工具审批，不在命令里加入sudo或替代的裸运行路径。

下列命令已在该获准环境验证，编译产物仅写lab的/tmp，全部日志和源码快照由lab保存。sm_89针对本机RTX4090；其他GPU需先核对架构。固定16KiB输出和单次kernel仅检查功能，不是吞吐测试。

```bash
python3 -B tools/lab.py run --label cuda-kernel-probe --stage dev --gpu \
  --seconds 45 --disk-mib 32 --memory-mib 8192 -- /bin/sh -c \
  '/usr/local/cuda/bin/nvcc -O2 -arch=sm_89 tools/cuda_probe.cu -o /tmp/cuda-probe && /tmp/cuda-probe'
```

[已保存结果](../runs/20260905T050609Z-cuda-kernel-probe-20d365fd/log.txt)：Toolkit/Runtime12.8、驱动580.173.02、compute capability8.9，4096个整数全部正确。驱动API报告13000不代表本次使用CUDA13 Runtime。查询时空闲显存约3.4GiB，不能据显卡24GiB标称容量规划可用显存。这项原生CUDA探针不能推出训练链路就绪。当前10h执行块按用户要求不干预其他GPU实验，已改为下述CPU专用PyTorch；不在本块重复GPU计算。

## 共享主机上的CPU训练（2026-09-05）

当前固定PyTorch2.9.1+cpu，175.9MiB官方CPU wheel；pyproject的pytorch-cpu索引设置explicit=true，只有torch绑定到该索引，其余依赖来自PyPI。选择固定CPU构建是为了不向其他人的GPU训练提交计算，不是最新版本声明。[PyTorch版本安装说明](https://pytorch.org/get-started/previous-versions/) · [uv的PyTorch索引规则](https://docs.astral.sh/uv/guides/integration/pytorch/)。锁文件不能替代运行验证；核心隔离回归已实际执行共享attention网络的CPU梯度与参数更新，见任务001。

恢复时沿用上方sync命令；可额外设置UV_CONCURRENT_DOWNLOADS/INSTALLS/BUILDS=1以减少安装并发。不要在lab负载运行时同步环境。CPU负载当前使用nice15、idle I/O、单逻辑CPU affinity14及torch/BLAS单线程；该CPU号是本机选择，不是跨主机假设：

```bash
nice -n 15 ionice -c 3 taskset -c 14 python3 -B tools/lab.py run \
  --label paper-train-tests --stage dev --config configs/paper_train_dev.json \
  --seconds 120 --disk-mib 128 --memory-mib 4096 \
  --runtime "$PWD/.venv" --runtime "$PWD/environments/python" \
  -- "$PWD/.venv/bin/python" -B -m unittest discover -s tests -p test_paper_train.py -v
```

单launcher串行、无--gpu、初期分钟级预算。内存参数是每进程地址空间上限，磁盘阈值是软监控，均不能当硬聚合资源配额。共享CPU/磁盘仍可能有少量影响，资源紧张时减少或暂停我们的任务，不调整其他人的进程。
