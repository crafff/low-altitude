# 项目Python环境

Python版本由 [.python-version](../.python-version) 固定，依赖声明在 [pyproject.toml](../pyproject.toml)，精确解析结果在 [uv.lock](../uv.lock)。uv.lock由uv生成并随Git保存；只在调整依赖时更新。应用库是否已接通、CUDA是否可用，以 [当前任务](../paper/NOW.md) 为准。

## 建立或恢复环境

在项目根目录执行，目录变量保留在当前shell中供后续uv命令使用：

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/environments/python"
uv python install --no-bin
uv sync --locked --managed-python --no-build
```

`.venv/`保存项目虚拟环境，`environments/python/`保存本项目的uv托管解释器，`.cache/uv/`保存下载缓存；这些目录均不提交Git。`--no-bin`避免向用户全局bin安装可执行入口。系统Python可以继续启动仅依赖标准库的lab监督器，科研负载使用`.venv/bin/python`。

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
