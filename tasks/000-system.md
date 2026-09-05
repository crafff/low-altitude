# 000 — 从事故恢复到简洁可用系统

状态：`done`（2026-09-05补充精简复核完成）。负责人：主线程。授权：用户要求检查、精简/修正系统，验证后commit并push，准备继续研究。没有启动训练或继承旧时限goal。

## 已完成的改动

- 原PDF重新接收并核对原SHA，重命名归位。
- 重建前的六个活动目录及.codex/.agents完整tar保存在legacy/pre-rebuild-20260904；旧测试、lab入口、baseline半成品和旧角色配置退出活动路径，未销毁。
- 三角色Astra/xhigh、明确读写范围和五字段交接；旧技能审批链禁用。
- PLAN/NOW/LESSONS/SOURCES/DECISIONS + 单任务记录；不恢复Claim/Experiment/Evidence。
- lab运行器使用固定只读源码快照、独立产物/临时目录、清环境/无网络/PID隔离、单运行锁及有界执行。全部记录自动生成，无手动绑定。

## 本轮两个子agent的交接

1. `paper_reproduction_audit`：只读原PDF和对应旧笔记；输出论文参数/页码、未披露值与现存计划差异；不得写文件或运行代码。结论已合并到paper/REPRODUCTION.md。
2. `safety_architecture_audit`：只读旧半成品；指出危险清理、失效路径、非PPO实现。复用该线程只读复核新lab，确认未发现子进程写回真实项目的通路；建议失败原因、隐私标记、runtime和锁三项小修正，已实施并复测。

两者均通过原生spawn显式请求gpt-6-astra/xhigh、fresh context；实际会话的全部turn_context已核对一致，不只是相信请求参数：

- paper_reproduction_audit：`01a06f3a-8d61-7100-b66c-e46d9c5a98c2`，实际模型gpt-6-astra，effort xhigh。
- safety_architecture_audit：`01a06f3a-d0c0-7233-ab73-88805a338b88`，首次与复核轮次均为gpt-6-astra/xhigh。
- 来源为 `/home/magic/.codex/sessions/2026/09/04/` 中对应session id的原生日志；本地时间文件名与UTC事件时间相差5小时。无需把整段聊天复制成另一个事实库。

主线程持有全部写入职责，未运行旧代码。新配置及三个角色TOML已用Python3.11标准库tomllib解析验证；七个旧本地技能均配置为disabled。未来角色加载仍须以实际线程元数据为准。

## 验证记录

- PDF：d60671217c061892017a6c3a5ec755a78b3f132f46b10d48ff7807e8dafc97a4，30页。
- 首轮隔离构造：[system-bootstrap](../runs/20260905T015403Z-system-bootstrap-df89da3d/log.txt)，13项通过，无跳过；不是BlueSky或GPU实验。
- 补充复核：[system-review](../runs/20260905T020214Z-system-review-66dd06f9/log.txt)，15项中2项失败。原因是嵌套运行试图将单文件硬上限从外层256MiB提高到默认2GiB，导致启动失败，并影响锁测试；不是源码/数据被写坏。已按继承的硬上限取最小值，并记录实际生效上限。
- 最终回归：[system-final](../runs/20260905T020602Z-system-final-312477ed/log.txt)，16项全部通过、无跳过，约5.9秒；包括实际私有标记、环境清除、只读runtime、单任务锁与嵌套上限回归。
- 活动Markdown相对链接检查无缺失；Git diff --check通过；doctor、index和search入口当时实际调用成功。当时尚未提交Git；index已在下述复核中移除。
- 源码快照与原始测试日志留在各自run内。失败记录不删除、不覆盖；构造夹具由其独立临时目录在外层沙箱内正常回收。

## 完成边界

交付可用入口、记忆与交接、模型配置、归档和验证过的运行器即可。新PPO/BlueSky环境、有效baseline、延迟曲线及独立物理备份是后续工作，不混入系统完成声明。

## 2026-09-05 系统精简与提交复核

- 范围：活动文档、运行器及必要回归；核对上轮重建的提交范围，提交并推送当前研究分支。不运行BlueSky或训练。
- 独立只读交接 `audit_launcher`：问题为运行器具体缺陷；范围为tools/lab.py、tests/test_lab.py及相关活动契约；输出按影响排序的最小修正；一轮约8分钟，只读、不执行负载、不嵌套委派。主线程承担全部修改和验证。
- 原生线程 `01a06f59-8be3-7e80-9723-234efa9ca6b8` 显式Astra/xhigh、fresh context；首审和修改后复核的两次turn_context均核对为gpt-6-astra/xhigh。子agent只读、没有运行测试或写文件。
- Git重建检查：当前924个已跟踪删除路径全部在legacy/recovery-20260904/tracked/有与原HEAD逐字节相同的副本；原HEAD为 `6775efd`。没有删除归档原件。
- 文档精简：日常只更新NOW与当前任务，其他文件按事件更新；入口移除易过期状态；去掉未被消费的index缓存入口，保留即时search，补充doctor断链/任务入口检查。
- 修复：以实际项目及祖先/私人/隐藏路径校验runtime；显式输入禁止隐藏配置；源码快照完整保留普通资产和可执行位，超限、符号链接、遍历错误明确失败。搜索仍限小文本，包含README/AGENTS。
- 首轮 [system-simplify](../runs/20260905T022321Z-system-simplify-e4bc6284/log.txt)：21项通过、无跳过，6.293秒。复核补充隐藏runtime和源码遍历错误回归后，最终 [system-simplify-final](../runs/20260905T022553Z-system-simplify-final-87328e2c/log.txt)：22项通过、无跳过。
- 两次均经外层launcher，每次90秒/256MiB上限、串行执行。最终命令：`python3 -B tools/lab.py run --label system-simplify-final --stage system --seconds 90 --disk-mib 256 -- /usr/bin/python3 -B -m unittest discover -s tests -v`。运行器、测试完整快照及日志在各run中保留。
- `doctor`无活动断链或缺失本地材料，search实际调用成功。官方Codex配置参考重新读取，现有配置键有对应依据；四份角色/主配置本轮没有修改。当前系统Python实测3.10.12，尝试python3.11及tomllib/pip解析入口均不可用，因此不把上轮3.11解析结果说成本轮重验；子agent实际加载/模型元数据已核对。
- 保留的限制：runtime实时只读挂载而非依赖快照；SIGTERM/SIGKILL可能留下未核验running记录；磁盘/RAM不是硬总配额；doctor只处理普通内联Markdown路径。这些边界已写入运行/系统说明，不扩展为新审批流程。
- Git发布范围为已核对的新系统与旧活动路径退出；本地PDF、权重、run和恢复归档不加入Git。提交身份/远端同步状态以当前研究分支的Git记录为准，避免再维护一份可过期状态表。
- GitHub目标核验（2026-09-05）：现有origin为 `https://github.com/crafff/low-altitude.git`。首次push被自动审批以缺少目标归属证据为由拒绝，未执行推送；随后只读 `gh api user --jq .login` 返回 `crafff`，`gh repo view crafff/low-altitude --json nameWithOwner,url,viewerPermission,isPrivate` 返回同名私有仓库及ADMIN权限。以该证据重试用户本轮已授权的当前研究分支推送，不强推、不改变其他分支。
- 普通推送随后被GitHub的GH001限制拒收：旧历史中 `experiments/E000-action-induced-conflict-definition/development/DEV-E000-015/branches.json` 为221.32MB，`experiments/E002-fixed-cascade-replication/runs/RUN-E002-001/branches.json` 为221.54MB，超过其100MB单文件限制；本次没有将旧文件重新加入当前树。系统实现提交为 `0cebfa8`，目标核验记录为 `3c4fe7e`。
- 发布处理：完整旧历史及上述提交继续保留在本地 `research/trc-reboot-20260904`。使用当前已验证文件树建立无历史父提交的新分支 `research/trc-baseline-system-20260905`，向同一个已核验私有仓库做普通推送。远端发布分支只保存当前系统；旧Git历史与大型本地产物仍需独立备份。没有改写原分支历史、删除对象或强推。

下一步：任务001的干净BlueSky环境与无控制小样。先≤15分钟DEV，不把本轮系统检查当作训练授权或科学结果。
