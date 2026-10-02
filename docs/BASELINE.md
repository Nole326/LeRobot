# 初始基准 / Initial baseline

基准版本：`v0.1.0-baseline`。这是项目的源码整合起点，不是训练完成的策略、正式课程环境或性能基准。后续开发从此版本建立对照；不要移动或覆盖这个标签。

Baseline version: `v0.1.0-baseline`. This is the source-integration starting point, not a trained policy, formal course environment or performance benchmark. Use it as the reference for subsequent development; do not move or overwrite this tag.

## 获取固定版本 / Retrieve the pinned baseline

```bash
git clone -c core.longpaths=true -c core.autocrlf=false --branch v0.1.0-baseline https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
git rev-parse HEAD
```

标签克隆处于 detached HEAD，适合检查或复现。开始开发时使用 `git switch -c feature/your-change` 创建分支。上述检查只需 Git 与 Python 3.11+ 标准库；不安装依赖、不联网下载资产、不使用 GPU。

A tag checkout has a detached HEAD, suitable for inspection or reproduction. Create a development branch with `git switch -c feature/your-change`. These checks require only Git and the Python 3.11+ standard library; they do not install dependencies, download assets or use a GPU.

## 固定内容 / Frozen contents

- LeRobot v0.4.2、LeIsaac 0.4.0、IsaacLab v2.3.0 的固定源码及原始许可。

  Pinned LeRobot v0.4.2, LeIsaac 0.4.0 and IsaacLab v2.3.0 source with original licenses.

- 4,361 个上游文件、45 个已物化的 LFS 测试素材，以及 18 个公开工具/依赖约束文件。准确版本和哈希见 [manifests](../manifests/)。

  4,361 upstream files, 45 materialized LFS test fixtures and 18 public utility/constraint files. Exact versions and hashes are recorded in [manifests](../manifests/).

- 双语项目说明、获准发布的海报，以及无 GPU 的仓库自动检查。完整性检查不等于模拟器或训练验收。

  Bilingual project documentation, the approved poster and GPU-free repository checks. Content integrity is not simulator or training acceptance.

## 尚未完成 / Not yet established

桌面推物正式环境与评分接口、端到端 PPO/SAC 训练、真实机械臂部署、完整依赖锁定及任意时刻的训练现场恢复，不属于本版完成承诺。现有诊断脚本中的 PickOrange 示例不能代替正式推物任务。

The formal tabletop-pushing environment and scoring interface, end-to-end PPO/SAC training, real-robot deployment, a complete dependency lock and arbitrary-point training-state restoration are outside this version's completion claims. PickOrange examples in diagnostic scripts do not replace the formal pushing task.

## 后续改动 / Future changes

每次实验记录本仓库 commit、上游版本、实际依赖、配置和输入数据哈希。修改上游文件时需保留原版本和补丁依据，并同步更新可审核的哈希清单；不要只改清单来掩盖不明差异。课程与服务器记录仍留本地。

Record the repository commit, upstream versions, actual dependencies, configuration and input-data hashes for each experiment. Upstream changes need preserved provenance, patch rationale and updated auditable manifests; never change hashes merely to hide unexplained differences. Course and server records remain private.

本基准不会给整仓套用一个新许可证。第三方源码继续适用各自许可；独立整合代码的新许可证由仓库所有者另行决定，详见[版权说明](../THIRD_PARTY_NOTICES.md)。

This baseline does not impose a new repository-wide license. Third-party code retains its own terms; a license for original integration code remains an owner decision. See the [third-party notices](../THIRD_PARTY_NOTICES.md).
