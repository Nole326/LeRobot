# 版本与基准 / Versions and baselines

本仓库区分历史源码快照与任务策略基准：

- `v0.1.0-baseline`、`v0.1.1-baseline`固定初始上游源码、工具和文档，不包含后续LiftCube规则脚本或训练权重。两个标签均不移动或覆盖。
- LiftCube视觉规则开发基准从提交[48824f2](https://github.com/Nole326/LeRobot/commit/48824f2f7791b63ec88502c6a44dee1f8d7ada5f)开始；配置、标定来源和各组结果见[规则指南](SCRIPTED_LIFTCUBE.md)。它不是课程正式多物体或真机验收版本。
- 日常开发从最新`main`创建分支；复现实验时固定实际使用的commit及外部文件身份，不因沿用“基准”名称而混用不同版本。

The initial `v0.1.0-baseline` and `v0.1.1-baseline` tags pin source/tools/documentation and remain immutable. The LiftCube rule-policy development reference starts at commit 48824f2; see its [guide](SCRIPTED_LIFTCUBE.md) for calibration and results. It is not an accepted formal multi-object or hardware version. Branch new work from current main, and pin the actual commit and external inputs for reproduction.

## 获取固定版本 / Retrieve the pinned baseline

```bash
git clone -c core.longpaths=true -c core.autocrlf=false --branch v0.1.1-baseline https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
git rev-parse HEAD
```

上述命令用于历史快照，标签克隆处于detached HEAD，不含后续LiftCube开发。需要规则基准时，正常克隆后执行`git switch --detach 48824f2f7791b63ec88502c6a44dee1f8d7ada5f`；该固定提交同样不附标定、镜像、演示或权重。检查只需Git与Python 3.11+标准库，不使用GPU。

The commands above retrieve the historical source snapshot. For the rule reference, use an ordinary clone and check out commit 48824f2 with `git switch --detach`; obtain calibrated external inputs separately. The source checks require only Git and Python 3.11+, not a GPU. Start ongoing development from current main rather than an old tag.

## v0.1.1固定内容 / Contents frozen in v0.1.1

- LeRobot v0.4.2、LeIsaac 0.4.0、IsaacLab v2.3.0 的固定源码及原始许可。

  Pinned LeRobot v0.4.2, LeIsaac 0.4.0 and IsaacLab v2.3.0 source with original licenses.

- 4,361 个上游文件、45 个已物化的 LFS 测试素材，以及 18 个公开工具/依赖约束文件。准确版本和哈希见 [manifests](../manifests/)。

  4,361 upstream files, 45 materialized LFS test fixtures and 18 public utility/constraint files. Exact versions and hashes are recorded in [manifests](../manifests/).

- 双语项目说明、包含官方 SO-101 照片的矢量文字海报与 4K PNG，以及无 GPU 的仓库自动检查和字体排版回归测试。

  Bilingual documentation, a vector-text poster and 4K PNG featuring the official SO-101 photograph, plus GPU-free repository checks and typography regression tests.

`v0.1.1-baseline`沿用`v0.1.0-baseline`的上游源码及版本，仅完善文档、海报和检查项。后续LiftCube及接口改动见[CHANGELOG](../CHANGELOG.md)。仓库检查不替代模拟器或训练验证。

This version retains the upstream sources and versions from `v0.1.0-baseline`, with improvements to documentation, artwork and checks. See the [changelog](../CHANGELOG.md) for a summary. Repository checks do not replace simulation or training validation.

## 尚未完成 / Not yet established

正式任务环境与评分接口、端到端 RL 训练、真实机械臂部署、完整依赖锁定及任意时刻的训练现场恢复，不属于本版完成承诺。PickOrange 是入门验证环境，不能代替完整多物体任务验收；海报不是正式接口规范。

The formal task environment and scoring interface, end-to-end RL training, real-robot deployment, a complete dependency lock and arbitrary-point training-state restoration are outside this version's completion claims. PickOrange is an introductory environment, not acceptance of the full multi-object task; the poster is not an interface specification.

## 后续改动 / Future changes

每次实验记录本仓库 commit、上游版本、实际依赖、配置和输入数据哈希。修改上游文件时需保留原版本和补丁依据，并同步更新可审核的哈希清单；不要只改清单来掩盖不明差异。课程与服务器记录仍留本地。

Record the repository commit, upstream versions, actual dependencies, configuration and input-data hashes for each experiment. Upstream changes need preserved provenance, patch rationale and updated auditable manifests; never change hashes merely to hide unexplained differences. Course and server records remain private.

本基准不会给整仓套用一个新许可证。第三方源码继续适用各自许可；独立整合代码的新许可证由仓库所有者另行决定，详见[版权说明](../THIRD_PARTY_NOTICES.md)。

This baseline does not impose a new repository-wide license. Third-party code retains its own terms; a license for original integration code remains an owner decision. See the [third-party notices](../THIRD_PARTY_NOTICES.md).
