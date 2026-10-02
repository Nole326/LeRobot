# LeRobot · 桌面操作 / Pick & Place

基于 LeRobot、LeIsaac 和 SO-101 的视觉桌面操作项目。从 PickOrange 入门，学习根据图像识别指定物体与目标区域，完成接近、抓取、抬升、搬运和稳定放置，再扩展到多物体泛化评测与真机部署。

A visual tabletop manipulation project built on LeRobot, LeIsaac and the SO-101. Starting with PickOrange, we learn to identify designated objects and goal regions from images, then approach, grasp, lift, transport and place objects stably before extending to multi-object generalization and real-robot deployment.

![LeRobot 桌面操作双语海报 / Bilingual pick-and-place poster](assets/lerobot-tabletop-pushing-bilingual.svg?v=20261003-pdf-aligned)

[查看项目海报 / View project poster](assets/lerobot-tabletop-pushing-bilingual.png)，[文字可无损放大版 / Scalable-text version](assets/lerobot-tabletop-pushing-bilingual.svg) · [4K PNG](assets/lerobot-tabletop-pushing-bilingual.png)

从环境接入到策略训练的实施步骤见[开发起步指南](docs/GETTING_STARTED.md)。任务、控制和评测遵循课程正式说明；课程指定的软件版本与接口优先于公开示例的默认设置。

See the [development guide](docs/GETTING_STARTED.md) for the path from environment integration to policy training. The formal course specification governs the task, control and evaluation; course-provided software versions and interfaces take precedence over public example defaults.

[项目目标 / Objectives](#项目目标--objectives) · [源码与工具 / Source-and-tools](#源码与工具--source-and-tools) · [快速开始 / Quick-start](#快速开始--quick-start) · [文档 / Documentation](#文档--documentation) · [参考与致谢 / References-and-acknowledgements](#参考与致谢--references-and-acknowledgements)

## 项目目标 / Objectives

**开发基准：`v0.1.1-baseline`。** 源码版本与使用方法见[基准说明](docs/BASELINE.md)，版本变化见[更新记录](CHANGELOG.md)。

**Development baseline: `v0.1.1-baseline`.** See the [baseline guide](docs/BASELINE.md) for pinned sources and setup instructions, and the [changelog](CHANGELOG.md) for version history.

**任务与控制。** 策略至少使用图像与机器人本体状态，不直接读取物体和目标的仿真真值位置。多物体任务输出桌面坐标系下的受限末端动作，经课程提供的 IK、限位、限速和安全接口执行。动作的具体维度、单位、范围与控制周期随正式接口确认。

**Task and control.** Policies use at least images and robot proprioception, without directly reading simulator ground-truth object or target positions. The multi-object task uses constrained end-effector actions in the table frame, executed through the course-provided IK, joint limits, rate limits and safety interfaces. Action dimensions, units, ranges and control timing follow the formal interface.

**算法与对照。** 通过遥操作采集示范，准备视觉规则、ACT 行为克隆和 SmolVLA 小型视觉语言动作策略基线，再研究强化学习与残差微调。算法不限于 PPO 或 SAC；BC 和预训练策略用于初始化或对照，不能替代 RL 训练。各方法统一测试条件，并用消融检验示范初始化、奖励设计或其他改进的作用。

**Algorithms and baselines.** Collect teleoperated demonstrations for visual rule-based, ACT behavior-cloning and SmolVLA compact vision-language-action baselines, then investigate reinforcement learning and residual fine-tuning. Algorithms are not restricted to PPO or SAC; BC and pretrained policies provide initialization or comparisons, not a substitute for RL training. Use shared test conditions and ablations to assess demonstration initialization, reward design or other improvements.

**仿真到真机。** 在仿真中完成标准任务和泛化评测，通过部署检查后，再在 SO-101 上运行学习策略。保持视觉处理和动作含义一致，分析外观、光照与延迟带来的差异；视觉追踪丢失或触发安全条件时暂停并等待人工处理。平台介绍见 [LeRobot SO-101 文档](https://huggingface.co/docs/lerobot/so101)。

**Simulation to reality.** Complete standard tasks and generalization evaluation in simulation, pass deployment checks, then run the learned policy on the SO-101. Align visual processing and action semantics, and analyze appearance, lighting and latency differences. Pause for human intervention if visual tracking is lost or a safety condition is triggered. See the [LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101) for the platform.

## 当前范围 / Current scope

本项目基于 LeRobot、LeIsaac 和 IsaacLab 开发，已提供固定版本源码、依赖清单和独立验证工具。当前仍是接口与训练准备阶段，没有一键可用的正式任务训练器；采集、BC、RL、评测和真机部署分别验收。

This project builds on LeRobot, LeIsaac and IsaacLab, with pinned sources, dependency inventories and standalone diagnostic tools. It is still in interface and training preparation, not a ready-to-run formal-task trainer. Collection, BC, RL, evaluation and hardware deployment have separate acceptance checks.

入门环境为 `LeIsaac-SO101-PickOrange-v0`，用于验证 reset/step、相机、夹爪、遥操作和数据链路。多物体任务使用课程提供的环境与评测接口；正式评测环境、控制器和评分规则不作修改。

The introductory environment, `LeIsaac-SO101-PickOrange-v0`, supports checks of reset/step, cameras, gripper control, teleoperation and the data pipeline. The multi-object task uses course-provided environments and evaluation interfaces; formal evaluation environments, controllers and scoring rules remain unchanged.

评测分别覆盖已见条件、组合外推条件和完全未见条件，记录成功率、完成时间、误抓率、漏抓率与放置误差。成功要求指定物体进入对应区域，并满足规定的位置误差与稳定时间；覆盖全部指定回合，不挑选表现好的场景。训练奖励可以设计，但不直接作为不同方法的评分依据。

Evaluate seen, compositional-generalization and entirely unseen conditions separately, reporting success rate, completion time, wrong-object grasp rate, missed-grasp rate and placement error. Success requires the designated object to reach its corresponding region within the specified position tolerance and stability duration. Complete every prescribed episode rather than selecting favorable scenes. Training rewards may vary, but do not directly determine comparative scores.

## 源码与工具 / Source and tools

**LeRobot。** 提供机器人数据、策略学习与硬件交互相关源码。本仓库固定版本包含 ACT、SmolVLA、SAC / HIL-SERL 等实现；包含实现不代表这些方法都已在本项目任务上验证。[上游项目](https://github.com/huggingface/lerobot)

**LeRobot.** Provides source code for robot datasets, policy learning and hardware interaction. The pinned snapshot includes implementations such as ACT, SmolVLA and SAC / HIL-SERL; inclusion does not mean that every method has been validated on this project's task. [Upstream project](https://github.com/huggingface/lerobot)

**LeIsaac 与 IsaacLab。** LeIsaac 提供基于 IsaacLab 的 SO-101 遥操作、数据采集与转换等工具；IsaacLab 提供机器人仿真与学习基础设施。两者均按固定版本完整导入，保留原始许可证。[LeIsaac](https://github.com/LightwheelAI/leisaac) · [IsaacLab](https://github.com/isaac-sim/IsaacLab)

**LeIsaac and IsaacLab.** LeIsaac provides SO-101 teleoperation, data collection and conversion tools built on IsaacLab, while IsaacLab provides robotics simulation and learning infrastructure. Both are included as complete pinned snapshots with their upstream licenses preserved. [LeIsaac](https://github.com/LightwheelAI/leisaac) · [IsaacLab](https://github.com/isaac-sim/IsaacLab)

**可核验的整合。** 独立工具覆盖源码完整性及环境、数据和交互接口的检查。版本和逐文件哈希记录在清单中；全部 45 项上游 LFS 测试素材已核验并作为普通 Git 文件保存，无需额外 LFS 下载或递归初始化子模块。

**Auditable integration.** Standalone tools support source-integrity checks and inspection of environment, data and interaction interfaces. Manifests record versions and per-file hashes. All 45 upstream LFS test fixtures have been verified and stored as regular Git files, with no additional LFS download or recursive submodule initialization required.

### 固定版本 / Pinned versions

| 项目 / Project | 版本 / Version | 固定源码 / Pinned source |
| --- | --- | --- |
| LeRobot | v0.4.2 | [58f70b6](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |
| LeIsaac | 0.4.0 | [24d3bcd](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| IsaacLab | v2.3.0 | [3c6e67b](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) |

### 仓库布局 / Repository layout

```text
.
├── assets/                   # Project artwork
├── third_party/
│   ├── lerobot/
│   ├── leisaac/
│   │   └── dependencies/IsaacLab/
│   └── isaaclab/
├── course/
│   ├── scripts/              # Standalone diagnostic utilities
│   └── constraints-*.txt     # Dependency constraint examples
├── configs/                  # Unconfirmed experiment-plan template
├── scripts/                  # Source audit, plan inspection, fixture recovery
├── tests/                    # First-party regression tests
├── manifests/                # Versions, hashes and asset inventories
├── docs/                     # Generic integration documentation
└── THIRD_PARTY_NOTICES.md
```

`third_party/` 保留上游源码；LeIsaac 内的 IsaacLab 依赖已展开为同版本完整副本。`course/scripts/` 仅包含通用验证代码，不包含私有课程材料或实验记录。依赖约束文件是参考，不是跨平台通用的一键安装方案。

`third_party/` preserves upstream source; LeIsaac's nested IsaacLab dependency is expanded into a complete copy of the same pinned version. `course/scripts/` contains generic diagnostic code, not private course materials or experiment records. The constraint files are references, not a universal one-command installation recipe.

## 快速开始 / Quick start

先获取仓库并执行不依赖仿真库或 GPU 的源码审计。以下命令只检查导入内容，不安装依赖、不启动训练。

Start by cloning the repository and running the source audit, which requires neither simulation libraries nor a GPU. These commands check the imported content only; they do not install dependencies or start training.

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

克隆参数仅作用于新仓库：兼容 Windows 的深层路径并保留原始换行，不修改全局 Git 配置。实际运行前，请按[依赖说明](docs/DEPENDENCIES.md)和[部署边界](docs/REPRODUCIBILITY.md)配置隔离环境，核实模拟器、Python、PyTorch、CUDA 与驱动兼容性。

The clone options apply only to the new repository: they accommodate deep Windows paths and preserve original line endings without changing global Git settings. Before execution, follow the [dependency inventory](docs/DEPENDENCIES.md) and [deployment boundaries](docs/REPRODUCIBILITY.md), use an isolated environment, and verify simulator, Python, PyTorch, CUDA and driver compatibility.

仓库检查只需 Git 和 Python 3.11+，覆盖上游内容哈希、公开工具语法、本地文档链接和海报尺寸；GitHub Actions 执行同一入口，不安装模拟器或训练依赖。完整固定版本的获取方式见[基准说明](docs/BASELINE.md)。

Repository checks require only Git and Python 3.11+, covering upstream content hashes, public utility syntax, local documentation links and poster dimensions. GitHub Actions uses the same entry point without installing simulator or training dependencies. See the [baseline guide](docs/BASELINE.md) to retrieve the fixed version.

Isaac Sim 运行时、独立场景资产、完整数据集与预训练权重需要通过官方渠道另行获取。所有 GPU 与硬件操作均应遵守所在机构的授权和安全要求；不要修改宿主驱动或干扰其他任务。

The Isaac Sim runtime, external scene assets, full datasets and pretrained weights must be obtained separately through official channels. GPU and hardware operations must comply with the relevant authorization and safety rules; do not modify host drivers or interfere with other workloads.

## 后续工作 / Next steps

无需安装仿真环境，即可检查实验计划。模板中未确认的字段保持空白，以下命令返回 `incomplete`（退出码 3）是预期行为；它不下载模型、不分配 GPU、不启动训练。

Inspect an experiment plan without installing the simulator. Unconfirmed fields remain blank; `incomplete` (exit code 3) is expected for the template. The command does not download models, allocate a GPU or start training.

```bash
python scripts/inspect_plan.py configs/experiment-plan.toml --print-config
```

先对齐观测、动作单位与采样时序，核验终止、截断及 `final_observation`；再采集少量示范检查同步、重放和成功标记，建立按完整回合划分的数据集。随后推进 ACT/SmolVLA、RL 基线与改进方法，并在统一场景、seed 和指标下完成对照、消融及失败分析。

First align observations, action units and sampling times, and verify termination, truncation and `final_observation`. Collect a small set of demonstrations to check synchronization, replay and success labels, then build episode-level dataset splits. Proceed to ACT/SmolVLA, RL baselines and improvements, comparing methods and analyzing ablations and failures under shared scenes, seeds and metrics.

保留数据来源与筛选说明、配置、模型、归一化、保存/加载流程、完整日志和成功/失败视频。仿真稳定并通过部署检查后再进行真机演示；未见物体、多目标或长程操作作为后续扩展，不替代基础任务。

Retain data provenance and filtering notes, configurations, models, normalization, save/load procedures, complete logs and success/failure videos. Attempt hardware demonstrations only after stable simulation and deployment checks. Unseen objects, multiple goals and longer-horizon operations are extensions, not replacements for the core task.

项目代码优先采用独立适配层；上游源码的修改记录在补丁和版本清单中，便于对照与复现。参与开发请参阅[贡献说明](CONTRIBUTING.md)。

Project-specific code uses separate adapters where possible. Changes to upstream sources are tracked through patches and version manifests for comparison and reproducibility. See the [contribution guide](CONTRIBUTING.md) to get involved.

## 文档 / Documentation

- [开发起步与阶段验收 / Development and acceptance guide](docs/GETTING_STARTED.md)
- [观测、动作、数据与恢复接口 / Observation, action, data and recovery contracts](docs/INTERFACE_CONTRACTS.md)
- [训练前验证与限制 / Pre-training validation and limits](docs/TRAINING_VALIDATION.md)
- [初始基准与版本范围 / Initial baseline and scope](docs/BASELINE.md)
- [修改与验证规范 / Contribution and validation guidelines](CONTRIBUTING.md)
- [版本记录 / Changelog](CHANGELOG.md)
- [依赖与外部资产 / Dependencies and external assets](docs/DEPENDENCIES.md)
- [部署布局与复现边界 / Deployment and reproducibility boundaries](docs/REPRODUCIBILITY.md)
- [源码导入清单 / Source import inventory](docs/IMPORT_MANIFEST.md)
- [海报与图像说明 / Poster and image notes](assets/README.md)
- [第三方许可与版权 / Third-party notices](THIRD_PARTY_NOTICES.md)

## 参考与致谢 / References and acknowledgements

感谢 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)、[Lightwheel LeIsaac](https://github.com/LightwheelAI/leisaac) 和 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 提供的开源基础。引用、分发和修改相关源码时，请遵守各自的许可证；本仓库的整合不改变上游版权归属。

We thank [Hugging Face LeRobot](https://github.com/huggingface/lerobot), [Lightwheel LeIsaac](https://github.com/LightwheelAI/leisaac) and [IsaacLab](https://github.com/isaac-sim/IsaacLab) for their open-source foundations. Follow each project's license when citing, distributing or modifying its code; integration here does not change upstream ownership.

海报为课程项目海报优化调整后的中英双语版。机器人图像来源为 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101)。

The poster is a refined and adapted Chinese-English version of the course project poster. The robot image is sourced from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101).
