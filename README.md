# LeRobot · 桌面推物 / Tabletop Pushing

面向强化学习课程项目的机器人学习工作区：从接近物块、建立接触到推动纠偏，探索桌面机械臂如何将物块推入目标区域并稳定停留。

A robot-learning workspace for a reinforcement learning course project: from approaching a block and making contact to correcting the pushing motion, we explore how a desktop robot arm can move a block into a target region and keep it there.

![LeRobot 桌面推物双语海报 / Bilingual tabletop pushing poster](assets/lerobot-tabletop-pushing-bilingual.svg?v=20261002-uniform-type)

[查看项目海报 / View project poster](https://raw.githubusercontent.com/Nole326/LeRobot/main/assets/lerobot-tabletop-pushing-bilingual.png)

[文字可无损放大版 / Scalable-text version](assets/lerobot-tabletop-pushing-bilingual.svg) · [4K PNG](assets/lerobot-tabletop-pushing-bilingual.png)

[项目目标 / Objectives](#项目目标--objectives) · [源码与工具 / Source-and-tools](#源码与工具--source-and-tools) · [快速开始 / Quick-start](#快速开始--quick-start) · [文档 / Documentation](#文档--documentation) · [参考与致谢 / References-and-acknowledgements](#参考与致谢--references-and-acknowledgements)

## 项目目标 / Objectives

**项目基准：`v0.1.0-baseline`。** 当前版本固定源码、公开工具与文档，作为后续开发起点；不是训练完成的模型基准。版本范围、复现命令与限制见[初始基准说明](docs/BASELINE.md)。

**Project baseline: `v0.1.0-baseline`.** This version fixes the source, public utilities and documentation as the starting point for future development; it is not a trained-model baseline. See the [baseline guide](docs/BASELINE.md) for scope, reproduction commands and limitations.

**任务与控制。** 以 SO-101 桌面单臂为参考平台，研究从随机起点将物块推入指定区域的闭环控制。海报中的策略动作是二维末端位移；重点关注接触位置变化后如何调整动作，而不只是移动到一个固定坐标。

**Task and control.** Using the SO-101 desktop arm as a reference platform, the project studies closed-loop pushing from randomized starting positions into a target region. The poster specifies two-dimensional end-effector displacements as policy actions, with an emphasis on adapting to changing contact points rather than simply reaching a fixed coordinate.

**算法与对照。** 计划探索 PPO / SAC 接触控制，与几何方法比较，并分析物块定位误差对策略表现的影响。具体环境、控制频率、奖励和成功判定需在接口确定后逐项验证；这里不将计划中的实验描述为已经实现的结果。

**Algorithms and baselines.** The planned study explores PPO / SAC for contact control, compares learned policies with geometric methods, and examines sensitivity to block-localization errors. The environment, control frequency, reward and success criteria must be verified once the interfaces are defined; planned experiments are not presented as completed results.

**仿真到真机。** 目标是结合关节反馈与相机物块定位，持续读取状态并修正动作，同时对齐仿真与真实机器人的观测和动作含义。SO-101 的硬件背景见 [LeRobot 官方文档](https://huggingface.co/docs/lerobot/so101)。

**Simulation to reality.** The goal is to combine joint feedback with camera-based block localization, repeatedly observe and correct motion, and align observation and action semantics between simulation and hardware. See the [official LeRobot documentation](https://huggingface.co/docs/lerobot/so101) for SO-101 hardware background.

## 当前范围 / Current scope

本仓库当前提供固定版本的完整源码快照、依赖清单和独立验证工具，为后续环境适配、策略训练与评估做准备。它不是 Hugging Face 官方 LeRobot 仓库，也不是课程正式环境或评分器；不宣称桌面推物的端到端训练、真机部署或正式评测已经完成。

The repository currently provides complete, pinned source snapshots, dependency inventories and standalone diagnostic tools to support subsequent environment adaptation, policy training and evaluation. It is neither the official Hugging Face LeRobot repository nor the course's official environment or grader. It does not claim completed end-to-end pushing training, real-robot deployment or formal evaluation.

已有 PickOrange 等上游示例用于接口与运行链路验证，不等同于海报中的桌面推物任务。保留这些源码是为了便于复用和对照，不表示沿用其任务定义、动作空间或评分规则。

Bundled upstream examples such as PickOrange serve as references for interface and execution-path checks; they are not the tabletop pushing task shown in the poster. Their inclusion supports reuse and comparison, not adoption of their task definitions, action spaces or scoring rules.

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
├── scripts/                  # Source audit and fixture recovery
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

首先对齐观测、动作单位与时间关系，核验终止、截断及 `final_observation` 处理；再接入正式任务环境，建立可复现的基线与评估协议。只有相应验证通过后，才推进训练与真机实验。

First align observations, action units and timing, and verify termination, truncation and `final_observation` handling. Then integrate the formal task environment and establish reproducible baselines and evaluation protocols. Training and hardware experiments follow only after their respective checks pass.

项目代码修改应优先采用独立适配层；确需修改上游时，记录版本、改动及验证依据，保留许可和来源。当前 README 不列未经验证的成功率或性能承诺。

Prefer separate adapters for project-specific changes. When upstream changes are necessary, record the version, patch and validation evidence while preserving licenses and attribution. This README does not report unverified success rates or performance claims.

## 文档 / Documentation

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

当前海报课程项目海报制作双语版。机器人图像参考来源为 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101)。

The current poster is a bilingual version of the course project poster. The robot image is sourced from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101).

公开内容限于源码、通用说明和获准发布的修复海报。原始课程材料、报告、服务器记录、实验输出、凭据及私有配置不随仓库发布。

Public content is limited to source code, generic documentation and the approved restored poster. Original course materials, reports, server records, experiment outputs, credentials and private configuration are not published with this repository.
