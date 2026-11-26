<p align="center">
  <a href="assets/lerobot-tabletop-pushing-bilingual.svg">
    <img src="assets/lerobot-tabletop-pushing-bilingual.svg" width="960" alt="LeRobot 桌面操作双语海报">
  </a>
</p>

<h1 align="center">LeRobot：强化学习项目</h1>
<h3 align="center">视觉桌面操作：从仿真学习到 SO-101 真机部署</h3>

<p align="center">
  基于 LeRobot、LeIsaac 与 SO-101，探索视觉抓放、模仿学习和强化学习。
</p>

<p align="center">
  <a href="README.en.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#仓库结构">仓库结构</a> ·
  <a href="#开发路线">开发路线</a> ·
  <a href="#文档">文档</a> ·
  <a href="CONTRIBUTING.md">参与开发</a>
</p>

---

当前阶段推进 LiftCube 和 PickOrange 两条任务路线，分别建立策略、数据与评测流程。课程正式多物体环境与真机迁移留待后续。

## 当前范围

### 仿真任务

- **LiftCube 仿真任务**：抓取桌面方块并抬起。仓库提供演示读取、行为克隆（BC）和参考 PPO 实现，环境配置与运行命令见 [LiftCube 运行指南](course/liftcube/README.md)。
- **PickOrange 仿真任务**：将三颗橙子放入盘中，再让机械臂归位。仿真环境采用 [LightwheelAI 的 LeIsaac](https://github.com/LightwheelAI/leisaac) 0.4.0。仓库提供 [ACT 行为克隆（qpos 与 EE）](course/pickorange/act/README.md)、[ACT 残差 SAC](course/pickorange/sac/README.md) 和 [GR00T 残差 SAC](course/pickorange/groot_sac/README.md) 的训练与评测代码，运行命令见 [PickOrange 运行指南](course/pickorange/README.md)。

### VLA 接口与控制脚本

- **VLA 推理接口**：LiftCube 使用前视相机，PickOrange 使用前视和腕部相机。各模型的输入要求和推理入口见 [VLA 接口指南](docs/VLA_INTERFACES.md)。
- **控制脚本**：两项任务均提供视觉控制脚本和特权state控制脚本，详见[控制脚本指南](docs/SCRIPTED_POLICIES.md)。LiftCube 的两种脚本共用关节控制接口，标定方法和固定场景成绩见 [LiftCube 控制脚本指南](docs/SCRIPTED_LIFTCUBE.md)。PickOrange 的[视觉控制脚本](course/pickorange/visual/README.md)用于抓放，[特权state控制脚本](course/pickorange/expert/README.md)支持专家演示采集和动作轨迹独立回放检验。

模型权重和演示数据见[资源清单](docs/RESOURCES.md)，各任务的运行命令见[训练与评测指南](docs/TRAIN_AND_EVALUATE.md)，环境版本和模型适配情况见[任务对照表](docs/TASKS.md)。

## 快速开始

获取代码并运行仓库检查（需要 Git 和 Python 3.11+）：

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

该检查不启动仿真或训练。运行 LiftCube 请按[镜像与环境说明](course/liftcube/README.md#运行环境--runtime)获取配套镜像；PickOrange使用[独立部署入口](docs/PICK_ORANGE.md)，不要混用两套依赖或启动脚本。

运行视觉控制脚本需要匹配的机器人与相机标定。请从[控制脚本指南](docs/SCRIPTED_LIFTCUBE.md)的离线标定检查开始。BC/PPO训练另按[训练指南](course/liftcube/TRAINING.md)准备数据和接口。

项目版本见[基准说明](docs/BASELINE.md)与 [CHANGELOG](CHANGELOG.md)。参与开发请从最新 `main` 创建功能分支，流程见[贡献指南](CONTRIBUTING.md)。

## 仓库结构

LiftCube集成代码位于 `course/liftcube/src/starter/`；PickOrange运行入口位于`course/pickorange/`，诊断工具和上游任务源码分别保留：

```text
.
├── course/liftcube/
│   ├── src/starter/
│   │   ├── interfaces/        # 观测、动作、策略与模型契约
│   │   ├── envs/              # 仿真接入、控制与终态处理
│   │   ├── data/              # 演示读取、转换、校验与划分
│   │   ├── algorithms/        # BC、RL与参考PPO
│   │   ├── training/          # 配置驱动的训练入口
│   │   ├── evaluation/        # 共用评测与策略加载
│   │   ├── rewards/           # 训练奖励
│   │   └── logging/           # 指标记录
│   ├── configs/              # 当前任务配置
│   └── patches/              # 相对课程原包的可逆修改
├── course/scripts/           # PickOrange环境、重放、ACT桥接等诊断
├── course/pickorange/        # ACT、残差 SAC、VLA 服务、仿真评测与演示采集
├── configs/                  # 公共实验计划模板
├── scripts/                  # 仓库、来源与计划检查
├── tests/                    # 项目回归测试
├── third_party/              # 固定版本的上游源码
├── manifests/                # 源码、版本清单
├── docs/                     # 架构、接口、运行与复现说明
└── assets/                   # 海报与项目素材
```

模型接入与各层职责见[架构说明](docs/ARCHITECTURE.md)。镜像、数据与权重单独获取，来源和版本见[依赖说明](docs/DEPENDENCIES.md)。

## 开发路线

1. 分别冻结LiftCube与PickOrange的环境、输入、动作和成功判据，补齐各自运行入口。
2. 保留LiftCube 控制脚本、演示准入及ACT/小规模VLA路线；当前重点推进PickOrange的VLA基线、可归因RL对照与其他模型能力验收。
3. 根据已验收的数据和模型推进微调、RL及必要消融，记录完整回合和失败原因。
4. 正式多物体环境发布后，再恢复跨物体迁移、泛化和真机对齐。

BC与预训练VLA作为初始化或对照，RL为独立训练阶段。评测记录成功率、完成时间、误抓/漏抓和放置误差，具体成功判据及时间上限随任务环境确定。

VLA 强化学习实现参考 [RLinf](docs/RLINF.md)，目前尚未接入本项目运行环境。

### 上游版本

以下是历史源码快照；当前 LiftCube 使用课程配套的 Isaac Sim 4.5.0 / Isaac Lab v2.1.1 / LeIsaac v0.1.2，二者分开管理。

| 项目 | 历史源码版本 | 固定提交 |
| --- | --- | --- |
| LeRobot | v0.4.2 | [58f70b6](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |
| LeIsaac | 0.4.0 | [24d3bcd](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| IsaacLab | v2.3.0 | [3c6e67b](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) |

源码来源与完整性信息见[导入清单](docs/IMPORT_MANIFEST.md)。

## 文档

- [开发起步与阶段验收](docs/GETTING_STARTED.md)
- [LiftCube与PickOrange：资源与适配状态表](docs/TASKS.md)
- [PickOrange模型检验、RL配对与显存参考](docs/PICKORANGE_VALIDATION.md)
- [模型权重与专家演示获取](docs/RESOURCES.md)
- [按任务进行控制脚本采集、训练和评测](docs/TRAIN_AND_EVALUATE.md)
- [SwanLab训练与评测曲线](docs/SWANLAB.md)：仅数值与脱敏配置，默认离线，在线使用私有项目
- [LiftCube 源码、镜像与运行方法](course/liftcube/README.md)
- [PickOrange 源码、独立环境与诊断入口](docs/PICK_ORANGE.md)
- [视觉控制脚本、标定与后续适配](docs/SCRIPTED_LIFTCUBE.md)
- [受限末端开发接口与验证限制](docs/PLANAR_EE_DEVELOPMENT.md)
- [架构、扩展位置与团队协作](docs/ARCHITECTURE.md)
- [RLinf 对本项目的帮助与接入边界](docs/RLINF.md)
- [观测、动作、数据与恢复契约](docs/INTERFACE_CONTRACTS.md)
- [训练前验证与限制](docs/TRAINING_VALIDATION.md)
- [依赖与外部资产](docs/DEPENDENCIES.md)
- [部署与复现边界](docs/REPRODUCIBILITY.md)
- [源码导入清单](docs/IMPORT_MANIFEST.md)
- [修改与版本管理](CONTRIBUTING.md)
- [智能体开发规范](AGENTS.md)
- [海报与图像说明](assets/README.md)
- [第三方许可与版权](THIRD_PARTY_NOTICES.md)

## 参考与致谢

感谢 [LeRobot](https://github.com/huggingface/lerobot)、[LeIsaac](https://github.com/LightwheelAI/leisaac) 和 [IsaacLab](https://github.com/isaac-sim/IsaacLab)。工程分层参考 [rl-garden](https://github.com/Nole326/rl-garden)，VLA 强化学习方案参考 [RLinf](https://github.com/RLinf/RLinf)。相关许可见[第三方说明](THIRD_PARTY_NOTICES.md)。

海报为课程项目海报优化调整后的中英双语版。机器人图像来源为 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101)。
