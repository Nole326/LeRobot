<p align="center">
  <a href="assets/lerobot-tabletop-pushing-bilingual.svg">
    <img src="assets/lerobot-tabletop-pushing-bilingual.svg?v=20261003-pdf-aligned" width="960" alt="LeRobot 桌面操作双语海报">
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

从 PickOrange 熟悉数据与控制链路，以 LiftCube 建立训练和评测流程，再对齐课程多物体任务与真机部署。

## 当前范围

- **LiftCube**：已加入课程起点、配套环境说明、BC/参考 PPO 及评测入口。它是抓取并抬升方块的开发环境，不是完整多物体抓放任务。
- **统一接口**：当前 LiftCube 使用 front RGB、6 维关节位置与绝对电机目标。速度、末端位置的兼容字段是补零，不是实测反馈；完整输入和频率见[任务说明](course/liftcube/README.md#接口--interface)。
- **模型路线**：ACT、SmolVLA、OpenVLA-OFT、π₀.₅、GR00T 是待接入或验证的方向，不代表已经提供可用的课程任务权重。
- **正式训练前**：演示动作映射、模型适配及完整存档恢复仍需验收。参考训练器可运行不等于训练、泛化或真机性能已经验证。

课程正式任务、控制器、评分规则优先于海报和上游示例。策略使用图像与许可本体状态，不读取物体/目标真值；训练奖励可按规则使用特权信息。最终任务接口与当前 LiftCube 的关节控制不能混为一谈。

## 快速开始

只检查仓库内容，不安装模拟器、不分配 GPU、不启动训练：

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

检查需要 Git 和 Python 3.11+。克隆参数只作用于新仓库，保留上游换行并兼容 Windows 深层路径。自动检查覆盖源码完整性、语法、文档链接、测试和海报尺寸，不代表真实仿真或训练通过。

开发环境从 [LiftCube 使用说明](course/liftcube/README.md) 开始：其中提供[配套镜像下载渠道、SHA256 与导入步骤](course/liftcube/README.md#运行环境--runtime)。运行代码应在符合所在平台规范的隔离环境中进行，保留配套 Torch/CUDA，不混入历史快照的 Python 路径。

原始源码开发基准为 `v0.1.1-baseline`，见[基准说明](docs/BASELINE.md)；后续团队开发从最新 `main` 创建功能分支。版本变化记入 [CHANGELOG](CHANGELOG.md)。

## 仓库结构

按 rl-garden 的职责分层组织，复用现有 `starter` 包，不重复建立另一套模型/环境框架：

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
│   │   ├── rewards/           # 训练奖励，不改正式评分
│   │   └── logging/           # 指标记录
│   ├── configs/              # 当前任务配置
│   └── patches/              # 相对课程原包的可逆修改
├── course/scripts/           # 独立诊断工具
├── configs/                  # 公共实验计划模板
├── scripts/                  # 仓库、来源与计划检查
├── tests/                    # 项目回归测试
├── third_party/              # 固定上游源码，不作实验草稿
├── manifests/                # 源码、版本与哈希清单
├── docs/                     # 架构、接口、运行与复现说明
└── assets/                   # 已批准的项目素材
```

[架构与扩展约定](docs/ARCHITECTURE.md)说明各层职责、模型接入位置和协作流程。模型在合入后共同存在于 `main`；分支用于开发改动，不用于永久存放不同算法、seed 或 checkpoint。

镜像、数据、权重、运行日志与私有部署记录不进入 Git，仓库记录其来源、许可、固定版本与校验方式。

## 开发路线

1. 对齐观测、动作、图像预处理和时序，核验终态、超时及安全接口。
2. 验收少量完整演示：单位、关节顺序、同步、重放和成功标记；按完整回合划分数据。
3. 建立规则、ACT 与小规模 VLA 基线，再比较 OpenVLA-OFT、π₀.₅ 等候选。
4. 先建立普通 RL 对照，再探索 VLA 直接微调或 residual；分别记录示范、交互与计算成本。
5. 使用相同任务、场景、seed 和成功判据评测，做消融与失败分析；通过部署检查后再上真机。

BC和预训练VLA用于初始化或对照，不替代课程要求的RL训练。评测区分已见、组合外推与完全未见条件，记录成功率、时间、误抓/漏抓与放置误差；具体成功条件和时限以正式发布为准，不套用其他示例的默认上限。

RLinf 可为 VLA 的 SFT→RL、流式动作策略优化、chunk 执行及评测提供参考，但不是开箱即用的课程 SO-101 训练器。具体可借鉴内容、版本差异与适配缺口见 [RLinf 接入评估](docs/RLINF.md)。目前未将其加入运行依赖。

### 上游版本

以下是历史源码快照；当前 LiftCube 使用课程配套的 Isaac Sim 4.5.0 / Isaac Lab v2.1.1 / LeIsaac v0.1.2，二者分开管理。

| 项目 | 历史源码版本 | 固定提交 |
| --- | --- | --- |
| LeRobot | v0.4.2 | [58f70b6](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |
| LeIsaac | 0.4.0 | [24d3bcd](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| IsaacLab | v2.3.0 | [3c6e67b](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) |

完整导入保留上游许可和来源清单；45 项 LFS 测试素材已作为普通 Git 文件保存，无需另行 LFS 下载或子模块初始化。

## 文档

- [开发起步与阶段验收](docs/GETTING_STARTED.md)
- [LiftCube 源码、镜像与运行方法](course/liftcube/README.md)
- [架构、扩展位置与团队协作](docs/ARCHITECTURE.md)
- [RLinf 对本项目的帮助与接入边界](docs/RLINF.md)
- [观测、动作、数据与恢复契约](docs/INTERFACE_CONTRACTS.md)
- [训练前验证与限制](docs/TRAINING_VALIDATION.md)
- [依赖与外部资产](docs/DEPENDENCIES.md)
- [部署与复现边界](docs/REPRODUCIBILITY.md)
- [源码导入清单](docs/IMPORT_MANIFEST.md)
- [修改与版本管理](CONTRIBUTING.md)
- [海报与图像说明](assets/README.md)
- [第三方许可与版权](THIRD_PARTY_NOTICES.md)

## 参考与致谢

感谢 [LeRobot](https://github.com/huggingface/lerobot)、[LeIsaac](https://github.com/LightwheelAI/leisaac) 和 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 提供的开源基础。工程分层参考 [rl-garden](https://github.com/Nole326/rl-garden)，VLA 强化学习方案参考 [RLinf](https://github.com/RLinf/RLinf)。引用与分发遵循各自许可，不改变上游版权归属。

海报为课程项目海报优化调整后的中英双语版。机器人图像来源为 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101)。
