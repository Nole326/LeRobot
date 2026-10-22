# AGENTS.md

本文件供开发智能体使用，适用于整个仓库。按任务读取下表中的子级指令；跨目录修改同时读取双方。项目介绍与安装说明见 [README](README.md)，协作流程见 [CONTRIBUTING](CONTRIBUTING.md)。

## 从这里开始

1. 检查 `git status --short`、`git branch --show-current` 和相关 diff，保留已有改动。
2. 选择下表中的入口，读实际调用代码及相关测试，再修改。只读取当前任务需要的文档。
3. 按用户任务完成实现与针对性验证；缺少环境、资产或设备时，继续能独立完成的工作，并报告具体缺口。
4. 提交、推送、合并和真机运行按用户授权执行。已有明确授权无需重复询问；示例命令不额外授予运行权限。

这是课程集成仓库，不是可在根目录安装的 Hugging Face LeRobot 包；不要在根目录执行 `pip install -e .`。克隆后的工作不依赖维护者的聊天记忆、相邻项目或私有文档。

首次接手时先读[开发起步](docs/GETTING_STARTED.md)。仅修改代码或文档可直接使用根目录检查环境，无需先下载镜像、数据或权重；需要运行 LiftCube 时，再按子级指令准备配套环境。新功能从最新 `main` 建短期分支；用户已经指定工作分支时沿用该分支，不自动切换。

## 按任务找入口

| 任务 | 修改位置与必读入口 |
| --- | --- |
| 仓库检查、文档、来源清单 | `scripts/`、`tests/`、`manifests/`；[来源说明](docs/IMPORT_MANIFEST.md) |
| SO101 接口、LiftCube 数据/训练/评测 | `course/liftcube/src/starter/`；[LiftCube AGENTS](course/liftcube/AGENTS.md) |
| PickOrange 与独立诊断 | `course/scripts/`；[诊断工具 AGENTS](course/scripts/AGENTS.md)、[部署边界](docs/REPRODUCIBILITY.md) |
| ACT、SmolVLA、OpenVLA-OFT、π₀.₅、GR00T 接入 | 对应任务的策略包装；[架构](docs/ARCHITECTURE.md)、[接口契约](docs/INTERFACE_CONTRACTS.md)、该任务的子级 AGENTS |
| 正式多物体任务或真机 | 先确认实际提供的环境、SDK、标定与评分包；[接口契约](docs/INTERFACE_CONTRACTS.md) |

LiftCube 已有 SmallCNN BC/PPO 参考训练与评测，`starter.utils.build_policy()` 仍固定 SmallCNN。上游含 ACT/VLA 实现不代表当前任务已经接通。PickOrange 是另一套任务和运行环境；正式多物体与真机能力必须依据可获取的实现确认。

## 环境边界

| 工作范围 | 运行前提 |
| --- | --- |
| 根目录源码检查 | Git + Python 3.11+；不需要模拟器 |
| LiftCube | 配套镜像：Isaac Sim 4.5.0 / Isaac Lab 2.1.1 / LeIsaac 0.1.2；Python 3.10.15 |
| 历史 PickOrange 诊断 | 独立仿真/BC 环境；包装脚本含固定容器路径及 Python 3.11 路径 |
| `third_party/` | 冻结参考：LeRobot v0.4.2 / LeIsaac 0.4.0 / IsaacLab v2.3.0；不等于 LiftCube 镜像内版本 |

依赖、镜像、资产与权重获取入口见 [DEPENDENCIES](docs/DEPENDENCIES.md)。运行前核对解释器、实际导入路径、设备和数据/输出位置。不要把全部上游路径加入 `PYTHONPATH`，或为运行仓库检查升级模拟器的 Python/Torch/CUDA。可选模拟器与模型依赖应延迟到对应入口加载，保持纯配置与软件检查可独立运行。

## 修改约定

- 当前任务共用代码位于 `course/liftcube/src/starter/`：`interfaces/` 定义契约，`envs/` 处理环境，`data/` 处理演示，`training/` 负责训练与存档，`evaluation/` 负责评测。模型包装复用这些边界，不复制环境和评分逻辑。
- `course/liftcube/configs/` 是运行配置；根 `configs/experiment-plan.toml` 只是计划模板。`course/scripts/` 保持为独立诊断工具。
- `third_party/` 为冻结来源，优先在外部适配。未经明确任务要求，不改第三方源码、依赖版本或生成文件来消除检查失败。
- 沿用邻近代码与测试风格，复用现有函数、标准库和测试。修复实际调用链，只增加当前需求所需的实现；不预建空注册器、无关抽象或按模型复制整套训练器。
- 代码标识符应清晰，注释解释不明显的原因。公开说明、提交消息和 PR 面向未参与对话的读者，不记录聊天过程或个人操作流水。

## 不得改变的语义

完整定义以 [INTERFACE_CONTRACTS](docs/INTERFACE_CONTRACTS.md) 与任务配置为准：

- SO101 六维为五个机械臂关节加夹爪。明确关节顺序、单位、标定、绝对/增量和夹爪范围；不能截掉模型输出的一维冒充适配。
- 请求动作、限幅后发送动作、实测位置分开；发送不等于到位。不得用下一帧实测位置替换请求动作标签，或静默裁剪演示标签。
- 兼容补零不是实际速度/末端反馈。必要关节状态缺失应报错；派生状态注明来源与时序。
- 相机布局、图像处理、环境周期、策略频率、chunk 与执行长度显式对应；训练和评测共用兼容执行层。
- 物体/目标真值、分割和成功标记不得进入普通视觉策略或其 metadata；教师、监督、奖励、特权 critic 与评分分别隔离。
- 自动 reset 前捕获终态；区分 `terminated`、`truncated`、bootstrap 与成功。接口修正不改变物理、限位、执行器或评分条件。
- 推理导出、权重初始化和完整续训分别处理。续训需同点模型、优化器、计数、数据位置/replay 与相关 RNG；重新 seed 不等于恢复。
- 成功率仅统计同协议完整回合。保留失败与 partial，不换 seed、跳回合或覆盖旧证据；加载和进程退出不能证明测试完成。

## 验证

先运行受影响的最小检查，复用身份一致的有效证据。目标测试和仿真 smoke 见对应子级 AGENTS。文档调整检查链接、命令和清单，不触发 GPU 或训练。

提交前，从仓库根目录运行统一检查（Python 3.11+）：

```bash
python scripts/check_project.py
git diff --check
git diff --cached --stat
```

统一检查覆盖来源 SHA、项目 Python 语法、Markdown 本地链接、海报和软件单测；不启动模拟器。依赖缺失导致的 skip 必须披露，不能当相关功能通过。交付时写明实际命令、结果和未验证部分，不把合成测试写成真实仿真/真机成绩。

`python scripts/inspect_plan.py configs/experiment-plan.toml --print-config` 只检查计划；退出码 0 为静态通过、2 为无效、3 为模板未填齐，均不表示训练已运行。

## 改动后同步什么

| 变化 | 同步内容 |
| --- | --- |
| 输入、动作、时序、数据或恢复语义 | 对应配置、契约、模型/数据身份与针对性测试 |
| `course/` 中新增或修改公开文件 | `manifests/course_files.json` 中受影响文件的大小与 SHA |
| 修改课程原始受管源码 | 同时更新 `manifests/liftcube_source.json` 当前 SHA 与 `course/liftcube/patches/team-interface.patch`，保留原始 SHA |
| 入口、命令、依赖或执行规则改变 | 根目录及受影响子级 AGENTS，同 PR 更新；其他变更无需机械改日期 |
| 用户可见功能或用法改变 | 对应文档；首页信息同步中英文 README，重要变化写 CHANGELOG |

`manifests/projects.json` 和 `source_files.json` 固定上游身份。不要用 `audit_repository.py --write-manifests` 消除普通检查差异。来源检查按 Git 索引核对 `course/` 文件集合；获准提交新增文件时，文件与清单一起暂存再检查。新增团队说明不伪造课程原归档条目。

## 执行与公开边界

- 按实际使用者的设备、预算和平台规则执行。平台配置留在仓库外；缺少部署信息只阻塞相关运行。不要停止或重启无关/健康任务、覆盖产物或执行全局清理。
- 真机运行需实际 SDK、校准、限速、急停与用户确认。仿真通过不能替代真机安全条件。
- 不提交凭据、完整环境变量、私有配置、个人机构/服务器/账号/设备信息、本机路径、预约、个人运维规则或未获准材料。公开示例使用通用占位符或源码必需的容器路径；数据、权重和运行日志保留在外部存储。
- 发布前复核文件、提交正文及 Git 作者/提交者身份；保持用户同意的公开身份，不擅自改全局配置或重写历史。保留许可、版权与来源，访问许可不等于再分发许可，详见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
- AGENTS 仅保留可执行的项目约定，不要求特定 agent 插件，也不存实验成绩、PID、故障流水或机器配置。详细用户教程放既有文档；[PR 模板](.github/pull_request_template.md)用于交付复核。
