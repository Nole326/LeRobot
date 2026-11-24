# SO-101 PickOrange

将三颗橙子放入盘中，再让机械臂返回休息姿态。这里说明环境与诊断入口；模型闭环评测和控制脚本采集见[PickOrange运行指南](../course/pickorange/README.md)。任务方法和结果与[LiftCube](../course/liftcube/README.md)分开管理。两者差异见[任务对照](TASKS.md)。

Put three oranges on the plate and return the arm to rest. This guide covers the public environment and diagnostic tools currently in the repository. Keep its methods and results separate from LiftCube; see the [task comparison](TASKS.md).

## 代码与支持范围 / Code and scope

| 内容 / Component | 入口 / Location |
| --- | --- |
| 环境与描述 / Task definition | [pick_orange_env_cfg.py](../third_party/leisaac/source/leisaac/leisaac/tasks/pick_orange/pick_orange_env_cfg.py) |
| 原生成功 / Native success | [mdp/terminations.py](../third_party/leisaac/source/leisaac/leisaac/tasks/pick_orange/mdp/terminations.py) |
| 单臂观测 / Single-arm observations | [single_arm_env_cfg.py](../third_party/leisaac/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py) |
| 控制器转换 / Action mapping | [action_process.py](../third_party/leisaac/source/leisaac/leisaac/devices/action_process.py)、[joint_contract.py](../course/scripts/joint_contract.py) |
| 环境/RGB检查 / Environment smoke | [validate_sim.py](../course/scripts/validate_sim.py) |
| 视觉规则评测 / Visual-rule evaluation | [运行命令、相机标定与 55 秒结果 / Commands, calibration and 55-second results](../course/pickorange/visual/README.md) |
| 轨迹/末帧/重放检查 / Trajectory diagnostics | [validate_rollout.py](../course/scripts/validate_rollout.py) |
| ACT数据桥 / ACT data bridge | [validate_bridge.py](../course/scripts/validate_bridge.py)、[训练诊断说明](TRAINING_VALIDATION.md) |

双相机 VLA 服务、原生辅助评测和轨迹审核见 [PickOrange运行指南](../course/pickorange/README.md)。OpenVLA-OFT 的双视角接口需要配套的受训权重。各模型的适配状态见[任务对照](TASKS.md)。

The [PickOrange run guide](../course/pickorange/README.md) covers dual-camera VLA services, native-assisted evaluation and trajectory auditing. The OpenVLA-OFT interface requires a matching trained export. See the [task matrix](TASKS.md) for model support.

学习策略的训练、加载及固定场景评测分别见 [ACT](../course/pickorange/act/README.md)、[ACT 残差 SAC](../course/pickorange/sac/README.md) 和 [GR00T 残差 SAC](../course/pickorange/groot_sac/README.md)。 / Follow the corresponding policy guide for training, loading and fixed-seed evaluation.

## 独立运行环境 / Separate runtime

### 已验证配置 / Verified configuration

以下 Linux 容器配置已通过 PickOrange 的真实 `reset`、物理步进和前视／腕部双 RGB 更新检查。它与 [LiftCube 课程镜像](../course/liftcube/README.md) 分开使用。

This Linux container setup has passed real PickOrange reset, physics stepping, and front/wrist RGB update checks. Keep it separate from the LiftCube course image.

| 组件 / Component | 实测版本 / Observed version |
| --- | --- |
| NVIDIA 基础镜像 / Base image | [`nvcr.io/nvidia/isaac-lab:2.3.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/isaac-lab/2.3.0/layers) |
| 镜像内系统 / Container OS | Ubuntu 24.04, glibc 2.39 |
| Isaac Sim | `5.1.0-rc.19+release.26219.9c81211b.gl` |
| Isaac Lab | `2.3.0`（镜像发行标签 / image release tag） |
| LeIsaac 任务源码 / Task source | `0.4.0` · [`24d3bcd`](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| Python | `3.11.13` |
| PyTorch / torchvision | `2.7.0+cu128` / `0.22.0+cu128`（CUDA 12.8 构建 / CUDA 12.8 builds） |
| LeRobot（BC 应用层 / BC application layer） | `v0.4.2` · [`58f70b6`](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |

基础镜像不含 PickOrange 任务和场景资产。按[依赖清单](DEPENDENCIES.md)和[外部资产清单](../manifests/external_assets.json)补齐源码、`kitchen_with_orange.zip` 和 `so101_follower.usd`。仿真和 BC 使用独立的 `venv_sim`／`venv_bc` 应用层，保留镜像的 PyTorch/CUDA；各 VLA 后端按[运行指南](../course/pickorange/README.md)单独配置。仓库未提供完整的可移植安装锁。

The base image does not include the PickOrange task or scene assets. Use the linked inventories for the task source, kitchen scene, and SO-101 USD. Keep simulator and BC dependencies in separate overlays without replacing the bundled PyTorch/CUDA. Configure each VLA backend separately. A complete portable install lock is not provided.

### 待验证方案 / Not yet validated

- **魔搭免费 Notebook 原生安装 / ModelScope Notebook without Docker.** [Isaac Sim 5.1 的 pip 安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)需要 Python 3.11，Linux 下要求 glibc ≥ 2.35。若预装的是 PyTorch 2.3.1+cu121，应新建独立 Python 环境，不覆盖 Notebook 的预装依赖。按 [LeIsaac 的版本配套](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/)安装；pip 发布版 5.1.0 与上表实测镜像的 RC 构建不同。 / Use a separate environment with the documented prerequisites and matching packages. The 5.1.0 pip release is a different build from the tested image.
- **PAI-DSW A10 自定义镜像 / PAI-DSW A10 with a custom image.** 如实例提供[自定义镜像入口](https://help.aliyun.com/zh/pai/create-and-manage-dsw-instances)，可尝试上表的 NGC 镜像；镜像能启动不代表相机已经可用。 / If custom images are available on the instance, try the pinned NGC image. An instance that starts is not yet a camera validation.

两种方案都要先确认实例能提供 NVIDIA Vulkan/RTX 渲染，再在不加载策略模型的情况下检查 `reset`、`step` 和两路 RGB 更新。`nvidia-smi` 能看到显卡只说明设备可见。 / For either option, verify NVIDIA Vulkan/RTX access and run a reset/step smoke test with both cameras before loading a policy. `nvidia-smi` alone does not validate rendering.

### 容器内路径 / Launcher layout

```text
/workspace/course/
├── scripts/                 # 本仓库course/scripts中的工具 / tools
├── leisaac/source/leisaac/  # 匹配的任务源码 / matching task source
├── venv_sim/                # 仿真应用覆盖层 / simulator overlay
├── venv_bc/                 # LeRobot/ACT应用环境 / model environment
├── cache/                  # 不放数据集 / cache outside datasets
└── logs/
/data/course/assets/         # 独立外部场景资产 / external scene assets
/data/course/validation/     # 诊断产物 / diagnostic outputs
```

将本地目录挂载到上述容器路径。`prepare_assets.py` 下载并解压[外部资产](../manifests/external_assets.json)。历史文件 `constraints-isaac51.txt` 记录安装冲突，运行版本以本页环境表为准。

Mount local directories at the paths above. Asset preparation downloads and extracts the files in the manifest. The runtime table lists simulator versions. The historical constraints file records dependency conflicts.

## 观测、动作与评分 / Observation, action and scoring

- 原生`policy`组含front/wrist RGB（各480×640）、关节位置/速度及其他仿真字段；`subtask_terms`和物体真值只能用于许可评分/诊断，不送入视觉策略。仿真有速度不意味着真机接口会提供速度。 / Native simulator fields are not an automatic policy whitelist or hardware contract.
- `joint_pos`按实际关节顺序取弧度；模型使用motor时先按明确标定转换。执行端的`so101leader`动作管理器再处理scale/offset，不能直接将motor或绝对q送给`env.step`。 / Convert model units to absolute radians, then to native raw actions using actual scale/offset.
- 原生成功要求三颗橙子同时满足盘区XYZ条件，且机械臂回位；一颗入盘或曾经命中过盘区是进度，不是整任务成功。 / Native success requires all three objects and the rest pose.
- 运行前记录实际`physics_dt`、`step_dt`、动作保持和时限；20帧10Hz、128×128桥接条件仅属于该诊断，不是统一策略规格。 / Record real timing; bridge settings are not a task-wide specification.
- 环境提供双相机与课程允许策略使用双相机是不同问题；按实际考核要求固定front-only或双视角协议，报告真实输入，不把双图结果当单图结果。 / Freeze the permitted view protocol and report it accurately.
- 保持原生物理、执行器、成功条件及reset前终态；键盘/状态机等上游路径可能启用重力或夹爪辅助，控制脚本采集与视觉baseline必须分组。 / Separate assisted control scripts from visual baselines.

## 已配置环境中的检查 / Checks in a configured runtime

仅在已获准运行仿真、资源和依赖均满足时，于上述容器内执行。`NEW_TAG`替换为未使用的名称；脚本会写日志与轨迹，不启动训练。

Run only with simulation approval and a configured runtime. Use a fresh tag; this writes diagnostic outputs, not training results.

```bash
cd /workspace/course
bash scripts/course_python.sh scripts/validate_rollout.py \
  --run-tag NEW_TAG --headless --enable_cameras
```

来源记录和轨迹验证通过后，才做ACT格式/前向桥接，使用另一个新tag / Validate source identity before the separate ACT bridge:

```bash
bash scripts/course_bc_python.sh scripts/validate_bridge.py \
  --source-tag SOURCE_TAG --output-tag NEW_BRIDGE_TAG
```

不要批量执行`validate_*.py`，也不要import探测其可用性；部分脚本在顶层创建应用/写文件。`validate_sim.py`使用固定输出路径，已有结果时不得覆盖重跑。详细工具副作用见[目录AGENTS](../course/scripts/AGENTS.md)。

Do not batch-run or import diagnostics as probes: some create applications or files at module scope. Preserve existing fixed-path outputs.

## 复现策略评测 / Reproducing a policy evaluation

从[资源清单](RESOURCES.md)选择权重，并按[训练与评测指南](TRAIN_AND_EVALUATE.md)启动对应的模型端和仿真端。复现已有成绩时，核对控制模式、双相机顺序、动作频率、夹爪设置、场景种子和时限。残差策略还需匹配冻结基座与部署幅度。

Choose a checkpoint from [Resources](RESOURCES.md) and follow the [training and evaluation guide](TRAIN_AND_EVALUATE.md) for its model and simulator. Match the control mode, camera order, action rate, gripper settings, seeds and time limit to the published evaluation. For residual policies, also match the frozen base and deployment scale.
