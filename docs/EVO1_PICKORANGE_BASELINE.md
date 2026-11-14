# Evo-1 PickOrange 接入 / Evo-1 PickOrange integration

Evo-1 是用于 PickOrange 的小规模视觉语言动作模型。本服务使用 [Evo-1 项目提供的 LeRobot 后端](https://github.com/MINT-SJTU/Evo-1/tree/evo1-lerobot/evo1_lerobot)，复用 [PickOrange 评测器和轨迹审核](../course/pickorange/README.md)。

Evo-1 is a compact vision-language-action model. This service uses the Evo-1 project's LeRobot backend and the shared PickOrange evaluator and trajectory audit.

| 项目 / Item | 接口 / Contract |
| --- | --- |
| 观测 / Observation | 前视 `front` 与腕部 `wrist`，各 640×480 RGB，实测六维关节位置和协议任务文本 / Front and wrist RGB, six measured joint positions, and the protocol instruction |
| 归一化 / Normalization | 六维本体与动作使用配套训练数据的最小值、最大值。原生模型负责图像处理 / Use matching training minima and maxima for state and actions. The native model preprocesses images. |
| 动作 / Action | 预测 50 个六维绝对 motor 目标。关节顺序和夹爪范围见 [SO101 映射](../course/scripts/joint_contract.py)。夹爪 0 为闭合，100 为张开 / 50 six-dimensional absolute motor targets in SO101 joint order. The gripper ranges from closed at 0 to open at 100. |
| 执行 / Execution | 32 步去噪。执行完整 50 步，每步保持两个 60 Hz 物理步，动作频率 30 Hz。回合上限 60 仿真秒，沿用原生成功判定和动态夹爪辅助 / 32 denoising steps. Execute all 50 targets with two physics ticks each. At 30 Hz, a full 60-second episode uses 36 queries. Native task scoring and gripper-effort assistance apply. |

本地任务导出目录需要 `config.json`、`model.safetensors` 及对应训练数据的 `meta/stats.json`。在独立推理环境中准备后端及其依赖，提前缓存配置对应的视觉语言主干与 tokenizer。模型服务使用原生处理器，评测器负责目标执行。后端版本由[实验协议](../course/pickorange/protocols/evo1-native-assisted-30hz.json)指定，运行命令见 [PickOrange 运行指南](../course/pickorange/README.md#evo-1-实验性接入--experimental-integration)。

The local export needs the policy config, weights, and matching training dataset statistics. Prepare a separate inference environment with that backend's dependencies and cache the required backbone and tokenizer. The experimental protocol specifies the backend and timing.

接口已通过 CPU 检查。实际任务权重前向、仿真闭环及新协议成绩待验收。

CPU checks cover the interface. Checkpoint inference, simulator rollout, and scores under this protocol remain pending.
