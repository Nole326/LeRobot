# RLinf接入评估 / RLinf integration assessment

[中文入口](../README.md) · [English entry](../README.en.md) · [本仓库架构 / Architecture](ARCHITECTURE.md)

核查日期：2026-10-21。参考 [RLinf](https://github.com/RLinf/RLinf) 公开文档及源码，读取时main为 [`91a302ec`](https://github.com/RLinf/RLinf/tree/91a302ec50ddacd1fdabc5859b85e3661d768236)。文档的latest会变化；这是接入分析，不是安装、训练或SO-101性能验收。目前未复制其实现、增加依赖或启动任务。

Reviewed on 2026-10-21 against public [RLinf](https://github.com/RLinf/RLinf) documentation and source; main resolved to [`91a302ec`](https://github.com/RLinf/RLinf/tree/91a302ec50ddacd1fdabc5859b85e3661d768236). Latest documentation can change. This is an integration assessment, not installation, training or SO-101 performance acceptance. No implementation was copied, dependency added or task launched.

## 可借鉴什么 / Reusable directions

| 方向 / Direction | 已公开的参考 / Public reference | 对本项目的帮助与缺口 / Benefit and gap |
| --- | --- | --- |
| π₀ / π₀.₅ | [PPO/GRPO与流策略](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html) | 可研究流动作的随机采样、log-prob、value head及RL更新；仍需课程数据/动作语义适配 / Reference flow-policy sampling, log-probabilities, value heads and RL updates; course data/action adaptation remains necessary |
| GR00T | [SFT冷启动→PPO](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html) | 可复用训练思路，核对chunk与去噪策略；不是现有PickOrange权重的即插即用续训入口 / Study the training path and chunk/denoising semantics; not a drop-in continuation of PickOrange weights |
| OpenVLA-OFT | [PPO/GRPO范例](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/openvla_oft.html) | 可参考动作chunk与RL接线；示例的7维末端动作不是SO-101六关节目标 / Reference chunk/RL integration; example 7-D EE actions are not six SO-101 joint targets |
| π₀-FAST | [LeRobot策略与token级GRPO](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0_fast.html) | 提供离散动作token的RL路线，区别于π₀.₅连续流策略 / A discrete action-token RL route, distinct from π₀.₅ flow policies |
| IsaacLab | [Franka堆方块](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/isaaclab.html) | 环境/rollout/actor拆分及TensorBoard参考；任务、机械臂、运行时均不同 / Environment/rollout/actor separation and TensorBoard reference; different task, robot and runtime |
| SO-101真机 / Hardware | [SO101设置与控制](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/so101.html) | 校准、关节反馈、遥操和相机时序参考；文档明确尚无受支持的SO101完整任务训练流程 / Calibration, feedback, teleoperation and camera-timing reference; no supported complete SO101 task-training workflow yet |

## π₀-FAST (LeRobot)并不等于SO-101适配 / LeRobot integration is not SO-101 adaptation

这里的LeRobot指 `PI0FastPolicy` 软件实现。该范例使用 `lerobot/pi0fast-libero`，任务是LIBERO-Long，动作7维，参考训练配置为两节点16GPU，并不是课程LiftCube的默认运行方案。它保留原生动作token生成，再用teacher forcing复算策略概率进行优化。详见[范例](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0_fast.html)与[固定配置](https://github.com/RLinf/RLinf/blob/91a302ec50ddacd1fdabc5859b85e3661d768236/examples/embodiment/config/model/pi0_fast.yaml)。

LeRobot here denotes the `PI0FastPolicy` implementation. The example uses `lerobot/pi0fast-libero` on LIBERO-Long with seven-dimensional actions and a two-node, 16-GPU reference training setup, not course LiftCube defaults. It generates native action tokens and replays their probabilities with teacher forcing for optimization. See the [recipe](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0_fast.html) and [pinned configuration](https://github.com/RLinf/RLinf/blob/91a302ec50ddacd1fdabc5859b85e3661d768236/examples/embodiment/config/model/pi0_fast.yaml).

不能把该范例的成功率当作SO-101、课程物体或单卡性能。若走此路线，要重新准备匹配课程动作的训练数据、action tokenizer、统计和checkpoint；只把 `action_dim` 从7改为6不构成适配，也不能复用LIBERO动作token统计。

Its success rates do not establish SO-101, course-object or single-GPU performance. This route needs course-aligned training data, action tokenizer, statistics and checkpoints. Changing `action_dim` from seven to six is not adaptation, and LIBERO action-token statistics are not interchangeable.

## 需要对齐的边界 / Integration boundaries

1. **观测 / Observation**：当前课程入口为单front与六关节位置；速度/末端位置补零不能冒充测量。模型所需图像键、尺寸、历史、padding和语言输入逐项核对。 / Match image keys, sizes, history, padding and language to the single-front/six-joint course interface; zero slots are not measured state.
2. **动作 / Action**：确认归一化、单位、关节顺序、绝对/增量及夹爪方向；用实际执行命令记录动作。 / Verify normalization, units, joint order, absolute/delta semantics and gripper direction; record applied commands.
3. **时间 / Timing**：分开网络预测horizon、实际chunk执行长度和环境tick；不能从帧数直接推出任务秒数。 / Separate network horizon, executed chunk length and environment ticks; frame counts do not directly define task seconds.
4. **终态 / Terminal state**：适配reset前终帧、terminated/truncated和bootstrap；RLinf的 `success_once` 与课程最终成功条件分别记录。 / Preserve pre-reset observations and termination/bootstrap semantics; distinguish RLinf's `success_once` from course scoring.
5. **训练 / Training**：连续流策略和离散token策略的概率计算不同；不能把普通高斯PPO损失直接套在任意VLA输出上。 / Flow and token policies need different probability calculations; ordinary Gaussian PPO cannot simply wrap arbitrary VLA predictions.
6. **恢复 / Recovery**：核对模型、优化器、采样、rollout与独立RNG存档，验证相关恢复路径；不能只加载模型或重新seed。 / Check model, optimizer, sampling, rollout and independent RNG state, and validate the affected recovery path—not weight loading or reseeding alone.

## 环境与资源 / Runtime and resources

RLinf的IsaacLab示例是Franka堆方块，第三人称＋腕部相机、7维末端动作，文档使用Isaac Sim 5.1.0。当前课程镜像是4.5.0；不能为了安装RLinf原地升级课程环境。固定[任务注册表](https://github.com/RLinf/RLinf/blob/91a302ec50ddacd1fdabc5859b85e3661d768236/rlinf/envs/sim/isaaclab/__init__.py)只注册Franka堆方块，并未注册课程LiftCube；[π₀.₅配置](https://github.com/RLinf/RLinf/blob/91a302ec50ddacd1fdabc5859b85e3661d768236/examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml)也设置了双图像和7维动作。

RLinf's IsaacLab recipe uses Franka cube stacking, third-person/wrist cameras, seven-dimensional EE actions and Isaac Sim 5.1.0, unlike the course's 4.5.0 runtime. Do not upgrade the course environment in place. The pinned [task registry](https://github.com/RLinf/RLinf/blob/91a302ec50ddacd1fdabc5859b85e3661d768236/rlinf/envs/sim/isaaclab/__init__.py) registers Franka stacking, not course LiftCube; the [π₀.₅ config](https://github.com/RLinf/RLinf/blob/91a302ec50ddacd1fdabc5859b85e3661d768236/examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml) also uses two images and seven actions.

模型侧可以另建获准的隔离运行环境，再验证与课程模拟器的观测/动作传输；是否采用跨进程服务或同环境接入，要先验证版本和端到端时序。分进程本身不保证兼容，Ray/FSDP、并行环境、offload与保存仍需实际测峰，不能从推理显存推算训练可行性。

A separately approved model runtime may communicate with the course simulator, subject to version and end-to-end timing checks. Process separation alone does not prove compatibility. Ray/FSDP, parallel environments, offload and checkpointing need actual peak measurements; inference memory does not establish training feasibility.

## 建议采用顺序 / Recommended order

先用课程固定接口验收数据及ACT/小规模VLA对照，再选一个主要VLA完成SFT与闭环评测。若选择π₀.₅或GR00T直接做RL，优先研究RLinf的对应流策略训练路径；OpenVLA-OFT走独立的chunk/训练适配；π₀-FAST先作为备选研究线。不同时搬入所有后端，不把LIBERO或Franka权重当课程基线。

Accept the course data/interfaces and ACT/compact-VLA comparisons first, then SFT and evaluate one primary VLA. For direct π₀.₅/GR00T RL, prioritize RLinf's corresponding flow-policy training path; adapt OpenVLA-OFT separately. Keep π₀-FAST as an optional research direction rather than importing every backend or presenting LIBERO/Franka checkpoints as course baselines.

RLinf是训练基础设施，不提供本课程未知多物体任务定义、成功判据或可保证泛化的权重。迁移实现时保留Apache-2.0许可/NOTICE与修改记录；模型、tokenizer和数据另外核查来源条款。

RLinf is training infrastructure, not the course's unpublished task definition, scoring rules or guaranteed generalizing weights. Retain Apache-2.0 licensing/NOTICE and modification records when reusing code; check model, tokenizer and data terms separately.
