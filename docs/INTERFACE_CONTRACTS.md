# 接口约定 / Interface contracts

本页规定我们开发时要核验的边界，不代替正式环境规范。配置模板是设计检查表，不是已接入的生产接口；空值表示待确认。接口变更应有新的 schema 标识、转换测试和迁移说明，不能只改一个维度数字。

This page defines development checks, not the formal environment specification. The configuration template is a design checklist, not an integrated production interface; empty values are unresolved. Interface changes need new schema identifiers, conversion tests and migration notes, not just a changed dimension.

## 1. 观测 / Observations

策略输入显式列出图像键、本体字段顺序、形状、单位、归一化、时间戳和任务条件。图像必须对应动作前的观测，记录曝光/采样时刻与策略读取时刻。物体和目标信息来自许可的视觉或任务条件通道，不能把仿真真值改名为本体字段后传入网络。训练 reward、诊断信息与网络输入分开；critic 输入也须独立确认许可。

List image keys, ordered proprioceptive fields, shapes, units, normalization, timestamps and task conditioning explicitly. Images must represent pre-action observations; record capture and policy-read times. Object and target information must come from permitted visual or task-conditioning channels, not simulator truth renamed as proprioception. Keep rewards and diagnostics separate from network inputs, and review critic-input permissions separately.

当前计划检查器只允许六个已登记 SO-101 关节字段；这是保守的开发白名单，不是宣布正式任务只有六维本体。增加末端或其他字段时，应先验证来源再扩展白名单。字段名称检查无法证明内容没有真值泄漏，还需要运行时来源检查。

The plan inspector currently permits six registered SO-101 joint fields. This is a conservative development whitelist, not a claim that the formal task has only six proprioceptive dimensions. Validate provenance before adding end-effector or other fields. Name checks cannot prove the absence of truth leakage; runtime provenance inspection is also required.

## 2. 动作 / Actions

区分四层：模型归一化输出 → 数据集动作语义 → 执行器命令 → 实际机器人运动。记录每一层的转换、坐标系、分量顺序、单位和边界。`joint_contract.py` 转换的是关节弧度与标定归一化坐标，后者不是角度；六关节目标也不是六维末端位姿。

Separate four layers: normalized model output → dataset action semantics → actuator command → measured robot motion. Record conversions, frames, component order, units and bounds at each layer. `joint_contract.py` converts joint radians and calibration-normalized coordinates, not physical degrees; six joint targets are not a six-dimensional end-effector pose.

ACT/VLA 的动作 chunk 要明确预测长度、每次实际执行步数及重规划方式。reset、人工接管、终止或任务条件变化后清空不匹配的缓存。限幅发生在哪层要写清；不能多乘一次尺度，也不能用限幅掩盖坐标或单位错误。末端动作不可达时明确报告，不能无说明地偷偷换轴或放宽姿态约束。

For ACT/VLA chunks, specify prediction length, executed steps and replanning behavior. Clear incompatible caches after reset, intervention, termination or changes in task conditioning. Specify where clipping occurs; do not apply scales twice or hide frame/unit errors with clipping. Report unreachable end-effector requests instead of silently changing axes or relaxing orientation constraints.

控制周期满足 `policy_hz × physics_dt_s × physics_steps_per_action = 1`，这里一个 action 指一次低层命令，不是整个 chunk。该关系仅适用于固定步长、固定保持步数；异步控制需要单独的时间契约。验证包括零动作、正负小动作、各轴响应、限速限位、夹爪方向及实际轨迹，不能只检查张量形状。

For fixed-step control, `policy_hz × physics_dt_s × physics_steps_per_action = 1`; an action here is one low-level command, not a whole chunk. Asynchronous control requires a separate timing contract. Test zero commands, signed small motions, per-axis response, rate/position limits, gripper direction and measured trajectories—not tensor shapes alone.

## 3. 数据与 BC / Data and BC

一条 transition 的顺序为 `obs_t → action_t → obs_next`；监督学习通常使用动作前观测预测该动作。记录原始命令与实际执行命令、时间戳、回合ID、场景/目标、阶段、结果、操作者接管与异常。不要通过整体平移一帧来“修正”延迟，应先测量延迟和采样含义。关节示范转末端命令不是简单改字段名，需要明确可重放的转换。

A transition is `obs_t → action_t → obs_next`; supervised learning normally predicts the action from its pre-action observation. Record requested and executed commands, timestamps, episode IDs, scene/target, stage, outcome, interventions and anomalies. Measure latency and sampling semantics before shifting labels. Converting joint demonstrations to end-effector commands requires an explicit replayable transformation, not a renamed column.

原始失败轨迹保留，可单独标记是否纳入 BC。训练/验证按完整回合划分，同一初始场景的重复录制按组隔离，正式测试场景不进入演示训练。只用训练集拟合归一化。ACT chunk 不跨回合，尾部 padding 不参与损失或统计；SmolVLA 同样核验动作有效维度和时间对齐。诊断用的短轨迹不算成功示范集。

Keep raw failures and label whether they enter BC. Split complete episodes, group repeated recordings of the same initial scene, and exclude formal test scenes from demonstration training. Fit normalization on training data only. ACT chunks must not cross episodes; padded tails must not enter loss or statistics. Check valid action dimensions and timing for SmolVLA as well. Short diagnostic trajectories are not successful demonstration datasets.

## 4. RL 转移与评测 / RL transitions and evaluation

自动 reset 的环境可能返回下一回合初始观测。replay 中上一回合的下一状态必须来自 reset 前真正末帧，图像也要确认刷新时序；下一轮策略则读取 reset 后观测。真正终止不 bootstrap；外部时间截断通常使用末帧 bootstrap；若任务定义为有限时域终止，则遵循该任务规则，不能仅凭 `done` 推断。结束标志与原因分别保存。

Auto-reset environments may return the next episode's initial observation. Replay must use the actual pre-reset final observation for the previous transition, including correctly refreshed images; the next policy call uses the reset observation. True terminals do not bootstrap. External time truncations usually bootstrap from the final observation; task-defined finite-horizon terminals follow that task's rules. Never infer this from `done` alone; retain flags and reasons.

正式评测清单、成功条件、时间上限、控制器和指标冻结并记录版本/哈希。训练 reward 与评测成功独立。逐回合保存实际场景、结果、时间和失败原因，不筛选成功 seed。中断从首个未落盘回合恢复并避免重复计数，不宣称恢复了回合中途的全部仿真随机现场。

Freeze and version/hash the evaluation manifest, success criteria, horizon, controller and metrics. Keep training rewards independent from evaluation success. Persist actual scenes, outcomes, timing and failure reasons per episode, without success-filtering seeds. Resume from the first uncommitted episode without double-counting; do not claim restoration of the entire mid-episode simulator state.

## 5. 存档与恢复 / Checkpoints and recovery

可评测的模型与可续训的存档是两种产物。前者需要权重、配置、归一化和接口标识；后者还需要优化器、调度器、计数、replay或采样排列/游标、Python/NumPy/Torch CPU/CUDA及独立随机流。多worker预取和规划器单独登记。所有文件来自同一保存点，写入完成并校验后才发布完整性标记，不覆盖上一个有效存档。

An evaluable model and a resumable checkpoint are different artifacts. The former needs weights, configuration, normalization and interface identifiers; the latter also needs optimizer, scheduler, counters, replay or sample order/cursor, Python/NumPy/Torch CPU/CUDA and private RNG streams. Inventory worker prefetch and planners separately. Publish a completion marker only after all matching files are written and verified, preserving the previous valid checkpoint.

先构建对象和加载参数，再按验证过的顺序恢复随机流；首次 reset 不得重新用 seed 覆盖已恢复的随机状态。没有完整仿真现场时，从新回合边界恢复并清空 ACT/VLA 缓存，不拼接跨回合 transition。恢复验收要比较下一批样本、下一次更新、计数和随机流，并记录保存耗时与资源峰值。现有 CPU 工具的边界见[训练验证](TRAINING_VALIDATION.md)。

Construct objects and load parameters before restoring RNG streams in a validated order. The first reset must not replace restored RNG with reseeding. Without full simulator state, resume at a new episode boundary, clear ACT/VLA caches and avoid cross-episode transitions. Compare the next sample batch, update, counters and RNG streams during recovery tests; measure save time and resource peaks. See [training validation](TRAINING_VALIDATION.md) for the current CPU tools' limitations.
