# PickOrange 双相机 ACT 与残差策略

PickOrange 在线训练与固定场景评测 / Online training and fixed-seed evaluation · [ACT residual SAC](../sac/README.md) · [GR00T residual SAC](../groot_sac/README.md)

[模型卡与固定场景成绩](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange) · [原始演示](https://huggingface.co/datasets/Nole326/LeIsaac-SO101-PickOrange) · [仿真与规则基线](../README.md)

部署观测为前视和腕部两路 640×480 的 RGB 图像（HWC、uint8）及六维实测关节位置（SO-101 标定电机坐标）。物理仿真为 60 Hz。策略每两步接收一次新观测，预测 15 步动作块，按系数 0.001 做时间聚合，当前目标保持两个物理步。

| 控制模式 | ACT 学习目标 | 残差作用位置 |
| --- | --- | --- |
| qpos | 五个臂关节和夹爪的绝对电机目标 | 在 ACT 关节目标上修正 |
| EE | 基座坐标系下的绝对目标位姿和夹爪角度；分段策略另预测段末信号 | 先由仿真控制器求逆运动学，再在标定电机坐标中修正 |

## 训练 / Training

在仓库根目录运行以下命令。安装仓库使用的 [LeRobot 与 PyTorch 依赖](../../../docs/DEPENDENCIES.md)及 PyAV。下载数据集对应的整个子目录，保留清单和每条轨迹的 NPZ、前视和腕部 MKV。数据检查会解码双相机并核对记录数，训练在内存中缓存无损压缩的 RGB 帧。

```bash
hf download Nole326/LeIsaac-SO101-PickOrange --repo-type dataset --include "vla-gr00t-qpos/*" "vla-gr00t-ee/*" --local-dir datasets/pickorange
python -m course.pickorange.act.train qpos datasets/pickorange/vla-gr00t-qpos/dataset.json --check-data
python -m course.pickorange.act.train ee datasets/pickorange/vla-gr00t-ee/dataset.json --check-data
python -m course.pickorange.act.train qpos datasets/pickorange/vla-gr00t-qpos/dataset.json --output outputs/act-qpos
python -m course.pickorange.act.train ee datasets/pickorange/vla-gr00t-ee/dataset.json --output outputs/act-ee
```

qpos 从 60 Hz 的偶数记录取观测，以其后第 2、4、…、30 个物理步的动作作为监督。两种 EE 策略都读取同一份 30 Hz 演示，从当前记录开始预测，用位置、旋转矩阵前两列和夹爪角度表达运动。

- `ee` 按每条演示的三个单颗搬运片段采样，在片段边界截断动作块并掩码。每个片段的最后两个记录时刻标为段末，用于训练额外的段末输出。发布的 ACT-2 使用此模式。
- `ee-full` 按完整演示采样，在整条轨迹结尾截断并掩码。策略连续执行三个橙子的搬运。训练时将上例的 `ee` 换成 `ee-full` 并指定独立输出目录。评测两种策略都选 `ee`，加载器从检查点读取段末配置。

默认有效 batch 为 64，每个 microbatch 为 8，共执行 100k 次优化器更新。主网络和视觉骨干的学习率分别为 0.0001 和 0.00001。输出目录包含 `loss.jsonl` 和按更新次数命名的 `checkpoint-<更新次数>/`。每个检查点包含模型配置、权重、归一化统计和执行配置，可直接加载推理。优化器和随机采样状态未保存，训练中断后无法从里程碑原样续训。再次启动训练需指定新的输出目录。可用 `--updates 1 --batch-size 2 --microbatch 1` 验证一次更新及检查点导出。

训练在前 3 次更新、之后每 25 次更新及最后一次更新记录损失。SwanLab 的 `train/loss` 与 `loss.jsonl` 使用相同的优化器更新次数和损失值。默认离线记录，`--check-data` 只检查数据。[SwanLab 配置](../../../docs/SWANLAB.md)

For qpos, sample every second observation and supervise the next 15 stride-two targets. Both EE modes use the same 30 Hz data. With `ee`, each action chunk stays within one segment and its final two recorded observations carry the segment-end label. With `ee-full`, chunks can cross placement boundaries and the model predicts motion through the full episode. Evaluate either checkpoint in `ee` mode. SwanLab and `loss.jsonl` use the same optimizer steps. Data checks do not create a run. Checkpoints support inference; exact training resumption would also require the optimizer and sampling states.

## 部署 / Deployment

按下表选择冻结的 ACT 基座和残差加载器。所有发布权重均可从[固定场景评测入口](evaluate.py)选择具名配置。每个新场景开始时重置策略历史。EE 基座预测到单颗段末时仅重置自身的时间聚合，场景和残差历史继续保留。

| 残差模型 / Model | 控制模式 | ACT 基座 | 加载器 / Loader |
| --- | --- | --- | --- |
| Residual-1、2、3 | qpos | ACT-1 80k | [`RGBController`](historical_policy.py) 调用 [`ACTResidualPolicy.from_safetensors`](rgb_residual.py)，按检查点配置选择 `edge_aware` |
| Residual-4 | qpos | ACT-1 100k | [`PlacementController`](historical_policy.py)，包含放置检测与归位控制 |
| Residual-5 | qpos | ACT-1 100k | [`make_policy`](residual.py)，从所选部署配置读取 `arm_fraction` |
| Residual-6、7、8 | EE | ACT-2 80k | [`EEResidual`](residual.py)，对逆运动学求得的电机目标施加修正 |

在已取得访问权限的账号下，将示例所用的模型和执行配置下载到同一目录。

```bash
hf download Nole326/LeIsaac-SO101-PickOrange --include "ACT/ACT-1/100k/*" "ACT/ACT-2/80k/*" "Residual/Residual-5/80k/*" "Residual/Residual-8/56k/*" --local-dir models/pickorange
```

**qpos**　纯 ACT 从检查点读取配置与归一化统计，直接返回六维标定电机目标。

```python
from course.pickorange.act.act_policy import QposPolicy

act = QposPolicy("models/pickorange/ACT/ACT-1/100k", device="cuda")
act.reset_scene()
motor_target = act.predict(front_rgb, wrist_rgb, joint_motor)
```

以下示例复用上面的 `act`，在场景开始时接入 Residual-5。五个臂关节使用四分之一幅残差，夹爪保持满幅。接入后，每次观测只调用 `policy.act`，由它推进基座的时间聚合。每个新场景调用 `policy.reset()`。本仓库的两种残差加载器均从权重旁的 `config.json` 核对 `action_mode`。

```python
import json
from pathlib import Path
from course.pickorange.act.residual import make_policy

weights = Path("models/pickorange/Residual/Residual-5/80k")
profile = json.loads((weights / "profiles/arm-quarter/config.json").read_text(encoding="utf-8"))
policy = make_policy(
    act.model, act.stats, weights / "model.safetensors",
    device=act.device, arm_fraction=profile["arm_fraction"],
)
policy.reset()
motor_target = policy.act(front_rgb, wrist_rgb, joint_motor)
```

**EE**　ACT 输出七维目标：机器人基座系中的工具中心点 XYZ（米）、夹爪连杆在基座系中的旋转向量（三维，弧度）和夹爪角度（弧度）。工具中心点相对 SO-101 的夹爪连杆原点（`wrist_roll` 的子连杆）的偏移为 **(0.042393, −0.001200, −0.093031) 米**，沿连杆局部 XYZ 轴度量。工具坐标系与连杆坐标系同向。[EE 权重与工具参考点](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/blob/main/ACT/ACT-2/80k/README.md)

示例中的 `ee_controller` 由仿真端提供。若控制器支持工具偏移，使用 [`TCP_OFFSET_M`](ee_command.py) 配置工具中心点，然后按下例直接传入 `ee_target`。若控制器求解的是夹爪连杆原点，使用同一模块的 `link_origin_target(ee_target)` 作为 IK 输入。该函数用目标姿态将局部偏移转到基座系，再从目标位置中减去偏移。两种接法均只应用一次偏移。

控制器以当前实测关节弧度为求解起点，将五维手臂关节解和预测的夹爪角度组成六维关节目标。`command` 示例返回关节目标及求解误差。残差接收 IK 输出对应的标定电机目标；执行端应用协议规定的驱动和关节限位。[LeIsaac 单位转换](../../../third_party/leisaac/source/leisaac/leisaac/utils/robot_utils.py)

```python
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)
from course.pickorange.act.ee_policy import EEPolicy
from course.pickorange.act.residual import EEResidual

act = EEPolicy("models/pickorange/ACT/ACT-2/80k", device="cuda")
residual = EEResidual("models/pickorange/Residual/Residual-8/56k", device=act.device, base_control=act.control)
act.reset_scene()
residual.reset_scene()
ee_target, segment_end = act.predict(front_rgb, wrist_rgb, joint_motor)
ik_rad, ik_error = ee_controller.command(ee_target, measured_joint_rad)
ik_motor = convert_leisaac_action_to_lerobot(ik_rad.reshape(1, 6))[0]
motor_target = ik_motor + residual.correction(front_rgb, wrist_rgb, joint_motor, ik_motor)
joint_target_rad = convert_lerobot_action_to_leisaac(motor_target.reshape(1, 6))[0]
```

For EE, the base predicts a TCP position, gripper-link orientation and gripper angle in the units above. Configure the IK tool offset in the child frame of `wrist_roll`, or call `link_origin_target` once for link-origin IK. Pass the calibrated motor targets from IK to the residual. In both modes, keep the control cadence and the checkpoint-specific amplitude profile. The [online SAC guide](../sac/README.md) covers training, replay, resumption and the fixed-seed evaluation profiles.

## 评测 / Evaluation

固定 50 个场景使用种子 731000–731049。所有橙子在盘内且机械臂归位时，记录首次整任务成功的物理步。60 秒和 120 秒分别以 3,600 和 7,200 步为界。报告成功数、已完成场数和百分率。配对消融在同一 seed 下保持重置、夹爪力矩、段末处理和时间聚合一致。原始逐场结果和训练曲线可从[模型日志目录](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/tree/main/logs)下载。

The fixed set uses the same 50 seeds across methods. A successful placement of one orange is a segment event. Whole-task success requires all three oranges on the plate and the arm back at rest. Record the first success step and score the two time windows separately.
