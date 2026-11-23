# PickOrange 双相机 ACT 与残差策略

[模型卡与固定场景成绩](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange) · [原始演示](https://huggingface.co/datasets/Nole326/LeIsaac-SO101-PickOrange) · [仿真与规则基线](../README.md)

部署观测为前视和腕部两路 640×480 的 RGB 图像（HWC、uint8）及六维实测关节位置（SO-101 标定电机坐标）。物理仿真为 60 Hz。策略每两步接收一次新观测，预测 15 步动作块，按系数 0.001 做时间聚合，当前目标保持两个物理步。

| 控制模式 | ACT 学习目标 | 残差作用位置 |
| --- | --- | --- |
| qpos | 五个臂关节和夹爪的绝对电机目标 | 在 ACT 关节目标上修正 |
| EE | 基座坐标系下的绝对目标位姿、夹爪角度和段末信号 | 先由仿真控制器求逆运动学，再在标定电机坐标中修正 |

## 训练 / Training

在仓库根目录运行以下命令。安装仓库使用的 [LeRobot 与 PyTorch 依赖](../../../docs/DEPENDENCIES.md)及 PyAV。下载数据集对应的整个子目录，保留清单和每条轨迹的 NPZ、前视和腕部 MKV。数据检查会解码双相机并核对记录数，训练在内存中缓存无损压缩的 RGB 帧。

```bash
hf download Nole326/LeIsaac-SO101-PickOrange --repo-type dataset --include "vla-gr00t-qpos/*" "vla-gr00t-ee/*" --local-dir datasets/pickorange
python -m course.pickorange.act.train qpos datasets/pickorange/vla-gr00t-qpos/dataset.json --check-data
python -m course.pickorange.act.train ee datasets/pickorange/vla-gr00t-ee/dataset.json --check-data
python -m course.pickorange.act.train qpos datasets/pickorange/vla-gr00t-qpos/dataset.json --output outputs/act-qpos
python -m course.pickorange.act.train ee datasets/pickorange/vla-gr00t-ee/dataset.json --output outputs/act-ee
```

qpos 从 60 Hz 的偶数记录取观测，以其后第 2、4、…、30 个物理步的动作作为监督。EE 直接使用 30 Hz 记录，从当前记录开始预测，动作块在单颗片段末尾截断并掩码。训练用位置、旋转矩阵前两列和夹爪角度表达运动；另一个输出预测段末。每条整轨包含三个单颗片段，最后两条记录的段末标签为正。

默认有效 batch 为 64，每个 microbatch 为 8，训练 100k 次优化器更新。主网络和视觉骨干学习率分别为 0.0001 和 0.00001。训练输出包括损失日志，以及含模型配置、权重、归一化统计和执行配置的推理检查点。指定全新的输出目录；可用 `--updates 1 --batch-size 2 --microbatch 1` 验证一次更新及检查点导出。

Training uses the original camera frames, calibrated joint positions and aligned action labels. For qpos, sample every second observation and supervise the next 15 stride-two targets. EE supervision stays inside one single-orange segment. The trainer writes deployable checkpoints and a scalar loss log.

## 部署 / Deployment

在仓库根目录启动 Python。下载相互匹配的基座与残差目录。每个仿真场景开始时重置策略历史。EE 模型在场内预测到单颗段末时仅重置基座，保留场景和残差历史。

在已取得访问权限的账号下，将示例所用的模型和执行配置下载到同一目录。

```bash
hf download Nole326/LeIsaac-SO101-PickOrange --include "ACT/ACT-1/100k/*" "ACT/ACT-2/80k/*" "Residual/Residual-5/80k/*" "Residual/Residual-8/56k/*" --local-dir downloads/pickorange
```

**qpos**　纯 ACT 从检查点读取配置与归一化统计，直接返回六维标定电机目标。

```python
from course.pickorange.act.act_policy import QposPolicy

act = QposPolicy("downloads/pickorange/ACT/ACT-1/100k", device="cuda")
act.reset_scene()
motor_target = act.predict(front_rgb, wrist_rgb, joint_motor)
```

以下示例在同一关节 ACT 基座上部署四分之一臂幅度的残差策略。

```python
import json
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file
from course.pickorange.act.residual import make_policy

base = Path("downloads/pickorange/ACT/ACT-1/100k")
weights = Path("downloads/pickorange/Residual/Residual-5/80k")
config = PreTrainedConfig.from_pretrained(base)
config.pretrained_backbone_weights = None
act = ACTPolicy(config)
act.load_state_dict(load_file(str(base / "model.safetensors")), strict=True)
stats = json.loads((base / "normalization.json").read_text())
profile = json.loads((weights / "profiles/arm-quarter/config.json").read_text())
policy = make_policy(
    act, stats, weights / "model.safetensors", device="cuda", arm_fraction=profile["arm_fraction"]
)
policy.reset()
motor_target = policy.act(front_rgb, wrist_rgb, joint_motor)
```

**EE**　位置单位为米，旋转向量与夹爪角度单位为弧度。示例中的 `ee_controller` 是仿真端已初始化的 SO-101 逆运动学控制器。它接收实测关节弧度和 EE 目标，返回关节弧度及求解误差。弧度与标定电机坐标之间使用 [LeIsaac 的转换函数](../../../third_party/leisaac/source/leisaac/leisaac/utils/robot_utils.py)。执行端仍应用该评测协议规定的驱动和关节限位。

```python
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)
from course.pickorange.act.ee_policy import EEPolicy
from course.pickorange.act.residual import EEResidual

act = EEPolicy("downloads/pickorange/ACT/ACT-2/80k", device="cuda")
residual = EEResidual("downloads/pickorange/Residual/Residual-8/56k", device="cuda")
act.reset_scene()
residual.reset_scene()
ee_target, segment_end = act.predict(front_rgb, wrist_rgb, joint_motor)
ik_rad, ik_error = ee_controller.command(ee_target, measured_joint_rad)
ik_motor = convert_leisaac_action_to_lerobot(ik_rad.reshape(1, 6))[0]
motor_target = ik_motor + residual.correction(front_rgb, wrist_rgb, joint_motor, ik_motor)
joint_target_rad = convert_lerobot_action_to_leisaac(motor_target.reshape(1, 6))[0]
```

For EE, the base emits a robot-base-frame pose plus a gripper target. Apply IK first and pass calibrated motor targets to the residual. In both modes, keep the control cadence and the checkpoint-specific amplitude profile. The residual module implements deployment; the online SAC trainer and replay service are separate components.

## 评测 / Evaluation

固定 50 个场景使用种子 731000–731049。所有橙子在盘内且机械臂归位时，记录首次整任务成功的物理步。60 秒和 120 秒分别以 3,600 和 7,200 步为界。报告成功数、已完成场数和百分率。配对消融在同一 seed 下保持重置、夹爪力矩、段末处理和时间聚合一致。原始逐场结果和训练曲线可从[模型日志目录](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/tree/main/logs)下载。

The fixed set uses the same 50 seeds across methods. A successful placement of one orange is a segment event. Whole-task success requires all three oranges on the plate and the arm back at rest. Record the first success step and score the two time windows separately.
