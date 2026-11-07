# PickOrange 特权state控制脚本

本目录提供特权state控制脚本（简称控制脚本）、动作轨迹独立回放检验（简称回放检验）和脚本专家演示采集工具。

回放检验是指：在独立启动的仿真环境中恢复初始场景，按记录的动作轨迹执行，验证用于BC的动作能否完成任务。

任务是在 **55 仿真秒内**将三颗橙子放入盘中，并使机械臂回到休息姿态。控制脚本执行和回放检验均满足原生成功条件后，轨迹才可作为脚本专家演示，用于行为克隆（BC）训练。

已在实际仿真环境中完成控制执行、动作轨迹独立回放检验与采集流程的联合验证。采集工具不加载 VLA 权重，也不自动启动训练。

## 代码结构

| 文件 | 用途 |
| --- | --- |
| [控制脚本入口](privileged_teacher.py) | 负责抓取、下降恢复、抬升验证、放置和松手检查 |
| 运动规划与控制组件 | 通过继承组合使用，由控制脚本入口调用 |
| [finger_geometry.py](finger_geometry.py)、[grasp_quality.py](grasp_quality.py) | 夹爪碰撞几何与抓取质量代理计算 |
| [joint_contract.py](joint_contract.py) | 六关节顺序及 rad/motor 坐标映射 |
| [diameter_geometry.json](diameter_geometry.json)、[finger_geometry_calibration.json](finger_geometry_calibration.json)、[object_center_calibration.json](object_center_calibration.json) | 抓取几何标定参数 |
| [verify_scenes.py](verify_scenes.py) | 执行控制脚本并记录动作、初态、终态和诊断 |
| [replay_success.py](replay_success.py) | 在独立仿真进程中重放动作，采集同步图像与关节状态 |
| [collect.py](collect.py) | 在指定尝试预算内顺序采集并筛选合格轨迹 |
| [rgb_video.py](rgb_video.py) | 无损 RGB 编码及解码一致性检查 |

## 控制与输入边界

环境为 `LeIsaac-SO101-PickOrange-v0`，使用单环境、机器人重力和原生关节执行器。控制脚本通过末端几何规划及逆运动学求解生成六维绝对关节目标，经正常动作接口执行。运行中不直接写入关节或物体状态；回放检验初始化时使用 `reset_to` 恢复记录的初态。

物理频率为 60 Hz，动作频率为 30 Hz，每条动作保持两个物理步。回合上限为 3300 个物理步。控制器保留原生软关节范围，并限制相邻目标变化量：通常为 0.03 rad，最终撤离及归位阶段为 0.04 rad。局部规划异常请求恢复，不单独作为轨迹拒收条件。

控制脚本可读取橙子的位姿、速度及几何尺寸，盘的位姿与表面几何，以及机器人和夹爪的位姿、关节状态与开度。控制脚本和回放检验每个物理步均调用上游 `dynamic_reset_gripper_effort_limit_sim`，该助手读取物体位置和质量。因此完整控制栈属于特权辅助，不能作为无特权视觉策略的成绩。

BC 输入仅包含 front/wrist 两路 RGB 和六维实测关节位置。物体真值、诊断、奖励与成功标记均不得进入 actor 输入。

关节顺序为 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。`motor` 是标定归一化坐标，不是角度：五个机械臂关节映射到 ±100，夹爪映射到 0–100。输入 `joints` 与监督标签 `action` 使用相同坐标，标签表示实际发送的目标，不能用下一帧实测位置替代。

## 运行前提与命令

按[环境指南](../../../docs/PICK_ORANGE.md)准备 PickOrange 环境与资产，使用对应仿真环境的 Python。依赖包括 NumPy、SciPy、Pillow、Torch、Isaac Lab/Sim、冻结的 LeIsaac，以及支持 `libx264rgb` 的 FFmpeg。

运行前应确认任务环境、机器人资产与标定相匹配。资产路径由环境配置提供。FFmpeg 从 PATH 查找，也可通过 `PICKORANGE_FFMPEG` 指定可执行文件。

在仓库根目录执行 CPU 检查：

```bash
python course/pickorange/expert/verify_scenes.py --check
python course/pickorange/expert/replay_success.py --check
python course/pickorange/expert/collect.py --check
python course/pickorange/expert/check_controller.py
python course/pickorange/expert/check_geometry.py
```

仿真需显式指定 `--run-simulation`，并只暴露一张已分配的 GPU。以下随机种子仅为命令示例；实际评估应明确记录使用的种子。所有输出目录必须不存在，不覆盖已有结果。

```bash
python course/pickorange/expert/verify_scenes.py --run-simulation \
  --out /outputs/control-script --seeds 0 1 2 \
  --headless --enable_cameras --device cuda:0

python course/pickorange/expert/replay_success.py --run-simulation \
  --source /outputs/control-script --out /outputs/replay --seeds 0 1 2 \
  --headless --enable_cameras --device cuda:0

# 多场景控制脚本验证与回放检验通过后，再进行批量采集。
python course/pickorange/expert/collect.py --run-simulation \
  --out /outputs/demonstrations --start-seed 0 --count 100 --max-attempts 400
```

直接调用回放检验入口要求源目录中的控制脚本场景全部成功。批量采集按单个种子分别启动控制脚本与回放检验进程，控制脚本失败时跳过回放检验。达到尝试上限仍未采满时，保留已完成数据与失败记录并报告失败。当前不支持中断恢复，不自动启动 BC。检查依赖断言，请勿使用 Python `-O`。

## 数据格式与准入

每条脚本专家演示包含 `front.mkv`、`wrist.mkv`、`demonstration.npz` 和 `result.json`。视频为 640×480、RGB24、30 Hz 无损编码。NPZ 包含 `joints[N,6]`、`action[N,6]` 和 `timestamp[N]`。

图像与实测关节位置在发送动作前同步记录，取自通过回放检验的执行过程中。每帧观测对应此时实际执行的动作，不混用控制脚本执行和回放检验过程的观测。动作标签耗尽后保持最后目标，直到成功或达到回合上限。

原生成功要求三颗橙子同时满足盘区条件且机械臂归位，终态在自动重置前捕获。图像、关节状态与动作必须完整且逐帧对应。关节跟踪误差作为诊断记录，不以固定误差阈值或轨迹等长作为额外准入条件。

`initial-state.pt`、`numeric-actions.npz`、逐步诊断和事件记录用于回放检验与审核，与 actor 数据分离。仅加载可信来源的初态文件。合格演示的同步图像须为 BC 保留；验证临时图像和失败视频分析后可以清理。采集器默认保留失败证据，应预留存储空间并限制尝试次数。

每个场景输出 `scalars.jsonl`，可接入数值指标同步。工具不包含账号或认证配置，不上传图像。演示、模型、运行日志和认证文件不应提交 Git。

机器人和场景资产需另行获取，上游代码的许可见[第三方说明](../../../THIRD_PARTY_NOTICES.md)。
