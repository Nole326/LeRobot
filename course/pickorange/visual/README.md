# PickOrange 视觉规则基线 / Calibrated visual rules

运行入口为 [evaluate.py](evaluate.py)。两种协议都通过 RGB 检测、机器人几何 IK 和实测关节反馈执行抓放。准备仿真环境、机器人与场景资产即可运行。物体真值和成功标记由评测器记录，控制器只接收各自协议声明的输入。

模型辅助版通过 `--sim-camera-calibration` 读取相机内参和相机到机器人基座的变换。位置伺服使用机器人关节的重力补偿力除以 17.8 作为前馈。以基座坐标系计，桌面默认 Z 为 0.030 m，橙子中心和盘面定位平面分别高出桌面 0.026 m 和 0.012 m。

The model-assisted actor receives front RGB, six measured joint angles, explicit camera calibration, support and size priors, and robot-model gravity bias. The runner evaluates success with the native scoring function. During closure, lift, carry and lowering, the final jaw target can request at most 0.020 rad of closure from the latest measured angle. The bound is applied after rate limiting.

策略复用 `expert/` 中的 `DiameterKinematics` 和 `FingerGeometry`。规划在图像估计的位置枚举夹持方向，用手指网格检查与桌面的间隙，并根据关节停滞推测接触。闭爪、抬升、搬运和下放阶段，最终夹爪指令最多比当前实测角度再闭合 0.020 rad。该限制在指令限速后应用。每次放置后重新扫描盘外橙子，最多尝试抓取五次。

## 仅视觉＋本体＋相机标定 / RGB, q and camera calibration only

添加 `--input-contract rgb-q-calibration` 启用独立输入边界，使用 [strict_policy.py](strict_policy.py) 和 [stereo.py](stereo.py)。控制器的动态输入仅有 front RGB、wrist RGB 和实测 q6；已知参数为两相机内参、front-to-base、wrist-to-tool 外参及机器人运动学／手指几何。机器人几何属于固定控制器标定，不读取动力学或物体状态。

此分支不调用重力补偿力接口，不输入桌面高度、橙子尺寸、盘子半径或深度图。先移动到固定高位扫描姿态，再由两路 RGB 的 SIFT 匹配、标定投影三角化和 RANSAC 拟合支撑平面。腕相机逐帧位姿由**关节角＋固定安装外参＋机器人 FK**推导，不读取逐帧相机或工具真值。HSV轮廓在视觉平面上推算尺寸；橙子采用球形接触近似，盘内区域和投放点按图像估计半径定义。这些是算法假设，不是给定的物体尺寸。控制使用关节误差积分修正，未改变原生执行器／物理。

两种协议都需要显式启用 `--sim-camera-calibration`。每次重置，前视相机从名义安装位姿施加本回合扰动。评测器在回合物理步为 0 时刷新双相机渲染与观测缓冲，再读取首帧及对应标定。两种协议都记录前视外参，双相机版同时记录腕部安装外参。

The RGB, q and camera calibration variant reconstructs the support plane and estimates object size from the two cameras. Both variants require `--sim-camera-calibration`. At reset, each perturbation is applied to the nominal front-camera pose. The runner refreshes both cameras and their observation buffers before reading the first RGB frames and calibration, with zero episode physics steps elapsed.

```bash
python course/pickorange/visual/evaluate.py --check --input-contract rgb-q-calibration
python course/pickorange/visual/evaluate.py \
  --run-simulation --sim-camera-calibration \
  --input-contract rgb-q-calibration \
  --headless --enable_cameras --device cuda:0 \
  --out /outputs/pickorange-visual/strict-new-run \
  --horizon 55 --seeds 1001 1002 1003
```

55 秒协议名为 `visual-rgb-q-calibration-fixed-55s-v2`。支撑平面缺少足够视觉匹配时记录失败。两种协议共用原生成功判定。

## 运行 / Run

工作目录为仓库根。`--help` 和 `--check` 只需要标准库，不加载模拟器、Torch 或 SwanLab：

```bash
python course/pickorange/visual/evaluate.py --check
python course/pickorange/visual/evaluate.py --help
```

闭环评测使用独立 PickOrange 仿真环境、官方机器人与厨房资产、NumPy、SciPy、OpenCV 和 `swanlab==0.10.3`。保存调试图还需要 Pillow。环境与资源来源见[任务指南](../../../docs/PICK_ORANGE.md)和[外部资产清单](../../../manifests/external_assets.json)。

```bash
python course/pickorange/visual/evaluate.py \
  --run-simulation --sim-camera-calibration \
  --headless --enable_cameras --device cuda:0 \
  --out /outputs/pickorange-visual/new-run \
  --horizon 55 --seeds 1001 1002 1003
```

输出目录必须尚不存在；容器以 root 运行时可额外传 `--kit_args=--allow-root`。默认 30 Hz 更新关节绝对位置目标，两 tick 保持，物理 60 Hz。动作顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper，单位 rad；依据实际 action manager 的 scale/offset 转为原生动作，不把 motor 归一化值当弧度。

模型辅助版默认使用 `visual-calibrated-native-fixed-55s-v2`，开发预算为 55 秒。使用 `--horizon 25` 可运行原生时限下的独立协议。两种输入协议都使用原生重力、执行器增益、固定夹爪力矩上限、碰撞、关节限位和成功判定，并关闭读取物体位置和质量的动态夹爪辅助。

默认保存数值记录。需要检查画面时，显式添加 `--save-debug-images` 保存双相机 RGB 与检测图。 / Runs store numerical records by default. Pass `--save-debug-images` to save camera frames and detection overlays.

## 结果与审计 / Results and audit

`manifest.json` 记录协议、seed、运行环境和执行器配置。`seed-*/commands.npz` 记录请求关节角、实测关节角、时间和控制阶段。`trace.jsonl` 的 `scoring_only` 字段记录评分真值。评分使用 `RecorderTerm.record_pre_reset` 捕获的原生终态。整批完成后，`result.json` 的 `status` 为 `completed`。SwanLab 以离线模式记录数值指标。

初版使用 20 个固定场景（seed 1001–1020），修订版按原顺序复测至表中所列场次。每回合上限 55 仿真秒。成功要求三颗橙子均在原生 XYZ 盘区内且机械臂归位。成功率以完整回合数为分母。

The initial evaluation used 20 scenes (seeds 1001–1020). The retest reports the completed prefix of those seeds. Each episode runs for up to 55 simulated seconds. Success requires all three oranges inside the native XYZ region and the arm at home.

| 输入协议 / Input protocol | 初版全量 / Full initial run | 初版同场景 / Initial on matched seeds | 修订版 / Retest | 复测成功耗时中位数 / Retest median successful duration |
| --- | ---: | ---: | ---: | ---: |
| 模型辅助 / Model-assisted | 16/20（80.0%） | 15/19（78.9%） | 16/19（84.2%） | 40.50 s |
| 双 RGB＋q6＋相机标定 / RGB, q6 and camera calibration | 15/20（75.0%） | 14/17（82.4%） | 11/17（64.7%） | 42.63 s |

表中分数适用于 55 秒仿真协议。25 秒协议和真机表现待验证。

These scores apply to the 55-second simulation protocols. The 25-second protocols and hardware performance remain unvalidated.

运行软件接口与合成 RGB 几何的 CPU 检查。

```bash
python -B -m unittest discover -s tests -p test_pickorange_visual.py -v
```

HSV 检测对光照和纹理变化敏感。平面投影用于桌面上的物体，局部遮挡会使椭圆中心偏移。抓取中的接触由关节反馈推测。
