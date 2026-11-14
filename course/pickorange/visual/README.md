# PickOrange 视觉规则基线 / Calibrated visual rules

运行入口为 [evaluate.py](evaluate.py)，无需演示或模型权重。使用 front RGB 的 HSV/椭圆检测、已知桌面与尺寸先验、机器人 USD 几何 IK、实测关节反馈及抓放状态机。控制器没有环境句柄，不能读取物体真值或成功标记。

**这是显式标定辅助协议。** 当前 runner 必须启用 `--sim-camera-calibration`，读取实际相机和机器人 base 的变换与相机内参；控制还使用机器人 articulation 的重力补偿力／17.8 作为位置伺服前馈。它不是免标定 RGB/q-only 策略。桌面默认 base Z=0.030 m，橙子中心高于桌面 0.026 m，盘面定位平面高于桌面 0.012 m；这些是显式静态先验，不随物体真值更新。

The actor receives RGB, q6, explicit camera calibration, support/size priors and robot-model gravity bias. Object poses, segmentation, raycasts, object masses and native score are excluded. Simulator scoring is isolated in the runner.

策略复用 `expert/` 中 **DiameterKinematics / FingerGeometry** 的机器人几何函数，未构造任何特权 teacher，也不加载包含场景物体位置的标定文件。规划在图像估计的位置枚举夹持方向，检查手指网格与桌面间隙；这一几何代理不是接触传感器，也不保证路径完全无碰撞。夹爪闭合目标最多领先实测值 0.020 rad，接触判断仅由关节停滞推测。放置序列数不代表成功；回到扫描姿态后根据 RGB 检查盘外橙子并允许最多 5 次抓取。

## 仅视觉＋本体＋相机标定 / RGB, q and camera calibration only

添加 `--input-contract rgb-q-calibration` 启用独立输入边界，使用 [strict_policy.py](strict_policy.py) 和 [stereo.py](stereo.py)。控制器的动态输入仅有 front RGB、wrist RGB 和实测 q6；已知参数为两相机内参、front-to-base、wrist-to-tool 外参及机器人运动学／手指几何。机器人几何属于固定控制器标定，不读取动力学或物体状态。

此分支不调用重力补偿力接口，不输入桌面高度、橙子尺寸、盘子半径或深度图。先移动到固定高位扫描姿态，再由两路 RGB 的 SIFT 匹配、标定投影三角化和 RANSAC 拟合支撑平面。腕相机逐帧位姿由**关节角＋固定安装外参＋机器人 FK**推导，不读取逐帧相机或工具真值。HSV轮廓在视觉平面上推算尺寸；橙子采用球形接触近似，盘内区域和投放点按图像估计半径定义。这些是算法假设，不是给定的物体尺寸。控制使用关节误差积分修正，未改变原生执行器／物理。

当前 runner 的 `--sim-camera-calibration` 在每次 reset 只读取相机内参、相机／机器人基座变换，生成已知 front 外参和 wrist 安装外参；这些值单独落盘。相机标定仍是允许且明确声明的辅助，不能称为免标定视觉策略。之后的控制路径不读取机器人模型重力项。

```bash
python course/pickorange/visual/evaluate.py --check --input-contract rgb-q-calibration
python course/pickorange/visual/evaluate.py \
  --run-simulation --sim-camera-calibration \
  --input-contract rgb-q-calibration \
  --headless --enable_cameras --device cuda:0 \
  --out /outputs/pickorange-visual/strict-new-run \
  --horizon 55 --seeds 1001 1002 1003
```

协议名为 `visual-rgb-q-calibration-fixed-55s-v1`，与模型辅助版结果分开；55秒和25秒仍分开报告。开发 seeds 1–3 为2/3。冻结策略后，固定 seeds 1001–1020 的20个完整配对回合为 **15/20（75%）**，Wilson 95%区间约53%–89%；成功耗时中位数42.47秒，范围40.77–43.77秒。与模型辅助版16/20使用相同初始场景：1014由失败变成功，1018、1020由成功变失败；失败1001、1006、1013、1018、1020全部计入分母。配对seed已用于旧版评测，不称为本版全新的未见测试集，评测期间未调参或替换seed。

支持平面缺少足够视觉匹配时记录失败，不回退到固定高度或仿真真值。原生完整成功和 reset 前评分逻辑与模型辅助版共用。审计核对全部20回合的初始场景一致性、源码哈希、输入白名单、视觉平面恢复、有限动作及原生终态；物体真值仅存在于隔离评分记录中。上述75%仅适用于55秒仿真协议，不代表原生25秒或真机成绩。

## 运行 / Run

工作目录为仓库根。`--help` 和 `--check` 只需要标准库，不加载模拟器、Torch 或 SwanLab：

```bash
python course/pickorange/visual/evaluate.py --check
python course/pickorange/visual/evaluate.py --help
```

真实闭环需要独立 PickOrange 仿真环境、官方机器人／厨房资产、NumPy、SciPy、OpenCV、Pillow，以及共享记录器要求的 `swanlab==0.10.3`。使用已配置的 **仿真解释器**，不要在普通 Python 中安装整套模拟器依赖。环境与资源来源见[任务指南](../../../docs/PICK_ORANGE.md)和[外部资产清单](../../../manifests/external_assets.json)。

```bash
python course/pickorange/visual/evaluate.py \
  --run-simulation --sim-camera-calibration \
  --headless --enable_cameras --device cuda:0 \
  --out /outputs/pickorange-visual/new-run \
  --horizon 55 --seeds 1001 1002 1003
```

输出目录必须尚不存在；容器以 root 运行时可额外传 `--kit_args=--allow-root`。默认 30 Hz 更新关节绝对位置目标，两 tick 保持，物理 60 Hz。动作顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper，单位 rad；依据实际 action manager 的 scale/offset 转为原生动作，不把 motor 归一化值当弧度。

协议 `visual-calibrated-native-fixed-55s-v1` 使用 **55 秒开发预算**；原生时限是 25 秒。`--horizon 25` 会产生独立的 `visual-calibrated-native-fixed-25s-v1`，不可混合成绩。保留原生重力、执行器增益／固定 effort、碰撞、关节限位和成功函数，显式禁用会读取物体位置／质量的动态夹爪辅助。成功仍要求 **三颗橙子全部满足原生 XYZ 盘区条件且机器人 home**，不是“完成三次抓放”。当前不提供真机标定／控制适配。

Each run stores its source hashes, robot USD hash, protocol, seeds, camera calibration, native action requests and measured q, stage events, RGB diagnostics and pre-reset terminal scores. The 55 s development budget is separate from the native 25 s protocol. Failed and timed-out episodes remain in the denominator.

## 结果与审计 / Results and audit

`manifest.json` 固定协议、seed、源码／机器人身份和执行器配置；`seed-*/commands.npz` 分开记录请求 rad 与实测 q、时间及阶段；`trace.jsonl` 中 `scoring_only` 真值仅用于诊断。`RecorderTerm.record_pre_reset` 捕获原生终态，缺失会拒绝统计，避免读取 auto-reset 后的新场景。`result.json` 只汇总已完成回合，`status=partial` 不能视作整批结束。共享 SwanLab 默认 offline、仅数值，先关闭 writer 再关闭环境和 SimulationApp。

模型辅助版开发 seeds 1–3 的实际完整回合为 **3/3**，约 38.6–40.5 仿真秒，使用 55 秒协议。该小样本仅证明闭环，不能估计泛化成功率。冻结策略后，独立固定 seeds 1001–1020 的实际整批结果为 **16/20（80%）**；全部按顺序完成，成功回合均满足原生三橙XYZ盘区＋home，4个超时失败仍计入分母。Wilson 95%区间约58%–92%，20回合样本有限；不将失败 seed 删除、替换或用于调参后仍称为 held-out。仅RGB/q6/相机标定版在相同20个初始场景为 **15/20（75%）**，详见前述独立输入协议。两版结果来自 Isaac Sim 5.1 / Isaac Lab 2.3.2 / LeIsaac 0.4.0 和官方资产；25 秒协议和真机均未由这些结果验收。

CPU 检查只验证软件边界与合成 RGB 几何，不代表仿真成功：

```bash
python -B -m unittest discover -s tests -p test_pickorange_visual.py -v
```

主要限制：HSV 对光照／纹理改变敏感；平面投影不能定位空中果实，局部遮挡会使椭圆中心偏移；当前缺少腕部视觉纠偏和真实接触反馈。保留失败回合以评估这些影响。
