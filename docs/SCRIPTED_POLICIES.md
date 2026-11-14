# 两个任务的规则脚本与特权state控制脚本 / Rules and state control scripts

视觉规则策略用于方法baseline；特权state控制脚本读取物体真值，供数据生成与诊断。二者分别标注输入、控制方式和成绩。下表链接仓库内的实际源码，上游文件保留原目录、固定版本和许可证，不再复制一份同名脚本。

Visual rules and privileged state control scripts serve different purposes. Links below point to checked-in code. Upstream files remain in their pinned directories with their licenses intact.

| 任务与类型 / Task and type | 源码 / Source | 运行与状态 / Usage and status |
| --- | --- | --- |
| LiftCube视觉规则 | [RGB定位与状态机](../course/liftcube/src/starter/scripted_liftcube.py)、[几何抓取](../course/liftcube/src/starter/scripted_geometry.py)、[关节链](../course/liftcube/src/starter/joint_chain.py) | [标定与运行指南](SCRIPTED_LIFTCUBE.md)。已有100回合45次成功的独立标定辅助协议；不是免标定专家 |
| LiftCube 特权state控制脚本 | [PrivilegedLiftCube](../course/liftcube/src/starter/privileged_liftcube.py)、[真值访问边界](../course/liftcube/src/starter/envs/lift_cube_privileged.py) | 共用[evaluate_scripted](../course/liftcube/src/starter/scripts/evaluate_scripted.py)，显式`--privileged-teacher`；CPU测试通过，真实抓取待验 |
| PickOrange 自研控制脚本 | [控制与采集入口](../course/pickorange/expert/README.md) | 原生关节目标、独立回放检验与同步双相机采集；已完成控制执行、回放检验与采集流程的实际仿真联合验证 |
| PickOrange 官方控制脚本 | LeIsaac官方[PickOrange状态机](../third_party/leisaac/source/leisaac/leisaac/datagen/state_machine/pick_orange.py)及[采集入口](../third_party/leisaac/scripts/datagen/state_machine/generate.py) | [运行与物理限制](../course/pickorange/README.md#特权state控制脚本采集--privileged-state-control-script-collection)。已收录；生成器改变物理并直接写关节状态，数据须独立验收 |
| PickOrange 视觉规则 | [RGB 检测、几何 IK 与状态机](../course/pickorange/visual/README.md) | 提供模型辅助版及双 RGB＋q6＋相机标定版。双相机版由 RGB 恢复支撑平面和物体尺寸。55 秒与 25 秒协议分别报告。 |
| 上游抬升状态机参考 | Isaac Lab [lift_cube_sm.py](../third_party/isaaclab/scripts/environments/state_machine/lift_cube_sm.py) | 基于Franka与特权位姿，不可直接用于SO101；用于研究状态机分阶段设计 |

## LiftCube

在配套环境中，从`course/liftcube`运行，设置`PYTHONPATH=$PWD/src:/leisaac/source/leisaac`。视觉脚本与特权state控制脚本使用同一个评测器，但标定要求不同：

```bash
# 视觉策略：仅检查外部标定与IK，不启动模拟器
python -m starter.scripts.evaluate_scripted \
  --calibration /calibration/liftcube-visual.json --check

# 特权state控制脚本：使用固定姿态标定，不启用grasp_presets或plan_while_opening
python -m starter.scripts.evaluate_scripted --privileged-teacher \
  --calibration /calibration/liftcube-control.json --check
```

真实运行需要显式`--run-simulation`、匹配配置和新输出目录，见[视觉指南](SCRIPTED_LIFTCUBE.md)及[控制脚本步骤](TRAIN_AND_EVALUATE.md#2-liftcube特权state控制脚本--liftcube-privileged-state-control-script)。模板不是有效标定；普通视觉输入不读取物体位置。显式运行时相机标定辅助应单独报告。

Both variants reuse the same evaluator. Calibration checks are CPU-only; actual simulation requires explicit execution and a fresh output directory. Runtime camera-calibration assistance must be reported separately.

## PickOrange

本项目的 [特权state控制脚本](../course/pickorange/expert/README.md)通过原生关节动作执行，支持动作轨迹独立回放检验准入和同步 BC 图像与本体状态采集。已在实际仿真环境中完成控制执行、回放检验与采集流程的联合验证。其控制方式与下述官方生成器不同，使用时应分别核对物理条件和数据准入要求。

从[任务指南](PICK_ORANGE.md)准备独立环境，在仓库根目录使用官方采集入口：

```bash
python third_party/leisaac/scripts/datagen/state_machine/generate.py \
  --task LeIsaac-SO101-PickOrange-v0 --num_envs 1 --seed 17 \
  --device cuda:0 --headless --enable_cameras --record \
  --dataset_file /datasets/pickorange/script-demonstrations.hdf5 --num_demos 1
```

这条命令会运行GPU仿真。`--num_demos`限制成功条数而非失败尝试总数，运行平台还需设置墙钟上限。官方生成器关闭机器人重力、改变阻尼／夹爪与终止逻辑，并在部分阶段直接写关节状态；其输出不能直接作为固定物理下的视觉baseline成绩或合格BC标签。采集前读[完整限制](../course/pickorange/README.md#特权state控制脚本采集--privileged-state-control-script-collection)，训练前按[数据准入](TRAIN_AND_EVALUATE.md#3-数据准入--data-admission)验证原始请求、物理重放和终态。

This runs GPU simulation. Bound wall time independently: the demo count limits successes, not retries. The generator changes physics and writes joint states directly, so its success labels require independent physical replay before BC admission.

## 来源与复用 / Provenance

- SO101视觉与state包装由本项目维护；控制保护和评测入口共用，定位输入分开。
- PickOrange状态机来自[LeIsaac](https://github.com/LightwheelAI/leisaac)固定commit `24d3bcd3f1e4585740fc79921782c41617237812`，Apache-2.0。
- Franka抬升参考来自[Isaac Lab](https://github.com/isaac-sim/IsaacLab)固定commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`，BSD-3-Clause。
- 许可证与附属声明见[第三方说明](../THIRD_PARTY_NOTICES.md)。演示、模型与场景资产按各自许可另行获取，不随脚本打包。

The repository preserves upstream notices and pinned source identities. A script's license does not replace the terms for datasets, weights or simulator assets.
