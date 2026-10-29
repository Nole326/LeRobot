# 受限末端开发接口 / Planar EE development interface

这是自研执行层，不是课程正式IK接口。入口为`starter.envs.factory.make_env`，实现位于`starter.envs.planar_ee`。原LiftCube六关节模式仍是默认值；已有抓取脚本、演示标签和历史成绩不变。

This opt-in execution layer is a development controller, not the course-provided IK API. The existing joint mode remains the default; old grasp scores and demonstration labels retain their original contract.

当前只完成执行链路短测，**物理定高精度尚未通过**：见下方验证范围。不要将它作为已验收的抓取控制器或真机安全层。

The execution prefix runs, but physical height tracking is not accepted. This is not a validated grasp controller or hardware safety layer.

## 动作与约束 / Contract

| 内容 | 定义 |
| --- | --- |
| 策略动作 | `[tcp_x, tcp_y, gripper.pos]`，形状严格为`(3,)` |
| 位置 | 桌面参考坐标系中的**绝对XY，单位米**；不是增量或归一化坐标 |
| 夹爪 | 绝对`motor_0_100`目标，不是弧度或力 |
| 固定参考 | 标定给定的TCP高度及工具接近轴；允许绕该轴转动，**不是完整三轴姿态固定** |
| 频率 | 策略30Hz；原60Hz环境每次`step`仍只推进一个物理tick，控制器求解一次后保持两tick |
| 保护 | 原关节范围、每决策≤1motor、领先实测臂关节≤8motor；名义TCP单步≤4mm；不可达/无效请求不裁剪 |
| 求解容差 | 名义TCP位置≤3mm、接近轴夹角≤0.15rad；不是实际机械臂无误差的高度/姿态锁定 |
| 可选跟踪补偿 | 标定文件显式`tracking_compensation: true`；静止参考连续三次后积分q误差，臂关节偏置限制±3motor，夹爪不补偿；默认关闭 |

Only measured joint positions and fixed robot/frame calibration enter the controller. The action is absolute XY in metres plus an absolute jaw motor target. Height and approach-axis references are fixed; axial rotation remains free. The five-axis robot is not promised arbitrary XY reachability. Infeasible requests are rejected, never repaired by relaxing limits or silently changing axes.

IK/跟踪失败锁定保持最后安全命令，须显式reset才能解除；保持命令不是硬件急停。本接口没有真机安全认证。实测TCP由实际q做FK得到，单独记录高度/接近轴误差；模型可达和名义约束通过，不等于物理跟踪、碰撞或抓取通过。

坏维度、非有限或超范围动作在执行前拒绝。`info.command`仍是原生关节执行记录；新增`info.ee_command`分别记录末端请求、名义关节、发送关节、名义/实测TCP及失败原因。观测中的六维上一关节命令不改名为末端命令，也不填造补零的EE反馈。

启用补偿时，平面约束作用于`reference_motor/reference_tcp_table`，发送值还包含单独记录的`steady_bias_motor`。此时`commanded_tcp_table`是发送目标的FK，可能偏离名义平面；不能把补偿目标算作实际TCP或隐去这层差异。发送范围、≤1motor限速与≤8motor反馈领先保护仍生效，超出拒绝，不裁剪策略请求。

`ee_command.measured_*`对应求解该次决策前的反馈，在保持tick重复；每次原生step之后的实测位置见`command.measured_motor`。两者不能当作同一时刻的测量。

## 标定与接入 / Calibration and use

将以下JSON保存到仓库外，并用独立核验结果填齐。`null`是未标定占位，不能用于运行。关节链获取见[导出及标定](SCRIPTED_LIFTCUBE.md)。桌面→机器人基座变换、TCP及接近轴必须属于同一模型/坐标定义；这里只读这份固定文件，不查询运行中的相机/物体位姿。

```json
{
  "schema": "so101-planar-ee-dev-v1",
  "evidence": "填写独立标定和机器人资产版本的引用",
  "chain_path": "so101-chain.json",
  "chain_sha256": null,
  "joint_names": null,
  "tcp_offset_m": null,
  "table_to_base": null,
  "xy_bounds": null,
  "fixed_height_m": null,
  "tool_axis_local": null,
  "tool_axis_table": null,
  "tracking_compensation": false
}
```

`xy_bounds`为`[minimum_xy, maximum_xy]`；两个工具轴均为三维单位向量，分别表达在工具frame与桌面frame。`chain_path`相对该JSON所在目录解析。reset时机器人必须已经处于已标定的工作区、高度与接近方向，接口不会盲目移动到起点。

在原环境配置的`env`下显式设置：

```yaml
control_mode: planar_ee_dev
planar_ee_config: /approved/calibration/planar-ee.json
```

然后照常调用`make_env(cfg)`。新策略须声明三维末端动作和30Hz决策频率，共同`run_episode`仍按原生环境时钟保持动作；直接调用环境时，每个决策的同一末端请求须连续送入两次60Hz `step`，遇到终止立即退出。wrapper不会在一次`step`里额外推进仿真。

旧`starter.scripts.evaluate_scripted`输出六关节目标，明确拒绝该模式。不能将其输出做FK后改名为“末端策略”，也不能把六维ACT/VLA权重或关节演示直接接入三维接口。完整抓放还需任务侧高度/阶段接口及视觉目标选择；当前固定高度执行层不声称已接通它们。

Use the factory with `control_mode: planar_ee_dev` and an external calibrated JSON. A compatible policy must emit three EE components at 30Hz. Existing joint policies/datasets are incompatible and are not automatically converted. Reset requires an already valid starting pose. Full pick-and-place and any official height/phase interface remain separate work.

## 软件验证 / Software validation

仓库根目录，NumPy/SciPy环境，不启动仿真或硬件：

```bash
python -m unittest discover -s tests -p test_planar_ee.py -v
```

覆盖单位/范围、不可达拒绝、跟踪保持、30/60Hz单次保持、终止、重置失败不复用旧命令及工厂模式隔离。测试替身不证明真实物理效果；使用匹配资产前还需核对关节链身份、校准坐标与真实短闭环。正式接口发布后按其字段、轴、约束、周期与安全语义重新接入，不把本开发约定当作课程要求。

开发验证（2026-10-29）：首版13项新增CPU测试、匹配机器人关节链的12项理想反馈运动学检查通过。未补偿首版在真实环境完成16个物理tick／8次决策，命令保持正确、无执行裁剪；名义高度最大偏差约0.0066mm，但每步实测q的FK高度最大偏差26.12mm、XY跟踪误差最大5.86mm。因此只验收链路与记录，**不验收物理平面保持、完整回合或抓取效果**。这段测试使用基座对齐的合成参考平面，不是已验证的桌面标定，也未验证任意初态或长时控制。

针对该偏差增加的可选补偿版已通过15项CPU测试（连同旧相关模块共48项），包含合成负载误差消除、补偿有界、失败保持实际发送值和reset清零。其独立240tick保持检查已准备，但尚未执行，不声称已改善物理误差，也未覆盖首版证据。

The uncompensated version passed a 16-tick live prefix, with post-step joint-derived FK errors up to 26.12mm in height and 5.86mm in XY. Physical tracking is not accepted. The opt-in compensation revision passes 15 controller CPU tests, but has no live result yet. The synthetic base-aligned reference plane is not a calibrated tabletop; no full episode or grasp score is claimed.

## 相机标定的边界 / Camera calibration boundary

PDF允许课程组提供的内参、外参及标准读取接口。当前规则脚本的可选`--runtime-camera-calibration`则直接读取仿真相机/基座真实位姿，每次reset更新外参；这不是图像估计，也尚未证明是课程公开给策略的标定接口。该辅助输入与物体真值输入不同，但仍必须披露并确认许可。不得把其成绩改写成仅RGB/q或已完成真机对齐的成绩。

The rule runner's optional per-reset simulator-pose calibration is an auxiliary input, not image-based calibration. Its availability under the official student interface is unconfirmed. This planar controller itself only consumes the fixed robot/table calibration file and measured joints; it does not enable that camera provider.
