# LiftCube 控制脚本 / LiftCube scripted controllers

评测入口是 `starter.scripts.evaluate_scripted`。前视 RGB 视觉控制脚本从图像定位方块；特权state控制脚本从仿真读取方块的位置和朝向。两者都依次完成开爪接近、下降、闭合与抬升，并由环境判定成功。

The evaluator runs two SO-101 controllers. One locates the cube in a calibrated front image; the other reads its position and yaw from the simulator. Both command the same six joints and use native LiftCube scoring.

## 固定场景评测 / Fixed-seed evaluation

两种脚本使用同一组 50 个场景（seed 1000–1049），沿用原生 60 Hz 物理和 30 Hz 控制。每场最多执行 960 个环境步。下表按首次满足环境成功条件的步数统计，8 秒和 16 秒分别包含第 480、960 步。

Both controllers ran on the same 50 seeds at 60 Hz physics and 30 Hz control. The two columns count first successes by steps 480 and 960 in the same episodes.

| 脚本 / Controller | 定位输入 / Localization | 8 s | 16 s |
| --- | --- | ---: | ---: |
| [视觉控制脚本（前视 RGB） / Visual control script](../course/liftcube/configs/scripted/front_rgb.json) | 前视图、关节反馈、重置后相机外参 / Front image, joint feedback and per-reset camera extrinsics | 43/50 (86%) | 47/50 (94%) |
| [特权state控制脚本 / Privileged-state control script](../course/liftcube/configs/scripted/privileged_state.json) | 方块位置与朝向、关节反馈 / Cube position, yaw and joint feedback | 44/50 (88%) | 49/50 (98%) |

在课程仿真环境中，从 `course/liftcube` 运行。先完成两项标定和运动学检查，再用[评测配置](../course/liftcube/configs/scripted/eval.yaml)分别启动。两组臂关节移动速度系数分别为 1.5 和 2。

From `course/liftcube` in the course simulator environment, check both calibrations, then run each controller. The RGB and privileged controllers use arm travel speed multipliers of 1.5 and 2 respectively.

```bash
export PYTHONPATH="$PWD/src:/leisaac/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
/isaac-sim/python.sh -m starter.scripts.evaluate_scripted --calibration configs/scripted/front_rgb.json --check
/isaac-sim/python.sh -m starter.scripts.evaluate_scripted --calibration configs/scripted/privileged_state.json --privileged-state --check
/isaac-sim/python.sh -m starter.scripts.evaluate_scripted --calibration configs/scripted/front_rgb.json --runtime-camera-calibration --config configs/scripted/eval.yaml --output outputs/liftcube-rgb --run-simulation
/isaac-sim/python.sh -m starter.scripts.evaluate_scripted --calibration configs/scripted/privileged_state.json --privileged-state --config configs/scripted/eval.yaml --output outputs/liftcube-state --run-simulation
```

环境在方块中心相对机器人基座的高度超过 0.20 m 时判定成功。每个输出目录中的 `episodes.csv` 记录每场的 seed、成功标志与环境步数。分别统计成功且 `length <= 480` 和 `length <= 960` 的场次，除以完成的场数即可得到两列成功率。`--privileged-teacher` 仍可作为 `--privileged-state` 的别名。

LiftCube scores a success when the cube center rises more than 0.20 m above the robot base. In `episodes.csv`, count successful episodes with `length <= 480` and `length <= 960` and divide each count by the number of completed episodes. The previous flag `--privileged-teacher` remains an alias for `--privileged-state`.

## 抓取与重试 / Grasp and retry

标定文件中的 `parallel_approach: true` 可用于视觉控制脚本和特权state控制脚本。目标定位稳定后，机械臂在开爪过程中向方块上方移动。空抓退让后也可以边张爪边重新接近，下降前检查实际开度。

Set `parallel_approach: true` in the calibration file to enable simultaneous arm motion and jaw opening in either controller. Once the target is stable, the arm approaches the hover pose as the jaw opens. The same behavior applies after retreating from a missed grasp. Descent requires the measured jaw aperture to reach the configured opening.

视觉控制脚本可另设 `fast_loaded_clearance: true`，与 `fast_clearance`、`probe_loaded_close` 配套使用。夹爪反馈显示已夹持、实测末端高于抓取位置 6 mm 后，加快初段抬升。

The visual controller also supports `fast_loaded_clearance: true` with `fast_clearance` and `probe_loaded_close`. Initial lift speed increases once the gripper reports contact and the measured end effector clears the grasp height by 6 mm.

`arm_travel_speed` 调整向悬停位移动、退让和带载上升时的五个臂关节速度，范围 1–2，默认 1。带载上升须先确认夹持且实测末端高于抓取位置 12 mm。关节跟踪误差从 4 增至 7 个电机单位时，提速倍数逐渐回落至 1；下探和夹爪仍按原速度执行。

`arm_travel_speed` scales arm motion during approach, retreat and loaded lift. It ranges from 1 (default) to 2. Faster lifting requires grasp feedback and 12 mm of measured clearance above the grasp height. As joint tracking error rises from 4 to 7 motor units, the multiplier falls back to 1. Descent and jaw motion retain their original rates.

本版本作为LiftCube开发基准保留，便于后续标定与控制接口适配。它不是完整多物体任务或真机部署的验收版本；相机真实位姿辅助模式与固定外参模式分别记录。

This is a LiftCube development reference for subsequent calibration and controller integration, not an accepted multi-object or hardware deployment. Keep fixed-calibration and simulator-pose-assisted results separate.

## 课程基线边界 / Course baseline scope

本实现对应课程的几何/规则对照：视觉检测、预设抓取位姿、简单反馈控制；不替代课程要求的强化学习方法、BC/VLA对照或期末学习策略部署。固定机器人几何、桌面和相机标定是运行先验，不能从测试回合的隐藏物体/目标位置反向拟合，也不能按seed编排关节轨迹。

当前发布的LiftCube起点使用六维关节目标，本脚本在策略侧完成IK并经过共同动作约束层；课程完整多物体任务描述的受限末端动作及课程低层IK接口，仍须在对应环境发布后接入。LiftCube的抬升成功不能算作完整Pick & Place、未见物体泛化或真机验收。

统计成功率前冻结代码、标定、场景seed和预算，调参回合与独立测试分开。指定测试集的全部完整回合均计入分母，失败和超时不重选；资源中断的partial单列，不能冒充完整失败或成功。报告样本数，不把小样本比例当总体性能保证。

This is a geometry/rule comparator, not a substitute for the required RL method or learned hardware policy. It uses the released LiftCube joint interface; the full task's restricted end-effector/course-IK adapter remains a separate integration. Freeze code, calibration, seeds and horizon before independent testing, retain all failures, and report the denominator. Lift success is not full pick-and-place or hardware acceptance.

可单独使用[受限末端开发执行层](PLANAR_EE_DEVELOPMENT.md)，但当前抓取脚本仍输出关节目标，入口拒绝将它误接为三维末端策略；正式课程接口及完整抓放迁移仍待相应发布。An independent planar EE development layer is available; this joint-output grasp runner does not use it or inherit its action contract.

**开发结果：开爪期间定位的实际标定辅助变体在冻结100场景中成功45/100，成功中位耗时9.38仿真秒。** 较早的几何模式原版为13/30（43.3%），固定标定front64方案为1/10，分别保留，不混算。单红方块结果尚不足以证明跨物体能力；当前没有完整碰撞规划，不用于真机直接执行。

**Development result: the calibration-assisted localization-during-opening variant achieved 45/100 on a frozen scene set, with median successful completion time of 9.38 simulation seconds.** Earlier geometric 13/30 and fixed-calibration/front64 1/10 protocols remain separate. These are single-red-cube results, not novel-object generalization or a hardware-ready collision planner.

## 输入与限制 / Inputs and limits

- 只读取单front RGB与本体前六维实际电机位置。默认图像为uint8 HWC 64×64×3；视觉控制脚本可独立配置128或256方形输入，须同步环境resize、标定`image_shape`与内参`K`。这是同一路相机的预处理，不改变相机原始分辨率或其他模型默认值。忽略补零速度、末端字段、metadata和环境info；记录器单独接收执行诊断与原生success。
- 原始实测位置只检查形状与有限性，不当作待发送动作裁剪；接触/求解器可能使反馈略超名义行程。所有发送目标仍须通过原范围与跟踪检查。Raw feedback is preserved; outgoing targets remain bounded.
- 图像检测使用红色通道阈值和连通域，不用分割真值、深度或物体坐标。多个候选、目标缺失或触边时拒绝猜测。像素射线与**预先标定的可见物体平面**相交；该平面不是每回合读取的真值。侧面可见、光照变化、夹爪遮挡会造成偏差或保守失败。
- 整个红色连通域可能同时包含顶面和侧面，不能把它的质心机械地当作顶面中心。平面高度须与检测方式配套，并用独立位置参考核验；不能根据单个测试场景的物体真值回填偏移。
- 运动学可使用固定SO101关节链（NumPy/SciPy），或LeRobot官方`RobotKinematics`/Placo。前者从匹配USD导出固定几何，不读取实时场景；后者按`URDF角度 = USD角度 × joint_sign + joint_offset_deg`转换。夹爪独立使用公共motor单位，不把motor直接当角度。
- 固定抓取方向只适用于其可达区域；SO101五轴不能实现任意六自由度位姿。IK检查位置误差≤3mm、旋转误差≤0.15rad，并拒绝超限解，不靠裁剪修复不可达目标。
- Cartesian参考每次至多移动4mm，朝向沿最短旋转每次至多推进0.02rad；默认每个控制步的关节指令增量最多为 1 个电机单位。启用 `arm_travel_speed` 后，五个臂关节在上方接近、退让和离桌后的带载抬升阶段按跟踪误差调整增量，夹爪仍保持原限速。位置和朝向均到位后才进入下一阶段。这不等于实际关节速度上限或完整路径碰撞保证。接近、下降、闭合或抬升阶段任一臂轴的候选指令与实测相差超过 8 motor 时，撤回该指令并进入恢复。
- 路径从上一发送目标继续推进，不以滞后反馈重新起算参考，避免重力下垂不断拖低接近轨迹；到位判断和跟踪保护仍使用实测关节。This command-referenced path does not replace measured-state arrival and lag checks.
- 可选`gravity_compensation: true`使用`SteadyBiasScriptedLiftCube`：只在接近/下降/抬升的名义目标稳定后按q反馈积分。臂轴偏置同时受±3motor和当前目标距关节边界的余量约束，夹爪不补偿；名义路径与发送目标分开记录。补偿后的指令同样遵守当前阶段的限速、关节范围和 8 个电机单位的跟踪保护。进入恢复时清空补偿，用实测关节位置卸去累积的位置偏差。

Steady-bias correction is bounded by both ±3 motor units and each nominal target's remaining joint travel. This prevents the correction from pushing an otherwise valid target past a joint boundary; nominal targets, command-rate limits and tracking checks retain their existing semantics.
- 起始TCP必须已在标定工作区且不低于hover高度；脚本不会盲目执行回零/抬臂恢复。下降后允许夹爪遮挡；闭合完成请求后停留0.5策略秒，不把夹爪实际无法完全闭合作为必然失败。
- 可选`contact_close: true`将闭合请求限制为至多领先实测夹爪2motor。连续三次位置变化小于0.1motor且反馈领先上一命令至少1.5motor时，保持当前夹爪目标，停留0.5策略秒后抬升；没有停滞则仍可闭合到原目标。`jaw_contact_proxy`只表示位置反馈停滞，也可能来自桌面等障碍，不是力测量或抓取成功；`grasp_motor`单独记录。原动作限速、臂跟踪保护、执行器及评分不变，缺省false保留原闭合行为。
- 接近、下降或闭合受阻后，脚本从实测关节位置退让、张爪并重新定位。RGB 视觉控制脚本启用 `clear_retry_view` 时，抬升途中会检查整张前视图。连续观测到方块留在原处或与夹爪分离时，脚本抬爪让出视野，清除上一次目标，再用稳定的红色轮廓计算抓取位置。单帧遮挡不会触发放手。特权state控制脚本使用方块位置、高度和朝向重新规划。允许重抓直至环境判定成功或超时。

The geometric controller uses front RGB, camera calibration and measured joint positions. With `clear_retry_view` enabled, it searches the whole front image during the lift. Repeated sightings of the cube at the pick location or separated from the fingers trigger a retreat. The controller then clears the old target and localizes the cube again. A single occluded frame does not trigger a release. The privileged controller replans using the cube pose. The environment ends each episode on success or timeout.

A whole-object color centroid can include top and side faces; it is not automatically the top-face center. Validate the detection plane independently instead of fitting offsets to a test scene's hidden object pose. A reachable commanded waypoint also does not prove that the physical arm reaches it under gravity and contact.

前视 RGB 视觉控制脚本与特权state控制脚本都可使用同一姿态选择方法。前者从图像和相机标定计算目标，后者读取方块位置。在标定文件中将 `grasp_presets` 设为空数组、将 `plan_while_opening` 设为 false，再设置 `"adaptive_orientation": true`。脚本先检查原抓取和抬升姿态。当目标接近关节边界或位置误差偏大时，再按方块相对基座的方位选择手腕朝向与倾角。接近和下降共用一个姿态，抬升单独选择。候选姿态需要大于 2 motor 的关节余量且位置误差小于 2.5 mm。如果候选均未达到该条件，原姿态仍须通过常规 IK 检查。缺省使用固定姿态。此选项需要 SciPy。

For front RGB or privileged localization, use empty grasp presets, disable planning while opening, and set `"adaptive_orientation": true` in the calibration file to try alternative wrist orientations when the calibrated waypoints leave little joint travel or have excessive position error. Approach and descent share one orientation. Lift is checked separately. If no candidate meets the tighter margins, the fixed reference goes through the standard IK checks. The default uses fixed orientations.

### 随机化相机的标定 / Randomized-camera calibration

配套LeIsaac 0.1.2镜像中，相机随机化调用`set_world_poses`后，`camera.data`位姿缓存可能仍是名义值；不能用缓存值未变推断渲染相机未变。核对实际相机位姿接口或独立标定，并记录坐标系与像素中心变换。逐回合实际外参属于额外的标定输入契约，须与固定外参结果分组，不从隐藏物体位置反推；正式真机接口能否提供或重建它仍待确认。公共入口缺省读取外部标定文件，只有显式开关才读取场景相机标定。

In the matching LeIsaac 0.1.2 runtime, cached camera poses can remain nominal after pose randomization. Verify the actual pose API or independent calibration rather than trusting an unchanged cache. Per-reset extrinsics are an explicit calibration-assisted input variant, not a hidden object-state correction; hardware availability is unconfirmed. By default the public runner reads an external calibration file; scene-camera calibration requires an explicit flag.

### 正式标定发布后如何适配 / Adapting official calibration

一般无需重写检测器和整个状态机，按变化范围修改：

| 发布内容 | 对应修改 |
| --- | --- |
| 内参、外参、图像尺寸或格式 | 转换为现有`K/image_shape/camera_to_base`；核对光学轴、变换方向、米/毫米、resize与畸变处理 |
| 每回合标定读取接口 | 在`camera_calibration.py`接正式读取方式，经`RecordedEnv.reset`更新并记录；不再默认依赖私有仿真view |
| 桌面/基座坐标定义改变 | 修正坐标变换、可见平面及工作区；保留独立FK/投影检查 |
| 正式末端动作SDK | 调整运动执行边界，将任务目标交给SDK；旧关节目标不能只做FK后改名当作末端策略 |
| 多物体与放置目标 | 扩展检测/选择、抓取几何和搬运/释放阶段，不把单红方块逻辑直接当作已支持 |

相机参数与机器人/TCP标定是两类前提：相机标定不会自动解决关节跟踪、夹持中心或物体高度差。新文件先通过`--check`及独立图像→坐标核对，再做获准的单回合闭环，最后冻结新协议评测。旧版本和成绩保留，适配后的分组不与旧组混算。

Usually only the calibration reader and coordinate conversion change: normalize official parameters into the existing schema and verify units, axes, resize and distortion. If the action SDK changes, update the motion/execution boundary rather than relabeling old joint commands. Detection and task staging can be reused, while new objects and placement need their own logic. Validate independent projection/FK references and a short rollout before evaluating a frozen new protocol; retain the old version and results.

## 可选几何抓取模式 / Optional geometric mode

在外部标定文件中填入非空`grasp_presets`，入口会选择`starter.scripted_geometry.GeometricLiftCube`；默认空数组继续使用原策略。此模式要求`kinematics_backend: "joint_chain"`与`gravity_compensation: true`。每个预设包括`tilt`标签、固定工具`rotation`和`tcp_offset_m`，不包含物体位置或按seed编写的动作。

视觉控制脚本可在标定中启用 `"fast_clearance": true`。夹爪到达闭合目标或触发接触代理后等待 8 个策略步，离桌阶段每步最多移动 0.5 motor。默认分别为 15 步和 0.25 motor。两种设置都依据实测关节的 FK 高度判断何时进入运输。

Set `"fast_clearance": true` with geometric grasp presets for an eight-tick dwell after closure and a clearance rate of 0.5 motor units per policy tick. The defaults are 15 ticks and 0.25 motor units per tick.

可参考[固定资产示例](../course/liftcube/configs/scripted_grasps.example.json)，先核对资产及其3cm方块、夹爪开度和名义净空假设，再将`presets`复制为自己标定文件的`grasp_presets`，并按配置模板填写资产信息。入口拒绝预设与关节链来源资产不匹配。示例不包含相机、工作区、桌面、FK独立核验等必要标定，不能单独启动。更换资产或抓取对象须重新验证相关几何，不盲用这些数字。

算法由RGB产生位置，联合求可达抓取偏航，依次核验抓取/hover/运输预设和关节余量。接近路径保持原参考/发送步长；闭合后先保持抓取方向，以≤0.25motor/策略步离桌。实测q的FK高度达到`hover_z−3mm`且朝向误差≤0.15rad连续三次后，进入原运输目标；不要求带载中间点完全静止，但硬跟踪保护不变。此子阶段只判断离桌，不判断是否夹住；空抓和掉落仍按原生success失败。最后的物理、执行器、关节限位、640tick预算及0.20m评分不变。

配套仿真可显式加入`--runtime-camera-calibration`，通过`starter.camera_calibration`读取相机/基座标定，记录到每回合JSONL的reset行；不开启则只用外部固定标定，不读取场景相机。记录清单同时保存开关和标定代码版本。此辅助接口仅适配当前仿真，不代表课程真机SDK已提供同样能力；仍不能读取隐藏物体位置回填标定。`--check`即使带此开关也不创建仿真环境。

The opt-in geometric policy chooses reachable fixed-CAD grasps, clears the table slowly, then tilts and lifts. Its intermediate FK-height gate is not grasp success; native scoring and hard guards remain unchanged. Copy presets only after checking the example's asset identity and assumptions, and complete independent calibration. `--runtime-camera-calibration` explicitly enables a matching-simulator-only camera/base provider and logs each reset transform; this is a distinct input contract, not a confirmed hardware feature.

几何模式还可在标定文件中设置`plan_while_opening: true`（默认false）：张开期间积累RGB稳定观测，原开爪轨迹和实测到位等待全部保留；张开结束时已有三次稳定观测才直接规划接近，否则沿用原定位阶段。不提前移动手臂，不缩短闭合或带载离桌等待。运行清单记录该开关，开关不同的成绩分别报告。

In geometric mode, `plan_while_opening: true` accumulates stable RGB observations while preserving the entire original jaw-opening motion and measured-state dwell. It plans after opening only if three stable observations are ready; otherwise it uses the original localization stage. It does not move the arm early or shorten closure/loaded-clearance waits. The default is false, and the manifest records the option.

## 标定文件 / Calibration

复制[模板](../course/liftcube/configs/scripted_calibration.template.json)到仓库外的运行目录，填入有依据的值。不要使用单测中的合成数据。以下数值统一在**所选运动学模型的机械臂基坐标系**中表达，长度为米、旋转为矩阵。

Copy the template outside the repository and fill it from measured or independently verified calibration. Test fixtures are not usable calibration. All geometry is in the selected robot model's base frame, in metres.

| 字段 / Field | 定义 / Meaning |
| --- | --- |
| `evidence` | 标定记录/版本引用，覆盖环境资产、相机、桌面、TCP与关节映射；不放凭据 / Calibration evidence reference |
| `image_shape`, `K` | 实际策略图像形状（64/128/256方形RGB）与对应内参；匹配resize像素中心约定 / Shape and intrinsics after the exact resize |
| `camera_to_base` | 4×4 optical-camera→robot-base变换，光学坐标x右/y下/z前 / Optical camera transform |
| `visible_plane_z` | 检测质心所用可见平面的高度，不能机械等同桌面高度 / Visible detection plane height |
| `grasp_z`, `hover_z`, `lift_z` | TCP抓取、接近、抬升高度，按当前资产/评分确定 / TCP waypoint heights |
| `grasp_xy_offset` | 单侧活动夹爪的固定XY抓点修正；固定抓取方向下有效 / Fixed grasp offset |
| `grasp_rotation` | 接近与夹持时URDF TCP在基坐标系中的3×3目标旋转 / Approach/grasp rotation |
| `lift_rotation` | 可选抬升方向，缺省沿用`grasp_rotation`；仍按0.02rad渐变及原IK容差验收 / Optional lift rotation |
| `gravity_compensation` | 可选布尔值，默认false；启用有界稳态补偿，不修改执行器 / Optional bounded steady-bias correction |
| `contact_close` | 可选布尔值，默认false；夹爪位置反馈限领先及停滞保持，不读取接触真值 / Optional jaw-position feedback closure |
| `plan_while_opening` | 可选布尔值，默认false；仅几何模式，张开期间积累视觉定位，不改变原开爪命令和到位等待 / Optional localization during unchanged jaw opening |
| `workspace` | `[xyz_min, xyz_max]`；必须是已核验的无遮挡工作范围 / Validated clear workspace |
| `open_motor`, `close_motor` | 已核验的公共夹爪请求，满足0≤close<open≤100 / Motor targets |
| `kinematics_backend` | `urdf`（缺省）或`joint_chain` / Backend selection |
| `urdf_path` | URDF后端的路径；独立核对与当前USD一致 / URDF backend source |
| `chain_path`, `tcp_offset_m` | 关节链JSON路径及固定夹爪frame中的三维TCP偏置；原生轴向、零偏置 / Joint-chain source and fixed tool offset |
| `tcp_frame`, `joint_names` | 已标定工具frame、按公共五臂关节顺序排列的URDF名称 / TCP and ordered arm names |
| `joint_sign`, `joint_offset_deg` | 五维轴向与零位映射，不能从同名模型猜测 / Axis and zero-offset mapping |
| `fk_checks` | 至少三个不同臂姿态，各含`motor`六维、独立获得的`tcp_xyz`三维和`tcp_rotation`3×3 / Independent FK pose references |

先独立验证TCP位置/方向、相机投影与关节映射，再填写该文件。`--check`只做数值、FK参考与IK回算检查，不证明标定记录真实或碰撞安全。仿真模型导出的固定机器人/相机标定可以作为几何先验，但每回合物体位置、分割或成功标记不得用于更新策略标定；正式课程是否允许这些固定先验仍按评分要求确认。

例如64→256同视场resize，`fx/fy`乘4，`cx/cy`按`(c+0.5)*4-0.5`转换；同时在独立评测YAML设置`image_height: 256`、`image_width: 256`。只改图像形状而复用旧内参是不合法标定。入口会拒绝配置与标定尺寸不一致；不同输入分辨率的成绩须注明，不自动视为同一比较协议。

The visual control script evaluator also accepts explicitly calibrated 128/256-square RGB from the same front camera. Match the environment resize, calibration shape and pixel-centre-adjusted intrinsics. The default remains64; other policies and native camera settings are unchanged. Declare resolution differences in comparisons.

SO101是一侧固定、一侧活动的夹爪。闭合时的参考点不一定是给定物体尺寸下的夹持中心；还须核对固定指/活动指的实际碰撞形状、张开角度、接近方向和桌面净空。先用固定资产几何检查，再以实测关节与独立接触/抓取结果验证；不能因为IK能到达就加大控制补偿硬推。

SO101 has one fixed and one moving finger. A closed-jaw reference is not necessarily the grasp center for a finite-sized object. Check both collision shapes, jaw opening, approach direction and table clearance before physical validation. Reachable IK alone does not justify increasing compensation against a contact obstruction.

Distortion must be negligible or already corrected by the observation producer; this baseline does not silently undistort images. Camera randomization invalidates a fixed transform unless the public interface supplies a permitted calibration update. A successful round trip is not independent proof of correct kinematics.

## 检查与运行 / Check and run

CPU软件单测（NumPy、Torch；集成测试另需PyYAML），仓库根目录：

```bash
python -m unittest discover -s tests -p test_scripted_liftcube.py -v
```

固定关节链后端需要NumPy/SciPy；JSON记录五个臂关节的父/子frame、原限位、轴向和来源USD。导出是CPU操作，不启动SimulationApp，不包含物体/相机真值或碰撞资产。只适配已核验的SO101五轴Z旋转链，不是通用机器人解析器。

配套镜像中导出（输出必须是新文件；USD Python库需按镜像实际路径配置）：

```bash
USD_LIB_DIR=/isaac-sim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
PYTHONPATH="$USD_LIB_DIR:$PWD/src" LD_LIBRARY_PATH="$USD_LIB_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /isaac-sim/python.sh -m starter.scripts.export_joint_chain \
  --asset /leisaac/assets/robots/so101_follower.usd --output /path/to/chain.json
```

在外部标定文件中设置`kinematics_backend: "joint_chain"`、`chain_path`及`tcp_offset_m`，并填写共同标定字段。该后端要求`joint_sign=[1,1,1,1,1]`、`joint_offset_deg=[0,0,0,0,0]`和导出文件中的关节顺序。使用前仍需独立完成FK和抓取标定。

The joint-chain backend uses NumPy/SciPy and fixed geometry exported from the matching SO101 USD. It requires native axes/zero offsets and independently validated camera/TCP calibration. Export does not launch the simulator or read live object state.

URDF后端另需兼容的LeRobot `lerobot.model.kinematics.RobotKinematics` 与Placo。优先用已有兼容依赖；不要升级课程Python/Torch来满足新包。实现兼容LeRobot v0.4.2单次IK迭代接口。URDF及标定不随仓库伪造提供。

本文开头给出了两种脚本的完整命令。打包的 `configs/scripted/eval.yaml` 运行 50 场、每场最多 960 步；通用 `configs/eval_lift_cube.yaml` 的默认值是 4 场、640 步，使用时须匹配标定图像尺寸。每场重置策略，结果写入新的输出目录。

Outputs: `manifest.json` identifies calibration, kinematics data/code, config, base policy, actual policy class/code and runner versions; each episode writes a flushed JSONL trace and RGB NPZ (reset plus post-step/terminal images), and completed episodes enter the shared `metrics.json`/`episodes.csv`. `run-status.json` distinguishes finished runs from errors. A partial episode is not counted as a complete failure. Recorded RGB and diagnostics stay outside Git.

适配器关闭LeIsaac默认的原生演示记录器：其每步GPU拼接完整分辨率RGB历史会造成显存池快速增长。本入口仍显式保存CPU图像和轨迹，不改变相机、物理或成功判据；原生演示HDF5导出不属于这个评测入口。

The adapter disables LeIsaac's implicit demonstration recorder, whose growing GPU RGB concatenations inflate the allocator cache. Explicit CPU image/trajectory recording remains enabled; cameras, physics and scoring are unchanged. Native demonstration HDF5 export is a separate collection workflow.

## 验收及后续扩展 / Acceptance and extension

### 几何模式＋实际相机标定辅助 / Geometric, calibration-assisted variant

冻结front256 RGB、6D实测q、实际相机/基座外参、固定CAD预设、反馈闭合和分阶段抬升；环境60Hz、策略30Hz、每回合≤640tick。原物体/相机随机化、动作范围、执行器和原生成功判据不变，没有训练或物体真值输入。

| 原版测试集 | 原生成功 | 成功seed | 成功回合中位耗时（仿真秒） |
| --- | --- | --- | --- |
| 首轮seed30–39 | 5/10 | 30、32、34、38、39 | 9.35 |
| 新增seed40–59 | 8/20 | 40、42、45、47、48、50、51、56 | 9.43 |
| 同配置合计seed30–59 | 13/30（43.3%） | 上述全部 | 9.38 |

新增20回合在原公共几何入口实际运行。全部完整轨迹的动作映射、原范围、无裁剪、发送连续性、原生终态/RGB经独立CPU审核。新增12个失败中，4个触发原跟踪保护，2个阶段超时，6个未形成有效抬升但无脚本故障。`hold`或夹爪停滞代理都不能算成功；原生高度判定也不等于额外的稳定持物测试。比例只描述这些有限场景。

### 定位调度优化配对 / Paired localization-scheduling comparison

开发只使用旧seed17–19；候选冻结前不读取新增20场景成绩，预先指定seed40–49作十回合配对。提前手臂接近和RGB跨度修正两种候选在开发中分别为1/3、0/3，未采用。保守候选只在原开爪期间积累视觉观测，开发3/3不计入正式分母。

| seed40–49同场景配对 | 原版 | `plan_while_opening: true` |
| --- | --- | --- |
| 原生成功 | 5/10 | 5/10 |
| 成功seed | 40、42、45、47、48 | 相同 |
| 共同成功回合中位耗时 | 9.40仿真秒 | 9.33仿真秒 |
| 视觉平面XY定位误差中位数（仅评分侧） | 0.815mm | 0.830mm |

五个共同成功回合分别节省6/2/8/4/4环境tick，中位节省4tick（约0.067仿真秒，0.7%）；未证明精度或成功率提高，也不作为服务器实际运行耗时提速。两组记录的初始关节、物体位置和相机变换一致，但未记录的物体朝向及隐藏PhysX/RTX状态不作位级等价承诺。优化版十回合也完成独立全轨迹审核，不与原版30回合混算。

公共开关实现通过26项课程CPU测试：保留开爪命令、视觉缺失回退及配置拒绝；以完整记录重算，开启时复现两个候选开发回合，关闭时复现一个原版扩展回合，阶段和逐命令最大差均0。物理配对来自冻结候选，共用原评测循环；不为公共重构重复同一批物理试验。

The unchanged assisted controller completed 30 scenes with 13 native successes, including 8/20 new runs through the public geometric entry point. All full trajectories were independently audited. The frozen scheduling candidate retained the same five successes on ten paired scenes; common-success median time improved by only 0.067 simulation seconds. Localization accuracy did not improve, and simulation-time savings do not establish wall-clock speedup. Development results are excluded. Recorded initial joints, object position and camera transforms match, but hidden simulator state equality is not claimed. Twenty-six targeted runtime CPU tests and full-trace command equivalence cover the public opt-in port; actual per-reset calibration remains an auxiliary input with unconfirmed hardware availability.

### 定位调度变体100场景 / Scheduling variant across 100 scenes

冻结seed1000–1099，每回合至多640环境tick，front256、实测6D q及每reset实际相机标定辅助，原60Hz环境/30Hz策略、原物理/限位/执行器/原生成功判据不变。100回合完整结束并通过轨迹、动作映射、原生终态审核：45成功、55失败，成功平均9.468、中位9.383仿真秒；失败中16次跟踪保护、2次阶段超时、37次无脚本错误但未达原生成功。

这组使用冻结开发入口的定位调度版本，对应公共几何模式的`plan_while_opening: true`。公共移植已复用完整记录输入进行逐命令/阶段等价检查，最大命令差0；没有另把公共CLI重跑100次。复现这组历史数据需使用当时的环境和标定。当前 50 场配置见本文开头。该成绩不属于自研末端执行层，也不能作为未来正式标定/控制SDK或真机的成绩。

The frozen development scheduling runner completed all 100 scenes (seeds1000–1099): 45 successes, 55 failures; successful mean/median times were 9.468/9.383 simulation seconds. Failures comprised 16 tracking guards, 2 stage timeouts and 37 native failures without a script error. The public equivalent uses geometric mode with `plan_while_opening: true`; its port was checked for exact command/stage equivalence on complete recorded inputs, rather than rerunning another 100 physical episodes through the CLI. Reproduction requires matching calibration and environment identity. This result does not validate the planar EE development controller, an unreleased official SDK, new objects or hardware.

### 平滑与泛化的边界 / Smoothing and generalization limits

后续在旧开发seed17–19分别检查了三种控制变体，全部完成原640tick预算或原生提前成功并独立审核：

| 开发对照 | 原生成功 |
| --- | --- |
| 已发布的开爪期间定位方案 | 3/3 |
| 连续步长接近＋臂指令因果滤波 | 1/3 |
| 上述方案＋RGB跨度适配 | 1/3 |
| 原路径＋全流程预设排序＋臂指令因果滤波 | 2/3 |

三个变体均未采用，公共默认与可选方案保持不变。最后一组在共同成功的18/19中，臂指令三阶差分RMS约减半，但活动窗口二阶差分p95上升，实测q差分也非一致改善；17从成功变为空抓。命令尖峰更低不等于真实抓取更稳或全程运动更平滑，不能用这些开发比例替换13/30的原版独立结果。这三种滤波/路径变体未通过开发门槛，没有进入预定新seed60–69独立测试。

当前原生环境的随机化范围为方块x/y各±7.5cm、偏航±30°，相机xyz各±5mm、朝向各±0.05°；未改分布。现有跨场景成绩只涉及同一3cm红方块，不覆盖新颜色、形状、尺寸、遮挡分布或真机。扩展物体时须更新检测/抓取几何并使用独立场景验收，不能仅更换seed便宣称跨物体泛化。

Three later development variants achieved 1/3, 1/3 and 2/3 versus the existing localization-scheduling variant's 3/3 on the same old scenes. None was adopted. Lower command-jerk peaks did not consistently improve measured motion or grasping; these results neither replace the independent 13/30 baseline nor establish generalization. These three variants did not advance to the planned new60–69 set. Existing scene variation covers the same 3cm red cube with ±7.5cm planar position, ±30° yaw, ±5mm camera position and ±0.05° camera rotation; novel objects and hardware remain untested.

### 闭合几何对照 / Closure-geometry comparison

两项仓库外控制实验沿用原物理、限位、执行器和640tick预算，输入仍为front256、实测六关节及实际相机标定辅助。先在旧seed40–44开发：仅张开目标30→35为1/5，低于现有方案2/5，弃用；另一方案用固定CAD模型与实测开度更新水平夹持中点，保持原开度，开发3/5后冻结，进入此前未使用seed60–69同场景对照。

| 新seed60–69，各10完整回合 | 现有定位调度方案 | 动态夹持中点候选 |
| --- | --- | --- |
| 原生成功 | 6/10 | 4/10 |
| 成功seed | 60、61、63、65、68、69 | 61、63、65、69 |

全部轨迹动作映射、原范围、无裁剪、连续性及原生终态/RGB审核通过。候选没有新增成功，丢失60/68，闭合推移分别从4.90/3.81mm增至25.14/17.00mm。四个共同成功回合中，候选中位慢0.083仿真秒；臂命令三阶差分RMS比值中位0.913，但实测关节二阶差分RMS比值中位1.191。因此两项新方案均不并入公共运行代码，保留现有控制。

两组记录的初始关节/物体XYZ/相机变换及臂接近前状态匹配，未承诺隐藏模拟器状态位级相等。物体位置只用于评分侧推移诊断，不输入策略。新对照与原控制配置的13/30分协议统计；同一红方块的位置/偏航随机化仍不足以证明跨物体能力。

Two external controller experiments preserved the original physics, actuators, limits and 640-tick budget. Wider opening achieved 1/5 on old development scenes versus 2/5 and was rejected. A CAD/measured-jaw midpoint controller achieved 3/5 in development, was frozen, then scored 4/10 versus the existing scheduling controller's 6/10 on new paired seeds60–69. It gained no successes and lost two. On four common successes it was 0.083 simulation seconds slower; median command-jerk ratio was 0.913, but measured-joint second-difference RMS ratio was 1.191. Full trace/native-result audits passed; neither experimental controller is included in the public runtime. Recorded initial and pre-approach states match without claiming hidden-state identity. Scoring-side object truth never enters the controller, and these paired results remain separate from the earlier 13/30 protocol.

### 原固定标定方案 / Earlier fixed-calibration variant

2026-10-28独立开发评测：公共关节链入口，front64 RGB＋6D q，原生物体/相机随机化、60Hz环境/30Hz策略、每回合最多640环境tick。固定19mm TCP侧偏置、8cm预抓高度、竖直夹持/倾斜40°抬升，启用`gravity_compensation`与`contact_close`。代码、标定和seed20–29在测试前冻结，十回合全部完成，未换seed或中途调参；原物理、执行器、限位和成功判据不变。

| 原生结果/主要失败 | 回合数 | 说明 |
| --- | --- | --- |
| 成功 | 1 | seed26，540tick；原生高度阈值通过 |
| 固定抓取方向不可达 | 6 | 阶段预检拒绝，未放宽IK容差 |
| 接近阶段超时 | 1 | 保留原5策略秒阶段保护 |
| 腕轴请求越界 | 1 | 拒绝发送，不裁剪或放宽原范围 |
| 空抓后超时 | 1 | 流程到hold，但原生评分失败 |

完整轨迹的动作映射、原范围、无执行限幅、连续性、reset前RGB/终态、原生高度评分均经独立CPU复核。1/10仅是这十个场景的实测比例，不是总体成功率保证，也不代表课程正式多物体评分。

反馈闭合的开发场景17/18分别在566/532tick成功，19仍不可达；原闭合方案在同三个开发场景中仅17成功。开发集改善没有转化为广泛场景稳定性。更早同seed的TCP选参和随后抬高/倾斜诊断均独立保留，不混入上表，也不把成功轨迹自动作为BC数据。

The frozen independent set completed all ten episodes: 1 success, 6 unreachable preflights, 1 approach timeout, 1 rejected wrist target and 1 empty grasp. All full traces and native terminal scoring were audited. This is a finite-set result, not a population estimate or the formal multi-object benchmark. Development improvements (2/3 versus 1/3 in the original three scenes) remain separate.

只调整末段闭合窗口时，设 `relaxed_close_window: true` 并保留原抓取姿态（`face_aligned_grasp: false`）。该组合沿用原来的定位、悬停和下降目标。

在 `contact_close: true` 下可用 `hold_contact_on_lift: true` 检验带载恢复。夹爪根据关节位置反馈检测到持续阻挡后，在抬升时保持该位置目标。如果此时腕部跟踪滞后，先让臂关节指令回到实测位置，再保持夹爪目标继续抬升。未检测到夹持的空抓沿用张开重试流程。`contact_preload_motor` 默认为 0，设为 1 会在首次确认阻挡时把目标开度再减小约 1.1°；保持期间该偏移只施加一次。

带载抬升可启用 `pace_loaded_lift`。夹爪保持闭合时，每个臂关节的位置目标最多领先实测位置 7.5 个 motor 单位，其他关节可继续沿抬升轨迹运动。稳态补偿后的目标使用相同上限。原有每步限速和 8 个单位的跟踪保护继续生效。

RGB 视觉控制脚本可启用 `low_fingertip_grasp`。脚本从已标定的零倾角姿态开始，依次尝试 −5°、−10°、−15° 和其余标定姿态，并检查接近、闭合与抬升目标的可达性。中间姿态复用零倾角对应的工具中心点。夹取深度由 `grasp_z` 单独设置，调整时应同时检查固定指和活动指距方块底面及桌面的间隙。

启用 `face_aligned_grasp: true` 后，控制器按方块两对相对侧面选择抓姿。RGB 视觉控制脚本用红色轮廓拟合 30 mm 方块的朝向，特权state控制脚本直接读取底座坐标系中的方块朝向。控制器在竖直至倾斜 30° 的姿态中检查悬停和抓取位置是否可达。首次接近优先选两指高度接近的一对侧面。特权state控制脚本在接近受阻或方块掉落后，先在当前腕姿下抬到悬停高度，再重新定位，优先换到正交的另一对侧面。若原抬升终点不可达，则预先检查相同高度的其他位置和腕姿，先垂直提离桌面，再转入高位运输。

末段闭爪窗口容许 TCP 横向误差 5 mm、低于目标 3 mm、高于目标 6 mm。特权state控制脚本已选定可达侧面抓姿时，高于目标的容差为 8 mm，给接近桌面的低侧指尖留出空间。朝向和关节跟踪须连续三个控制时刻达标。

对 SO‑101 前视 RGB 视觉控制脚本，`follow_retry_close: true` 配合 `hold_contact_on_lift: true` 用于检查夹持开度。轮廓可辨时，按方块沿闭合方向的投影宽度核对指垫间距。首次遇阻时开度仍过宽，脚本在相同平面位置抬高 3.5 mm，再张爪闭合一次。完整重抓使用重新拟合的方块中心，夹持反馈触发后最多再收紧 1.5 motor，以免滑脱时一直追到全闭。重抓试提达到 30 mm 后，连续检查原位置是否仍能看到方块。

软件测试覆盖检测歧义、投影、IK残差/动作范围、状态机、限速、失败锁定、状态重置、不读取补零/metadata，以及2tick保持和原生终帧记录。假运动学与替身环境只能验证软件流程。

用于其他标定/运行环境时仍需：匹配标定与真实IK后端→单回合核对初态、RGB定位、请求/实测、真实夹持和原生success→冻结场景集合评估。当前重点是闭合推移、有效夹持与带载跟踪，不据小幅定位误差继续增加闭合力度或放宽保护。脚本成绩不能替代候选公开演示的两组重放检查。

Multi-object work should replace target detection and grasp geometry, then add transfer/release stages while retaining the shared execution and native scoring. Do not widen color thresholds or change task physics to make a benchmark pass. No multi-object capability is claimed here.

设计参考：[Isaac Lab v2.1.1状态机](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/scripts/environments/state_machine/lift_cube_sm.py)、[LeRobot v0.4.2运动学](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/model/kinematics.py)及[SO101单活动指与分阶段姿态说明](https://github.com/lukehollis/lerobot-lewam/blob/main/docs/so101-sim-pick-lift.md)。本策略独立实现：借鉴接近/闭合/抬升流程与夹爪几何思路，不复制社区控制代码或使用其物体真值输入；官方运动学作为可选依赖调用，沿用其许可。
