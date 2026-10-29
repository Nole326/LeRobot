# LiftCube视觉规则基线 / Visual rule baseline

入口为 `starter.scripts.evaluate_scripted`，策略在 `starter/scripted_liftcube.py`。流程：张开夹爪 → RGB定位 → 上方接近 → 下降 → 闭合 → 抬升 → 保持。没有学习模型、训练步骤或演示标签依赖。

Entry point: `starter.scripts.evaluate_scripted`. The policy opens, localizes the cube from RGB, approaches, descends, closes, lifts and holds. It uses no learned policy or demonstration labels.

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

- 只读取单front RGB与本体前六维实际电机位置。默认图像为uint8 HWC 64×64×3；规则策略可独立配置128或256方形输入，须同步环境resize、标定`image_shape`与内参`K`。这是同一路相机的预处理，不改变相机原始分辨率或其他模型默认值。忽略补零速度、末端字段、metadata和环境info；记录器单独接收执行诊断与原生success。
- 原始实测位置只检查形状与有限性，不当作待发送动作裁剪；接触/求解器可能使反馈略超名义行程。所有发送目标仍须通过原范围与跟踪检查。Raw feedback is preserved; outgoing targets remain bounded.
- 图像检测使用红色通道阈值和连通域，不用分割真值、深度或物体坐标。多个候选、目标缺失或触边时拒绝猜测。像素射线与**预先标定的可见物体平面**相交；该平面不是每回合读取的真值。侧面可见、光照变化、夹爪遮挡会造成偏差或保守失败。
- 整个红色连通域可能同时包含顶面和侧面，不能把它的质心机械地当作顶面中心。平面高度须与检测方式配套，并用独立位置参考核验；不能根据单个测试场景的物体真值回填偏移。
- 运动学可使用固定SO101关节链（NumPy/SciPy），或LeRobot官方`RobotKinematics`/Placo。前者从匹配USD导出固定几何，不读取实时场景；后者按`URDF角度 = USD角度 × joint_sign + joint_offset_deg`转换。夹爪独立使用公共motor单位，不把motor直接当角度。
- 固定抓取方向只适用于其可达区域；SO101五轴不能实现任意六自由度位姿。IK检查位置误差≤3mm、旋转误差≤0.15rad，并拒绝超限解，不靠裁剪修复不可达目标。
- Cartesian参考每次至多移动4mm，朝向沿最短旋转每次至多推进0.02rad；最终关节请求按统一比例插值，每次最多变化1motor单位（30Hz下30单位/s）。位置和朝向均到位后才进入下一阶段。这不等于实际关节速度上限或完整路径碰撞保证。臂关节命令领先实测超过8单位时锁定失败。
- 路径从上一发送目标继续推进，不以滞后反馈重新起算参考，避免重力下垂不断拖低接近轨迹；到位判断和跟踪保护仍使用实测关节。This command-referenced path does not replace measured-state arrival and lag checks.
- 可选`gravity_compensation: true`使用`SteadyBiasScriptedLiftCube`：只在接近/下降/抬升的名义目标稳定后按q反馈积分，臂轴偏置限±3motor，夹爪不补偿；名义路径与发送目标分开记录。仍拒绝原范围外请求，保持1motor/策略步及8motor领先保护。不能用补偿追赶碰撞；触发保护后保持最后发送值。
- 起始TCP必须已在标定工作区且不低于hover高度；脚本不会盲目执行回零/抬臂恢复。下降后允许夹爪遮挡；闭合完成请求后停留0.5策略秒，不把夹爪实际无法完全闭合作为必然失败。
- 可选`contact_close: true`将闭合请求限制为至多领先实测夹爪2motor。连续三次位置变化小于0.1motor且反馈领先上一命令至少1.5motor时，保持当前夹爪目标，停留0.5策略秒后抬升；没有停滞则仍可闭合到原目标。`jaw_contact_proxy`只表示位置反馈停滞，也可能来自桌面等障碍，不是力测量或抓取成功；`grasp_motor`单独记录。原动作限速、臂跟踪保护、执行器及评分不变，缺省false保留原闭合行为。
- 失败后保持最后命令，直到原生终止或评测预算结束；不是硬件急停。`hold`只表示控制流程结束，不表示抓取成功。成功率仍来自原生任务评分，失败和超时均保留。

Only front RGB and measured joint positions reach the policy. Detection is color/connected-component based; localization uses a calibrated plane, not per-episode object state. IK and motor bounds are checked, commands are rate-limited, and faults latch a hold until the episode ends. Native environment scoring alone determines success. Fixed orientation, occlusion, initial-pose assumptions and the absence of full collision checking limit the operating envelope.

A whole-object color centroid can include top and side faces; it is not automatically the top-face center. Validate the detection plane independently instead of fitting offsets to a test scene's hidden object pose. A reachable commanded waypoint also does not prove that the physical arm reaches it under gravity and contact.

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

可参考[固定资产示例](../course/liftcube/configs/scripted_grasps.example.json)，先核对资产SHA及其3cm方块、夹爪开度和名义净空假设，再将`presets`复制为自己标定文件的`grasp_presets`、`asset_sha256`复制为`grasp_asset_sha256`。入口拒绝预设与关节链来源资产不匹配。示例不包含相机、工作区、桌面、FK独立核验等必要标定，不能单独启动。更换资产或抓取对象须重新验证相关几何，不盲用这些数字。

算法由RGB产生位置，联合求可达抓取偏航，依次核验抓取/hover/运输预设和关节余量。接近路径保持原参考/发送步长；闭合后先保持抓取方向，以≤0.25motor/策略步离桌。实测q的FK高度达到`hover_z−3mm`且朝向误差≤0.15rad连续三次后，进入原运输目标；不要求带载中间点完全静止，但硬跟踪保护不变。此子阶段只判断离桌，不判断是否夹住；空抓和掉落仍按原生success失败。最后的物理、执行器、关节限位、640tick预算及0.20m评分不变。

配套仿真可显式加入`--runtime-camera-calibration`，通过`starter.camera_calibration`读取相机/基座标定，记录到每回合JSONL的reset行；不开启则只用外部固定标定，不读取场景相机。记录清单同时保存开关和标定代码SHA。此辅助接口仅适配当前仿真，不代表课程真机SDK已提供同样能力；仍不能读取隐藏物体位置回填标定。`--check`即使带此开关也不创建仿真环境。

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
| `urdf_path`, `urdf_sha256` | URDF后端的路径和SHA256；独立核对与当前USD一致 / URDF backend source |
| `chain_path`, `chain_sha256`, `tcp_offset_m` | 关节链JSON路径/SHA及固定夹爪frame中的三维TCP偏置；原生轴向、零偏置 / Joint-chain source and fixed tool offset |
| `tcp_frame`, `joint_names` | 已标定工具frame、按公共五臂关节顺序排列的URDF名称 / TCP and ordered arm names |
| `joint_sign`, `joint_offset_deg` | 五维轴向与零位映射，不能从同名模型猜测 / Axis and zero-offset mapping |
| `fk_checks` | 至少三个不同臂姿态，各含`motor`六维、独立获得的`tcp_xyz`三维和`tcp_rotation`3×3 / Independent FK pose references |

先独立验证TCP位置/方向、相机投影与关节映射，再填写该文件。`--check`只做数值/哈希、FK参考与IK回算检查，不证明标定记录真实或碰撞安全。仿真模型导出的固定机器人/相机标定可以作为几何先验，但每回合物体位置、分割或成功标记不得用于更新策略标定；正式课程是否允许这些固定先验仍按评分要求确认。

例如64→256同视场resize，`fx/fy`乘4，`cx/cy`按`(c+0.5)*4-0.5`转换；同时在独立评测YAML设置`image_height: 256`、`image_width: 256`。只改图像形状而复用旧内参是不合法标定。入口会拒绝配置与标定尺寸不一致；不同输入分辨率的成绩须注明，不自动视为同一比较协议。

The rule runner also accepts explicitly calibrated 128/256-square RGB from the same front camera. Match the environment resize, calibration shape and pixel-centre-adjusted intrinsics. The default remains64; other policies and native camera settings are unchanged. Declare resolution differences in comparisons.

SO101是一侧固定、一侧活动的夹爪。闭合时的参考点不一定是给定物体尺寸下的夹持中心；还须核对固定指/活动指的实际碰撞形状、张开角度、接近方向和桌面净空。先用固定资产几何检查，再以实测关节与独立接触/抓取结果验证；不能因为IK能到达就加大控制补偿硬推。

SO101 has one fixed and one moving finger. A closed-jaw reference is not necessarily the grasp center for a finite-sized object. Check both collision shapes, jaw opening, approach direction and table clearance before physical validation. Reachable IK alone does not justify increasing compensation against a contact obstruction.

Distortion must be negligible or already corrected by the observation producer; this baseline does not silently undistort images. Camera randomization invalidates a fixed transform unless the public interface supplies a permitted calibration update. A successful round trip is not independent proof of correct kinematics.

## 检查与运行 / Check and run

CPU软件单测（NumPy、Torch；集成测试另需PyYAML），仓库根目录：

```bash
python -m unittest discover -s tests -p test_scripted_liftcube.py -v
```

固定关节链后端需要NumPy/SciPy；JSON记录五个臂关节的父/子frame、原限位、轴向和来源USD SHA。导出是CPU操作，不启动SimulationApp，不包含物体/相机真值或碰撞资产。只适配已核验的SO101五轴Z旋转链，不是通用机器人解析器。

配套镜像中导出（输出必须是新文件；USD Python库需按镜像实际路径配置）：

```bash
USD_LIB_DIR=/isaac-sim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
PYTHONPATH="$USD_LIB_DIR:$PWD/src" LD_LIBRARY_PATH="$USD_LIB_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /isaac-sim/python.sh -m starter.scripts.export_joint_chain \
  --asset /leisaac/assets/robots/so101_follower.usd --output /path/to/chain.json
```

在外部标定文件中设置`kinematics_backend: "joint_chain"`、`chain_path`、`chain_sha256`及`tcp_offset_m`，并填写共同标定字段。该后端要求`joint_sign=[1,1,1,1,1]`、`joint_offset_deg=[0,0,0,0,0]`和导出文件中的关节顺序。SHA证明文件身份，不能替代独立FK和抓取标定。

The joint-chain backend uses NumPy/SciPy and fixed geometry exported from the matching SO101 USD. It requires native axes/zero offsets and independently validated camera/TCP calibration. Export does not launch the simulator or read live object state.

URDF后端另需兼容的LeRobot `lerobot.model.kinematics.RobotKinematics` 与Placo。优先用已有兼容依赖；不要升级课程Python/Torch来满足新包。实现兼容LeRobot v0.4.2单次IK迭代接口。URDF及标定不随仓库伪造提供。

在配套容器、`course/liftcube`目录中：

```bash
export PYTHONPATH="$PWD/src:/leisaac/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
# No environment construction, GPU allocation or automatic Docker launch:
/isaac-sim/python.sh -m starter.scripts.evaluate_scripted --help
/isaac-sim/python.sh -m starter.scripts.evaluate_scripted \
  --calibration /path/to/calibration.json --check

# Only after runtime authorization, calibration and resource acceptance:
/isaac-sim/python.sh -m starter.scripts.evaluate_scripted \
  --calibration /path/to/calibration.json \
  --config configs/eval_lift_cube.yaml \
  --output /path/to/new-output --run-simulation
```

默认沿用4回合、每回合640环境tick、30Hz策略/60Hz环境，不扩预算、不改物理或成功条件。每回合新建状态机，调用共同`run_episode`。自有运行器用`finally`关闭环境，不触发训练或真机连接；输出目录已存在时拒绝覆盖。

Outputs: `manifest.json` identifies calibration, kinematics data/code, config, base policy, actual policy class/code and runner hashes; each episode writes a flushed JSONL trace and RGB NPZ (reset plus post-step/terminal images), and completed episodes enter the shared `metrics.json`/`episodes.csv`. `run-status.json` distinguishes finished runs from errors. A partial episode is not counted as a complete failure. Recorded RGB and diagnostics stay outside Git.

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

新增20回合在原公共几何入口实际运行。全部完整轨迹的动作映射、原范围、无裁剪、发送连续性、原生终态/RGB和SHA经独立CPU审核。新增12个失败中，4个触发原跟踪保护，2个阶段超时，6个未形成有效抬升但无脚本故障。`hold`或夹爪停滞代理都不能算成功；原生高度判定也不等于额外的稳定持物测试。比例只描述这些有限场景。

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

冻结seed1000–1099，每回合至多640环境tick，front256、实测6D q及每reset实际相机标定辅助，原60Hz环境/30Hz策略、原物理/限位/执行器/原生成功判据不变。100回合完整结束并通过轨迹、动作映射、原生终态与SHA审核：45成功、55失败，成功平均9.468、中位9.383仿真秒；失败中16次跟踪保护、2次阶段超时、37次无脚本错误但未达原生成功。

这组使用冻结开发入口的定位调度版本，对应公共几何模式的`plan_while_opening: true`。公共移植已复用完整记录输入进行逐命令/阶段等价检查，最大命令差0；没有另把公共CLI重跑100次。准确复现仍需同版本环境及匹配标定，仓库只提供模板与获取/核验方式，不附私有标定、原始轨迹或图像。该成绩不属于自研末端执行层，也不能作为未来正式标定/控制SDK或真机的成绩。

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

全部轨迹动作映射、原范围、无裁剪、连续性及原生终态/RGB/SHA审核通过。候选没有新增成功，丢失60/68，闭合推移分别从4.90/3.81mm增至25.14/17.00mm。四个共同成功回合中，候选中位慢0.083仿真秒；臂命令三阶差分RMS比值中位0.913，但实测关节二阶差分RMS比值中位1.191。因此两项新方案均不并入公共运行代码，保留现有控制。

两组记录的初始关节/物体XYZ/相机变换及臂接近前状态匹配，未承诺隐藏模拟器状态位级相等。物体位置只用于评分侧推移诊断，不输入策略。新对照与旧r25的13/30分协议统计；同一红方块的位置/偏航随机化仍不足以证明跨物体能力。

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

完整轨迹的动作映射、原范围、无执行限幅、连续性、reset前RGB/终态、原生高度评分与SHA均经独立CPU复核。1/10仅是这十个场景的实测比例，不是总体成功率保证，也不代表课程正式多物体评分。

反馈闭合的开发场景17/18分别在566/532tick成功，19仍不可达；原闭合方案在同三个开发场景中仅17成功。开发集改善没有转化为广泛场景稳定性。更早同seed的TCP选参和随后抬高/倾斜诊断均独立保留，不混入上表，也不把成功轨迹自动作为BC数据。

The frozen independent set completed all ten episodes: 1 success, 6 unreachable preflights, 1 approach timeout, 1 rejected wrist target and 1 empty grasp. All full traces and native terminal scoring were audited. This is a finite-set result, not a population estimate or the formal multi-object benchmark. Development improvements (2/3 versus 1/3 in the original three scenes) remain separate.

软件测试覆盖检测歧义、投影、IK残差/动作范围、状态机、限速、失败锁定、状态重置、不读取补零/metadata，以及2tick保持和原生终帧记录。假运动学与替身环境只能验证软件流程。

用于其他标定/运行环境时仍需：匹配标定与真实IK后端→单回合核对初态、RGB定位、请求/实测、真实夹持和原生success→冻结场景集合评估。当前重点是闭合推移、有效夹持与带载跟踪，不据小幅定位误差继续增加闭合力度或放宽保护。脚本成绩不能替代候选公开演示的两组重放检查。

Multi-object work should replace target detection and grasp geometry, then add transfer/release stages while retaining the shared execution and native scoring. Do not widen color thresholds or change task physics to make a benchmark pass. No multi-object capability is claimed here.

设计参考：[Isaac Lab v2.1.1状态机](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/scripts/environments/state_machine/lift_cube_sm.py)、[LeRobot v0.4.2运动学](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/model/kinematics.py)及[SO101单活动指与分阶段姿态说明](https://github.com/lukehollis/lerobot-lewam/blob/main/docs/so101-sim-pick-lift.md)。本策略独立实现：借鉴接近/闭合/抬升流程与夹爪几何思路，不复制社区控制代码或使用其物体真值输入；官方运动学作为可选依赖调用，沿用其许可。
