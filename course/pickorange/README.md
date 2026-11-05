# PickOrange：模型评测与教师采集

本目录负责PickOrange独立运行入口；LiftCube走[另一目录](../liftcube/README.md)。环境安装与资产见[PickOrange环境指南](../../docs/PICK_ORANGE.md)，模型和演示链接见[资源清单](../../docs/RESOURCES.md)。

This directory contains PickOrange-specific execution code. It does not use the LiftCube factory or its single-front model contract.

模型能力、配对RL结果与显存需求见[检验摘要](../../docs/PICKORANGE_VALIDATION.md)。本目录提供推理与评测代码；实验RL训练器和学习参数尚未发布。 / See the validation summary for results and memory requirements. This directory provides inference and evaluation code; experimental RL trainers and learned artifacts have not been released.

## 当前入口 / Available entry points

| 入口 / Entry | 用途 / Purpose |
| --- | --- |
| `service.py` | 固定GR00T N1.5任务权重、双RGB/q6/文本，独立模型进程；本地NPZ通信，不开放端口 / Separate local inference process |
| `service_n16.py` | wsagi GR00T N1.6独立后端；原生解码相对手臂动作、绝对夹爪动作 / Native relative-arm and absolute-gripper decoding |
| `service_pi05.py` | Aikwed π₀.₅任务权重及其LeRobot处理器，30 Hz策略 / Task checkpoint and its saved processors at 30 Hz |
| `service_smolvla.py` | Edge SmolVLA双视角任务权重、原统计与处理器，30 Hz；共享评测／审核，CPU验证 / Dual-view task service; CPU-validated integration |
| `openvla_oft.py` | 双相机、六关节OFT接口；需匹配的受训包，不是可运行的已受训策略 / Dual-view boundary; trained package required |
| `evaluate.py` | 单持续仿真进程，显式seed序列、原生成功与超时、reset前终态、逐步轨迹与逐回合结果 / Full-episode development runner |
| `audit.py` | CPU核对闭合轨迹、SHA、动作映射、连续性、查询索引和原生终态；不自动收入BC / Closed-trajectory audit |
| [官方特权教师](../../third_party/leisaac/scripts/datagen/state_machine/generate.py) | 已收录上游状态机采集入口；采用不同物理/状态写入机制，使用限制见下文 / Upstream privileged generation, with explicit caveats |

## 选用协议 / Selected development protocol

默认配置为[GR00T原生辅助](protocols/groot-native-assisted.json)，历史完整100回合结果为**52/100**。固定执行器另一协议为25/100；两组控制配置不同，成绩分开报告。

The selected development protocol previously achieved **52/100** under its own conditions. It is not a course score or proof of a single-factor improvement over another protocol.

- 任务权重`LightwheelAI/leisaac-pick-orange-v0@10f2014`，GR00T后端`4af2b62`；front+wrist各480×640、实测六关节motor、完整任务文本。
- 60Hz物理与动作、16步chunk全部执行、4次去噪；每回合60仿真秒，独立reset seed17–116。
- 保持重力与原生物理关节限位，模型有限目标不作软件裁剪；**每步上游夹爪effort助手读取物体位置/质量**。整个控制栈是特权辅助，真值不发送给网络。
- 三颗橙子满足原生XYZ盘区条件且机械臂回位才成功；丢弃auto-reset观测后显式reset到下一个seed。
- 历史100回合与新运行独立记录，配对小样本不合并到历史成绩。各后端实际验收见[任务对照](../../docs/TASKS.md)。

Images, feedback, requested targets and actual positions stay distinct. RGB is recorded at model queries; the pre-reset terminal RGB capture still needs runtime freshness validation. This source port does not claim arbitrary in-episode simulation recovery.

## 配置检查 / CPU-only check

工作目录为仓库根，Python3.11+及NumPy即可；不会导入模拟器 / From the repository root, without a simulator:

```bash
python course/pickorange/evaluate.py --check
```

## GR00T模型与仿真 / Model and simulator processes

先按[资源清单](../../docs/RESOURCES.md)下载完整权重，在独立环境安装固定GR00T源码，缓存其骨干与tokenizer。模拟器使用匹配的LeIsaac/IsaacLab及外部场景资产，不能借用LiftCube镜像的旧LeIsaac。两个进程必须共享同一个可写目录；在各自环境中从仓库根执行以下命令。

Prepare the pinned model backend and the independent simulator first. Both processes must see the same shared directory. These commands actually load a model and run simulation; use them only with approved compute resources and execution budgets.

模型进程先启动；`/outputs/pickorange/new-run`必须尚不存在。`/models`、`/backends`、`/outputs`均为使用者外部挂载，不是仓库自带资产。

```bash
export HF_HUB_OFFLINE=1
python course/pickorange/service.py --run-service \
  --model /models/pickorange/groot-10f2014 \
  --backend-root /backends/Isaac-GR00T \
  --run-dir /outputs/pickorange/new-run --device cuda:0 --wall-seconds 3600
```

服务会核验固定模型文件SHA、后端commit和严格加载结果。后端工作树有受管修改时拒绝；安装环境应与源码保持分离。检查`service-ready.json`后，在匹配的**仿真解释器**执行：

```bash
python course/pickorange/evaluate.py --run-simulation \
  --run-dir /outputs/pickorange/new-run --episodes 1 \
  --device cuda:0 --headless --enable_cameras --wall-seconds 1800
```

此处`--episodes 1`是单回合smoke，不是100回合成功率测试。验收通过后，使用新的目录重新启动两个角色，省略该参数才沿配置执行100回合；按预期墙钟时间设置两个进程预算。资源、有效运行窗口、RAM、磁盘与完整组合峰值由运行平台预检；脚本不预约资源、不重启容器，也不停止其他工作。

相机、动作映射、控制周期与任务源码会现场核对。结果目录独占，重复evaluator拒绝；失败保留`error.json`、partial与已完成前缀。当前版本**不提供自动resume**，不要删除claim、伪造ready或覆盖旧目录续跑。`STOP`仅请求这一对角色结束，不能用其结束无关任务。

CPU读盘审核使用另一个尚不存在的输出文件 / Audit closed trajectories with a fresh output file:

```bash
python course/pickorange/audit.py --run-dir /outputs/pickorange/new-run \
  --output /outputs/pickorange/new-run-audit.json
```

`report.json`是原生完成前缀，`audit.py`通过后才对外发布该前缀成绩。数据准入还需原始模型查询、真实终态RGB新鲜度、独立冷重放、完整动作与场景审查；成功轨迹不自动变成训练数据。RNG存档供审计，不宣称保存了隐藏物理/RTX状态。

`evaluate.py`复用[SwanLab共享记录层](../../docs/SWANLAB.md)，evaluator环境需SwanLab 0.10.3；默认在本run目录离线写回合与查询耗时曲线，只有显式在线配置才上传。模型service不创建重复run；RGB、HDF5和RNG存档均不上传，曲线不能替代审核。The evaluator alone records scalar curves, with media and artifacts retained locally.

退出时先保存并关闭曲线记录，再关闭环境和仿真应用。该收尾顺序已通过CPU测试，真实仿真集成仍待验证。 / Tracking closes before the environment and SimulationApp; this shutdown order has CPU test coverage, with simulator integration validation pending.

默认曲线名为`<协议名称>-eval`，可按SwanLab指南设置自定义名称。 / Run names default to `<protocol-name>-eval` and can be customized through the SwanLab settings.

## 其他后端 / Other backends

N1.6 与 π₀.₅ 各用[固定版本与权重](../../docs/RESOURCES.md)、独立推理环境和新输出目录。把上面模型命令替换为对应服务，并在**模型与仿真命令**中都传入同一 `--protocol`：

| 服务 / Service | 协议 / Protocol | 动作 / Actions |
| --- | --- | --- |
| `service_n16.py` | `course/pickorange/protocols/groot-n16-native-assisted.json` | 60 Hz、执行16步；原生处理器已把手臂相对量转成绝对motor目标，不再加q / Native processor already returns absolute targets |
| `service_pi05.py` | `course/pickorange/protocols/pi05-native-assisted-30hz.json` | 30 Hz、执行50步输出的前16步，每步保持2个物理tick；使用随权重保存的统计 / Execute 16 of 50 predictions, two physics ticks each, using saved statistics |

π₀.₅ 使用 LeRobot v0.6.0 对应的 Python 3.12+ 推理环境；`PI05_TOKENIZER_DIR` 指向使用者已获准下载的 PaliGemma tokenizer。不要更换模拟器 Python。两版 GR00T 可用相同 seed、60 秒时限、原生执行器和成功判据对照；π₀.₅ 的动作节奏不同，应单独报告。

Use the same protocol file for each service/evaluator pair. Keep the LeRobot v0.6.0 Python 3.12+ model environment separate from the simulator, and point `PI05_TOKENIZER_DIR` at the locally authorized PaliGemma tokenizer. Report the π₀.₅ action timing separately.

四个服务均核对各自固定后端commit，后端受管文件有修改时拒绝加载。 / All four services require their pinned backend commit and an unchanged tracked working tree.

### SmolVLA双相机 / Dual-view SmolVLA

使用Edge固定导出、LeRobot v0.4.2独立后端及[资源清单](../../docs/RESOURCES.md)中的SmolVLM配置／tokenizer。服务保留front、wrist两个真实输入和权重自带统计，不生成空相机或使用LiftCube统计。导出中的三个非活动兼容字段仅在后端不支持且值匹配时移除，其余未知字段拒绝。

```bash
export HF_HUB_OFFLINE=1
export SMOLVLA_VLM_DIR=/models/smolvlm2-500m-7b375e1
python course/pickorange/service_smolvla.py --run-service \
  --model /models/pickorange/edge-smolvla-71cf4a9 \
  --backend-root /backends/lerobot-v0.4.2 \
  --protocol course/pickorange/protocols/smolvla-native-assisted-30hz.json \
  --run-dir /outputs/pickorange/smol-new --device cuda:0 --wall-seconds 3600
```

仿真侧沿上文`evaluate.py`命令，使用同一新目录和同一`--protocol`；审核仍用`audit.py`。新协议为完整任务文本、60秒时限、30 Hz／50步预测执行16步、原生夹爪辅助。**这是CPU验收的新集成，尚无该协议的GPU／成功率成绩**，不继承历史固定执行器单例。每次预测清空模型缓存，不重seed；公共评测器只执行chunk前缀，reset后从新观测预测。

Use the same protocol and shared directory on the evaluator. This is a new CPU-validated integration, not a reproduction of the older fixed-actuator rollout. Both real views, original statistics and explicit chunk/reset handling are retained.

OFT 包须声明 front+wrist 的训练顺序、六维绝对 motor 目标、30步与 BOUNDS 统计，并提供同源的训练后核心、连续动作头、本体投影和统计。使用固定 SO101 后端，须在导入前通过 `--robot_platform SO101` 选择常量。普通 OpenVLA 权重、单相机或其他机器人导出不适用。尚无确认满足合同的公开权重，不提供 OFT 闭环命令。

The OFT boundary requires a matching trained dual-view package with six absolute motor targets, a 30-step continuous head and BOUNDS statistics. A vanilla backbone or single-view package cannot substitute for it.

## 特权教师采集 / Privileged teacher collection

两个任务的视觉／state源码集中在[脚本指南](../../docs/SCRIPTED_POLICIES.md)；PickOrange目前只有可复用的官方state教师，没有已验收的传统视觉规则策略。

仓库已含官方[状态机](../../third_party/leisaac/source/leisaac/leisaac/datagen/state_machine/pick_orange.py)和生成脚本，无需另下载教师模型。在匹配环境中，**仅作为辅助教师原型**运行：

```bash
python third_party/leisaac/scripts/datagen/state_machine/generate.py \
  --task LeIsaac-SO101-PickOrange-v0 --num_envs 1 --seed 17 \
  --device cuda:0 --headless --enable_cameras --record \
  --dataset_file /datasets/pickorange/new-teacher.hdf5 --num_demos 1
```

**重要限制：**此官方入口会关闭机器人重力、改变阻尼/夹爪控制、替换终止逻辑，且在setup、回位及`check_success()`中直接写关节状态。它与上面的GR00T原生辅助协议不同，状态机的“成功”不能作为固定物理下的BC成功标签；`--num_demos`只限制成功条数，失败重试可能持续，应由操作者另加墙钟上限。

The upstream generator teleports/blends joint states and changes physics/termination. Its success flag is not BC acceptance under the fixed evaluation physics. Independently replay and score the original physical trajectory before admitting data. Do not merge teacher-generator and model-assisted results.

尚无公开验收通过的自研高可靠PickOrange教师采集器或视觉脚本；历史单场景教师冷重放不等于本入口已通过。后续优先改造成只发送原生关节目标、保持物理与评分的教师，再做批量采集；不把官方直接写状态的机制隐藏在普通baseline中。

## 训练与其他模型 / Training and other models

本目录不含正式训练器。ACT/SmolVLA先走[数据准入与上游训练入口](../../docs/TRAIN_AND_EVALUATE.md)；其训练导出不能直接交给本GR00T服务。π₀.₅与OFT的仓库包装、资源和缺口见[四VLA接口](../../docs/VLA_INTERFACES.md)与[状态表](../../docs/TASKS.md)。新增后端要单独对齐双视角、单位、统计、chunk和实际执行频率。
