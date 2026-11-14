# Evo-1：PickOrange 小规模 VLA 基线实验记录 / Evo-1 small-scale baseline log

本页记录 Evo-1（InternVL3-1B + flow-matching 动作头）在 PickOrange 上的 BC 基线实验，截至 **2026-11-13**。评测入口、双视角契约与成功判据见[PickOrange运行指南](../course/pickorange/README.md)与[检验摘要](PICKORANGE_VALIDATION.md)；本页结果来自独立实验，**不是课程正式评分**，也不与 GR00T、π₀.₅、SmolVLA 的历史成绩合并分母。

This page records a small-scale Evo-1 behavioural-cloning baseline (frozen or unfrozen InternVL3-1B backbone plus a flow-matching action head). It reports a **negative result**: all completed evaluation episodes failed the full-task predicate. Numbers are experiment notes, not course scores.

## 范围与数据 / Scope and data

- 数据来自本仓库自研特权教师采集流程（[控制脚本说明](../course/pickorange/expert/README.md)），每条演示都经过独立回放准入；观测只含 front/wrist 两路 RGB 与实测六维，不回放物体真值。
- 两套数据：**22 条 / 29,846 帧**（早期集合）与 **56 条 / 75,019 帧**（在 22 条基础上追加 34 条）。30 fps，480×640 双视角，六维 motor 动作，无成功标签。
- 上游另有两套各 100 条的公开演示（30 Hz 与 60 Hz 两种动作时钟），混用前需对齐动作频率、关节坐标与时间戳，本页结果未使用。

Dual-RGB, q6 and full task text are the only policy inputs. All demonstrations passed an independent replay admission step. Two local sets of 22 and 56 episodes were used; the published 30 Hz and 60 Hz collections were not mixed into these runs.

## 训练配置 / Training setup

| 项目 / Item | 取值 / Value |
| --- | --- |
| 策略 / Policy | Evo-1，InternVL3-1B，flow-matching 动作头 |
| chunk / 推理去噪 / chunk and denoising | 50 / 32 |
| stage1 | 冻结视觉语言主干，只训动作头，batch 8 |
| stage2 | 解冻视觉语言主干（四个 `finetune_*` 全开），约 2.5 倍单步耗时 |
| 学习率 / LR sweep | 5e-6、1e-5、3e-5、1e-4（含 warmup 与 cosine 衰减） |
| 主要运行 / Main runs | A4：56 条、14,000 步、stage1；S2：从 A4 解冻再训 2,000 步；A6：56 条、1,000 步、lr 3e-5 |

训练损失（归一化 chunk MSE）在 0.022–0.048 之间；换算成归一化 RMSE 约 0.15，相当于 7–8 个 motor 单位。单条 episode 过拟合实验（400 步、lr 1e-4）只看了约 0.1 个 epoch，损失反而升到 0.045–0.057，说明该预算下不足以拟合单条演示。

## 评测协议 / Evaluation protocol

- 完整回合、60 仿真秒、`hold_ticks=2`（50 步策略目标按 30 Hz 执行，与演示动作时钟一致），初始状态取全零关节位姿，成功判据为三颗入盘且机械臂回位。
- 每回合约 225 次模型查询；单回合墙钟 128–202 秒（含仿真），模型查询均值约 0.12–0.15 秒。

Full episodes only; each policy target is held for two physics ticks so the 30 Hz action clock matches the demonstrations. Success requires all three oranges on the plate plus the arm at rest.

## 结果 / Results

| 运行 / Run | 数据 / Data | 评测 seed | 完整成功 / Success |
| --- | --- | --- | ---: |
| A2 | 19 条 | 0 | 0/1 |
| A3 | 22 条、3,000 步 | 0 | 0/1 |
| A4 | 56 条、14,000 步 | 0,1,2 | 0/3 |
| A4 中间存档 | 56 条、8k/12k 步 | 0 | 0/1、0/1 |
| S2 | 56 条、解冻 500 步 | 0,1,2 | 0/3 |
| S2 | 56 条、解冻 2,000 步 | 0,1,2 与 1123 | 0/3、0/1 |
| A6 | 56 条、1,000 步、lr 3e-5 | 0 与 1123 | 0/2 |
| 单条过拟合 | 1 条、400 步 | 1123 | 0/1 |

合计约 20 个完整回合，**0 次成功**；最好的一次也只出现单颗橙子被短暂移动。所有回合都在 60 秒时限处 truncated，没有 terminated。

Twenty complete episodes across eight checkpoints, three data sets, four learning rates and both training stages produced **zero** full-task successes. Partial object progress is not success and is not counted.

## 现象 / Behaviour

机械臂在开局 2 秒内抬起并移动到橙子上方，约 4.8 秒后停在某个姿态（肘关节接近限位），夹爪始终张开（motor 35–50，从未闭合），三颗橙子的世界坐标在整回合内不变。跟踪日志显示关节位置紧跟随动作命令，因此“手臂不动”是策略输出本身停住，而非执行器或通信问题。

The arm lifts, aims above the oranges, then freezes near the elbow limit with the gripper open. Joint tracking follows the commanded actions closely, so the stall comes from the policy output, not from the actuator or the transport.

## 诊断 / Diagnostics

按序列排除的可能原因：

1. **评测链路**：用同一脚本重放一条通过准入的教师演示，得到 `success=True`（3,103 ticks），动作换算、30 Hz 执行、成功判据与 reset 全部正常。
2. **观测域**：导出仿真首帧与数据集帧对比，双相机画面一致（均值 111/144 对 108/160），无通道、缩放或单位问题。
3. **动作单位**：rad 与 motor 的线性映射与课程 `joint_contract` 一致；动作管理器 `scale=1`、`offset=0`，直接发送 rad 正确。
4. **首帧暗图**：强制首块使用全黑图像，失败模式不变。
5. **指令文本**：修正策略服务中的任务文本使其与演示一致，失败模式不变。
6. **采样噪声**：同一请求重复 6 次，逐关节标准差最高 13 个 motor 单位；取 4 次动作块平均后仍悬停，排除采样方差。
7. **离线精度**：在通过准入的演示帧上，动作头预测的 chunk MAE 为 1.9–9.8 个 motor 单位，明显优于“保持当前动作”基线的一部分检查点，预测随观测变化，视觉通路正常。

因此离线拟合并不算差（损失对应约 7–8 个单位误差），但闭环中前几块动作开始偏离演示流形后，flow-matching 头的输出收敛到“保持姿态”，误差随回合累积。

A replay of an admitted demonstration through the same harness succeeds, so the evaluation path is sound. Camera domain, action units, the dark first frame, the task string and sampling noise were all ruled out. Offline chunk MAE on admitted demonstrations is 1.9–9.8 motor units, yet closed-loop rollouts collapse to a stationary pose after a few chunks — the classic compounding-error failure of a fine-tuned action head on few, long demonstrations.

## 结论与后续 / Conclusion and next steps

- 在当前演示规模（56 条、约 7.5 万帧）与训练预算下，Evo-1 动作头没有学到可闭环复现的抓取策略；失败不是评测脚本、图像通路、动作单位或指令文本造成的。
- 数据侧的主要欠缺是抓取事件的多样性：每条演示约 1,500 步，大量帧是往返与悬停，真正闭合夹爪的片段很少。
- 后续方向：使用上游两套各 100 条的演示（先对齐 30 Hz 与 60 Hz 动作时钟再混合）、改用 SmolVLA 双视角任务权重作为主基线、或在闭环中加入人工纠错式的补充数据。
- 本仓库不含 Evo-1 训练器与权重；复现实验需要外部训练环境，评测按[运行指南](../course/pickorange/README.md)的协议执行。

The negative result is a property of the policy, not of the harness. The next steps are larger and more diverse demonstrations, the dual-view SmolVLA baseline, and closed-loop correction data.

## 数据与复现 / Data and reproduction

演示数据不随仓库分发，按[控制脚本说明](../course/pickorange/expert/README.md)在仿真环境中重新采集；权重、数据集与运行日志留在外部存储，公开入口只写身份、参数与验收级别。

### 采集 / Collection

```bash
python course/pickorange/expert/collect.py --run-simulation \
  --out <demos_dir> --start-seed <first_seed> --count <episodes> --max-attempts <attempts>
```

- 本页使用的种子段：`18–34`、`35–50`、`51–54`、`61–65`、`1100–1118`、`1123–1142`、`2000–2014`；通过率约 55%–65%，单次尝试约 6–7 分钟，每次成功约 0.5 GB 无损双视角视频。
- 每个种子先跑特权教师，再做一次独立回放；只有回放同样满足“三颗入盘且回位、55 秒内完成”的样本才进入演示集合。
- 22 条集合对应 `18–54` 段，56 条集合追加 `61–1142` 段。仓库不保存原始视频，也不把它们当作 BC 标签直接使用。

Each candidate seed is admitted only when an independent replay reproduces the full task within 55 seconds. Roughly 55%–65% of attempts pass; one accepted attempt yields about 0.5 GB of lossless dual-view video that is kept outside the repository.

### 转换为 LeRobot v3 数据集 / LeRobot conversion

原始回放录像是双视角无损视频加 `demonstration.npz`（`action`、`joints` 各 `(T, 6)`）。转换为 LeRobot v3 目录时保持：`fps=30`；`observation.images.front`、`observation.images.wrist` 为 `(480,640,3)` 视频，`observation.state`、`action` 为 `float32 (6,)` motor 归一化单位；任务文本 `Pick three oranges and place them on the plate, then return the arm to rest.`；`repo_id` 由使用者自定。转换脚本属于外部训练环境，不在本仓库内。

Converted datasets keep 30 fps, the two 480×640 video streams, six-dimensional motor state/action and the full task string. The conversion utility lives with the external training environment, not in this repository.

### 训练 / Training

| 阶段 / Stage | 关键设置 / Settings |
| --- | --- |
| stage1 | 冻结视觉语言主干，只训 flow-matching 动作头；chunk 50、去噪 32、图像 448×448、batch 8 |
| stage2 | 解冻视觉语言主干，学习率 1e-5、warmup 100 步，单步耗时约为 stage1 的 2.5 倍 |
| 预算 / Budget | 56 条约 14,000 步（stage1）或 2,000 步（stage2）；多卡不在本页范围内 |

Training uses the checkpoint and optimizer state of the previous stage; a new run resets the step counter. Learning rates above 3e-5 were unstable on the 56-episode set, losing accuracy compared with 5e-6–1e-5.

### 评测 / Evaluation

评测使用本仓库的完整回合运行器，模型侧通过 NPZ 请求/响应提供 `(chunk, 6)` motor 动作：

```bash
python course/pickorange/evaluate.py --run-simulation --run-dir <run_dir> \
  --episodes <n> --wall-seconds <budget> --min-free-gib <gib>
```

- `episodes` 只取协议 `scene_seeds` 的前缀；动作按 30 Hz 保持两个物理步执行，与演示时钟一致。
- 本页所有回合都保存逐回合轨迹与终态截图，失败结果同样保留；成功率只按完整回合统计。

Evaluation follows the repository's full-episode runner; the policy service returns one chunk of six-dimensional motor actions per query, and each target is held for two physics ticks. Failed episodes are retained rather than replaced.

### 后端与接口契约 / Backend and interface contract

**后端版本与代码入口 / Backend and code entry**

- 训练与推理使用 Evo-1 官方仓库 [`MINT-SJTU/Evo-1`](https://github.com/MINT-SJTU/Evo-1) 的 `evo1-lerobot` 分支（LeRobot 训练栈移植版）。记录时该分支最新提交为 `fb177e4add5c7d803715be3af721fb08b1fb3f0f`；本地代码根为 `<Evo-1>/evo1_lerobot/lerobot`，训练入口 `python -m lerobot.scripts.lerobot_train`。该分支把 LeRobot 源码随仓库分发（版本由 `importlib.metadata` 读取，未安装时显示 `unknown`），依赖清单见 `evo1_lerobot/requirements.txt`（`torch==2.5.1+cu124`、`transformers==4.53.3`、`flash-attn==2.8.3`、`torchcodec==0.1.1`）。
- 实际训练机为 AMD ROCm 7.2 / Python 3.12 / `torch 2.12.0`，未安装 flash-attn（`use_flash_attn=false`）；本地对数据集视频解码加了 PyAV 兼容补丁（原路径依赖 torchcodec，ROCm 下缺 `libnvrtc.so.13`），补丁只影响读盘，不改变策略加载、预处理或动作解码。推理侧复用同一份 `evo1_lerobot` 代码与 `EVO1Policy.from_pretrained(..., strict=True)`。
- 视觉主干为 `OpenGVLab/InternVL3-1B`（revision `4415a3b810e636d11dfa86b0e9ba40bb00535aa8`），动作头为 flow-matching（8 层、hidden 1024、chunk 50、去噪 32 步）。任务权重不随仓库分发；上传 Hugging Face 后在此补链接。

**观测与动作接口 / Observation and action interface**

| 项目 / Item | 约定 / Contract |
| --- | --- |
| 相机键 / Camera keys | `observation.images.front`、`observation.images.wrist`，均 480×640×3 RGB uint8；顺序固定 front→wrist，服务请求键名 `front`、`wrist` |
| 关节顺序 / Joint order | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| 单位 / Units | 六维 motor 归一化单位：前五维 ±100，夹爪 0–100；与课程 `joint_contract.rad_to_motor` 一致 |
| 动作类型 / Action type | 绝对关节目标（非增量），与 `observation.state` 同一 motor 坐标系；夹爪 0 为闭合、100 为张开 |
| 任务文本 / Task text | `Pick three oranges and place them on the plate, then return the arm to rest.` |

**归一化 / Normalization**

- `configuration_evo1.py` 的 `normalization_mapping` 为 `VISUAL: IDENTITY`、`STATE: MIN_MAX`、`ACTION: MIN_MAX`；state/action 用训练数据集的 `meta/stats` 做 min–max 归一化与反归一化，推理时由 `service_evo1.py` 把 `LeRobotDataset(<dataset_root>).meta.stats` 传给 `make_evo1_pre_post_processors`，所以 `--dataset-root` 必须与训练快照一致。
- 图像不做数值归一化（`VISUAL: IDENTITY`）；VLM 侧把输入转 PIL、缩放到 448×448，再按 ImageNet 均值/方差归一化并以 bf16 编码。
- 动作在模型内零填充到 24 维，输出时截回 6 维；服务与仿真之间只传 motor 单位，rad↔motor 换算由课程 `joint_contract`/`robot_utils` 完成，动作管理器 `scale=1`、`offset=0`，没有额外缩放。

**推理命令与执行节奏 / Inference command and cadence**

训练（示例，路径为占位符）：

```bash
cd <Evo-1>
PYTHONPATH=$PWD/evo1_lerobot python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/pickorange_teacher --dataset.root=<dataset_root> --dataset.video_backend=pyav \
  --policy.path=<previous_ckpt>/pretrained_model --policy.training_stage=stage1 --resume_pretrain=true \
  --policy.finetune_action_head=true --policy.optimizer_lr=5e-6 --batch_size=8 --steps=14000 \
  --output_dir=<output_dir>
```

推理服务（本页成绩使用外部服务进程 + 本地运行器；服务也可由 `service_evo1.py` 直接提供）：

```bash
EVO_CKPT=<ckpt>/pretrained_model EVO_DATA=<dataset_root> EVO_PORT=5555 \
  python <Evo-1>/evo1_service.py &
```

- 模型每次查询预测 **chunk 50**，整块执行 **50 步**（开环）；每个目标保持 **2 个 60 Hz 物理步**，即 30 Hz；回合 **60 仿真秒 = 3,600 物理步 = 36 块**；单次查询均值 0.12–0.15 秒。
- 初态为全零关节位姿，`warmup_ticks=10` 后重置再预测；夹爪辅助 `dynamic_reset_gripper_effort_limit_sim` **保持开启**（未加 `--no_dynamic_gripper`），与仓库协议一致，因此本页成绩属特权夹爪辅助设置。
- 本页成功率由与 `evaluate.py` 同协议的本地运行器统计：完整回合、同一成功判据、失败回合保留；仓库 `service_evo1.py` 走同一 NPZ 契约，可由 `evaluate.py` 驱动。

### 服务接入 / Service adapter

仓库补充了[`service_evo1.py`](../course/pickorange/service_evo1.py)：沿用 `service.py` 的 NPZ 请求/响应契约、请求白名单与原子写盘，加载 Evo-1 权重及其随训练保存的 normaliser 统计。任务权重不随仓库分发，因此它不核对公开 revision，`service-ready.json` 标记 `experimental: true` 并记录权重文件 SHA-256、代码 commit（或 `unpinned`）与数据集根目录；协议只接受 30 Hz／chunk 50 的动作时钟。该服务用于本地复现与接口审计，不作为协议基准成绩。

The supplemental adapter reuses the shared NPZ contract and request validation but cannot verify a public revision; its ready file marks the run experimental and records weight digests, code commit and the data root.
