# GR00T N1.5 计数与混合奖励 / GR00T residual SAC

在 [GR00T 环境](../README.md)中下载匹配版本的 PickOrange 任务权重和后端。将 `GROOT_MODEL`、`GROOT_BACKEND` 和 `SO101_USD` 分别指向本地任务权重、固定版本的 Isaac-GR00T 源码和 SO-101 follower USD。在仓库根目录打开两个终端。模型端需要 GR00T、PyTorch、NumPy 及 safetensors。仿真端使用 Isaac Lab 2.3 和 LeIsaac 0.4。两个进程共用本地输出目录。源码会核对后端、资产与模型版本。

### 训练与续训 / Training

```bash
# 模型端
python -m course.pickorange.groot_sac.train --model "$GROOT_MODEL" --backend "$GROOT_BACKEND" --reward count --periodic-validation --out runs/groot-count
# 仿真端
python -m course.pickorange.groot_sac.simulate --mode train --out runs/groot-count --robot-usd "$SO101_USD" --headless --enable_cameras
```

计数组与混合组共用模型和训练参数。混合组将 `--reward count` 换成 `--reward hybrid`，两个终端都使用新的输出目录。每次决策生成并执行 16 个关节目标，物理频率 60 Hz。Actor 读取冻结 Eagle 的双路 4×4 视觉特征、关节位置和速度、基座动作块。训练 Critic 另读取 30 维场景状态。每步臂修正上限为 2，夹爪为 1，单位为标定电机坐标。原生动态夹爪力矩在仿真中启用。超时使用自动重置前的最后观测估计后续价值。

| 阶段 | 奖励 | Actor 目标更新数 | 回放容量 |
| --- | --- | ---: | ---: |
| 计数、混合奖励对照 | 分别选 count、hybrid | 20,000 | 100,000 |
| 混合奖励续训 | hybrid | 32,000 | 120,000 |

训练种子 1704。先收集 10,000 个动作块，随后预热 Critic 10,000 个动作块，每 4 个动作块进行一次梯度更新，batch 32。训练回合上限 60 秒。短测使用 `--qa`。运行目录下的 `latest.json` 指向完整续训点，配套的二进制文件保存冻结视觉特征。保留原目录，在新目录中运行后续阶段。

```bash
python -m course.pickorange.groot_sac.train --model "$GROOT_MODEL" --backend "$GROOT_BACKEND" --reward hybrid --resume "$FULL_CHECKPOINT" --target-updates 32000 --replay-capacity 120000 --out runs/groot-hybrid-32k
python -m course.pickorange.groot_sac.simulate --mode train --out runs/groot-hybrid-32k --robot-usd "$SO101_USD" --headless --enable_cameras
```

### 固定场景评测 / Evaluation

```bash
python -m course.pickorange.groot_sac.evaluate --model "$GROOT_MODEL" --backend "$GROOT_BACKEND" --weights "$ACTOR_WEIGHTS" --out runs/groot-eval
python -m course.pickorange.groot_sac.simulate --mode eval --out runs/groot-eval --robot-usd "$SO101_USD" --headless --enable_cameras
```

纯 GR00T 省略 `--weights`。评测已发布策略时，将 `ACTOR_WEIGHTS` 指向以下目录中的 `residual_actor.safetensors`。三组都需要同一份 GR00T N1.5 PickOrange 基座。

| 模型 / Model | 训练奖励 / Training reward | 已发布检查点 / Published checkpoints |
| --- | --- | --- |
| VLA-1 | 计数 / Count | [12k](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/tree/main/VLA/VLA-1/12k) |
| VLA-2 | 混合 / Hybrid | [12k](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/tree/main/VLA/VLA-2/12k)、[32k](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/tree/main/VLA/VLA-2/32k) |

复现两组 20k 对照时，分别加载对应训练的里程碑或完整存档。只加载自己生成或信任的 PyTorch 存档。默认使用种子 731000–731049，分别统计 3,600 和 7,200 个物理步内完成任务的场数。逐场记录写入 `episodes.jsonl`。

Each training or evaluation command pairs a model process with the Isaac simulator. The base uses native 16-step execution. The 32k run continues the hybrid 20k experiment with a larger replay. Retain both its numeric feature files and its training checkpoint to resume. Evaluation accepts the released actor without a training replay.
