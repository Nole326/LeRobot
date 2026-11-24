# PickOrange 残差训练与固定场景评测 / Residual SAC

本目录以冻结的 ACT 为基座。qpos 策略修正标定电机目标，EE 策略先求解绝对末端目标的逆运动学，再修正六个电机目标。前视、腕部 RGB 和实测关节位置输入策略。训练时，仿真端根据物体状态计算奖励。训练记录保存数值，双相机图像通过本机 Unix socket 传输。

### 环境与权重 / Setup

从仓库根目录运行。仿真端使用 [PickOrange 环境和资产](../../../docs/PICK_ORANGE.md)，包括 Isaac Lab 2.3、LeIsaac 0.4、SO-101 follower USD 和场景资产。EE 求解器在仿真端使用 SciPy 和随 Isaac 提供的 USD Python 绑定。模型端沿用 [ACT 环境](../act/README.md)。闭环短测使用 Linux、Python 3.11、PyTorch 2.7.0 (CUDA 12.8)、torchvision 0.22.0、LeRobot 0.4.1、NumPy 和 safetensors。两个进程须能访问同一个本地输出目录。下载权重时保留配置与归一化文件。

```bash
hf download Nole326/LeIsaac-SO101-PickOrange --include "ACT/*" "Residual/*" --local-dir models/pickorange
```

Residual-1 至 4 的训练命令、回放切换和固定场景配置见[历史 qpos 策略](history/README.md)。

### ACT 与残差训练 / Training

纯 ACT 的数据核验、依赖和训练入口见 [ACT 指南](../act/README.md)。例如，训练 qpos 至 160k 可同时得到 100k、120k 和 160k 推理权重。训练分段 EE 至 100k 可同时得到 80k 和 100k 权重。

```bash
python -m course.pickorange.act.train qpos datasets/pickorange/vla-gr00t-qpos/dataset.json --updates 160000 --output runs/bc-qpos
python -m course.pickorange.act.train ee datasets/pickorange/vla-gr00t-ee/dataset.json --updates 100000 --output runs/bc-ee
```

以下每行模型端命令与下一行仿真端命令配对，在两个终端中分别运行。`--out` 使用新目录。EE 的 `SO101_USD` 指向当前场景使用的 follower USD。

```bash
python -m course.pickorange.sac.train --config course/pickorange/sac/qpos.json --base models/pickorange/ACT/ACT-1/100k --out runs/qpos
python -m course.pickorange.sac.simulate --runs runs/qpos --headless --enable_cameras --device cuda:0

python -m course.pickorange.sac.train --config course/pickorange/sac/ee.json --base models/pickorange/ACT/ACT-2/80k --arm-fraction 0.25 --out runs/ee
python -m course.pickorange.sac.simulate --runs runs/ee --robot-usd "$SO101_USD" --headless --enable_cameras --device cuda:0
```

| 训练参数 | qpos | EE |
| --- | --- | --- |
| 冻结基座 | ACT-1 100k | ACT-2 80k，段末重置策略 |
| 原生动作与频率 | 关节目标，30 Hz | 末端目标经 IK，30 Hz |
| 训练夹爪力矩 | 固定 10 N·m | 环境动态力矩 |
| 训练臂残差幅度 | 满幅 | 满幅、半幅、四分之一幅 |
| 训练步数上限 | 默认 800，可选 400、600 或 1,000 | 800 |
| 奖励 | 连续三颗任务分档回报 | 同左 |

两种配置都先以冻结基座收集 1,024 条转移，再单独更新 Critic 10,000 次；Actor 目标为 100,000 次更新。每两次决策更新一次，batch 128，回放容量 150,000。两路 128×96 RGB 进入卷积编码器。截断回合在这两组 ACT 实验中按终点处理。随机种子、场景起始种子见 [qpos 配置](qpos.json)和 [EE 配置](ee.json)。

400 步组从头训练时使用 400 步上限。续训重建回放时，回合开始若池中不足 1,024 条，上限临时为 700 步，达到阈值后的下一回合恢复 400 步。

模型端加 `--smoke` 可运行缩小的闭环检查，验证预热时 Actor 保持不变，并记录两套网络更新前后的参数差。正式训练每 2,000 次 Actor 更新保存权重。部署使用 `policy-XXXXXX/`。回合结束时的 `latest.pt` 保存两个网络、目标网络、优化器和计数。用相同配置、原基座、新输出目录及 `--resume /path/to/latest.pt` 继续训练。ACT 图像回放仅在内存中，恢复后先重建回放池；重启会改变后续采样轨迹。

### 固定场景评测 / Evaluation

[评测配置表](../act/evaluation_profiles.json)给出报告中已发布模型的基座、权重和部署系数。例如在模型端运行 `python -m course.pickorange.act.evaluate --profile Residual-8-56k-quarter --out runs/eval-ee`，即可按训练幅度评测 EE 四分之一幅权重。自训权重使用以下显式路径。

模型端用 `course.pickorange.act.evaluate` 启动，仿真端用上面的 `sac.simulate` 连接同一目录。固定集使用 731000–731049，关闭探索，开启时间聚合。具名配置决定夹爪力矩、首帧同步和场景槽位。每场记录在 3,600 和 7,200 个物理步内是否三颗入盘且机械臂归位。纯 ACT 省略 `--residual`。EE 的仿真命令仍需提供机器人 USD。

```bash
python -m course.pickorange.act.evaluate qpos --base models/pickorange/ACT/ACT-1/100k --residual models/pickorange/Residual/Residual-5/80k --arm-fraction 0.25 --out runs/eval-qpos
python -m course.pickorange.sac.simulate --runs runs/eval-qpos --headless --enable_cameras --device cuda:0

python -m course.pickorange.act.evaluate ee --base models/pickorange/ACT/ACT-2/80k --residual models/pickorange/Residual/Residual-8/56k --out runs/eval-ee
python -m course.pickorange.sac.simulate --runs runs/eval-ee --robot-usd "$SO101_USD" --headless --enable_cameras --device cuda:0
```

| 报告对照 | 训练设置 | 评测设置 |
| --- | --- | --- |
| 纯 ACT | qpos 100k、120k、160k；EE 80k、100k | 选择对应基座目录 |
| qpos 部署幅度 | 同一满幅训练的 80k 权重 | `--arm-fraction` 依次取 1、0.5、0.25、0，夹爪均保持满幅 |
| qpos 回合上限 | 600、800、1,000 步分别训练 | 各取 80k，均设 `--arm-fraction 0.25` |
| EE 训练幅度 | Residual-6、7、8 分别以 1、0.5、0.25 训练 | 各取 56k，评测系数 1 保留各自训练上限 |
| EE 仅部署缩幅 | Residual-6 56k | 同一权重再测 `--arm-fraction 0.25` |

EE 原实验使用同进程的两个场景，完整轨迹策略在第一个槽位，分段策略在第二个槽位。完整轨迹基座可用同一份 EE 数据训练，命令与上面的 EE BC 相同，将模式改为 `ee-full` 并选择其 80k 检查点。重建该布局时，分别启动两个 EE 模型进程，再按原顺序将两个目录传给 `--runs`。单槽位也可运行；物理和渲染调度可能改变逐场结果。

The learner and simulator run in separate Python environments on the same Linux host. The training arm fraction sets the learned bounds. At evaluation the fraction multiplies the stored bounds for the five arm joints only. Use the episode journal for paired comparisons. For the GR00T reward experiment and its feature replay, see the [GR00T guide](../groot_sac/README.md).
