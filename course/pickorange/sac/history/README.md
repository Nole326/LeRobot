# qpos 残差策略的训练与评测

Residual-1 至 3 冻结 ACT-1 80k。Residual-4 冻结 ACT-1 100k，使用视觉放置信号衔接抓放和归位。所有策略共用[固定场景评测入口](../../act/evaluate.py)和[权重目录](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/tree/main/Residual)。模型与仿真环境的依赖、下载目录和启动方式见[安装指南](../README.md)。模型环境另需 Pillow，Residual-4 的视觉判定使用 OpenCV。

| 发布权重 | 训练输入与奖励 | 原评测配置名 | 力矩、重置与场景 |
| --- | --- | --- | --- |
| Residual-1 24k | 双路 RGB，混合奖励 | `Residual-1-24k` | 动态力矩，首帧同步，多场景 |
| Residual-2 24k | RGB 加边缘，计数奖励 | `Residual-2-24k` | 同上 |
| Residual-3 26k | RGB 加边缘，混合奖励 | `Residual-3-26k` | 动态力矩，环境原生重置，独立单场景 |
| Residual-4 100k | 双路 RGB，分档奖励与放置后归位 | `Residual-4-100k` | 固定 10 N·m，首帧同步，独立单场景 |
| Residual-5 90k | 连续三颗，800 步训练上限 | `Residual-5-90k-full` | 固定 10 N·m，首帧同步，满幅，双场景 |

## Residual-1 至 3

原实验同时训练 RGB / RGB 加边缘与计数 / 混合奖励的四种组合。每种组合先使用自己的 120,000 条回放池，共用 ACT-1 80k。预收集 5,000 条转移后进行 5,000 次 Critic 预热更新，再联合训练 Actor 和 Critic。每四次决策更新一次，batch 为 256，超时保留 bootstrap。模型进程在完整回合边界保存权重、优化器与计数，各组切换共享池的更新位置见[配置](cohort_profiles.json)。

在模型环境启动四组训练。在仿真环境的另一个终端设置相同的 `root`，运行最后一条命令。两个环境从仓库根目录启动。

```bash
root=runs/cohort
for source in rgb-count rgb-hybrid edges-count edges-hybrid; do
  python -m course.pickorange.sac.history.cohort --profile "$source" \
    --base models/pickorange/ACT/ACT-1/80k --out "$root/residual-$source" &
done

python -m course.pickorange.sac.history.cohort_sim --outs \
  "$root/residual-rgb-count" "$root/residual-rgb-hybrid" \
  "$root/residual-edges-count" "$root/residual-edges-hybrid" --headless --enable_cameras
```

四组私有阶段结束后，保留各自的输出目录。下面三条命令各占用一个终端。先在模型环境启动共享池，再在另一模型终端启动采集器，最后在仿真环境启动场景。采集器读取同一父目录下的四份训练存档，每条转移记录两种奖励。

```bash
python -m course.pickorange.sac.history.pool runs/cohort/replay.sock
python -m course.pickorange.sac.history.cohort --profile rgb-count --shared-collect runs/cohort/replay.sock \
  --base models/pickorange/ACT/ACT-1/80k --out runs/cohort/collector
python -m course.pickorange.sac.history.cohort_sim --outs runs/cohort/collector --headless --enable_cameras
```

每个来源累计至少 5,000 条后，在采集目录创建 `stop-request` 空文件，让采集器在回合结束时退出。保持共享池运行，按下面的命令恢复模型。四个输出目录的 `ready.json` 都显示续训的 `start_cursor`（大于零）后，再启动四场景仿真进程。共享池总容量为 500,000，每批一半来自本组，另一半分配给其余三组。训练进程接手写入。回放在内存中，四组私有池的旧转移不随模型存档迁移。

```bash
root=runs/cohort
for source in rgb-count rgb-hybrid edges-count edges-hybrid; do
  python -m course.pickorange.sac.history.cohort --profile "$source" --resume \
    --shared-train "$root/replay.sock" "$source" --actor-updates 40000 \
    --base models/pickorange/ACT/ACT-1/80k --out "$root/residual-$source" &
done
```

## Residual-4

[放置归位配置](placement.json)列出随机种子、回放和训练预算。每个场景最多 2,700 个物理步；单次抓放和归位阶段分别最多 1,200 和 270 步。放置触发使用双相机与夹爪观测，物体真值用于训练奖励和评分。预收集 1,024 条，Critic 预热 10,000 次，batch 为 128，回放容量 150,000。双路 128×96 的 uint8 回放图像保存在 GPU 上，仅图像存储约占 10.3 GiB。模型端与仿真端分别运行：

```bash
python -m course.pickorange.sac.history.skill --base models/pickorange/ACT/ACT-1/100k --updates 100000 --out runs/placement
python -m course.pickorange.sac.history.skill_sim --outs runs/placement --headless --enable_cameras
```

## 权重导出与固定场景评测

对自己训练的 Residual-1 至 4，可用 `python -m course.pickorange.sac.history.export <训练存档.pt> <新权重目录>` 导出 Actor、编码器和部署配置。对发布权重直接使用表中的配置名。以下两条命令分别在模型和仿真环境运行，默认评测相同的 50 个种子。模型端加 `--check-actions 4` 可检查首个场景的四次决策。

```bash
python -m course.pickorange.act.evaluate --profile Residual-4-100k --out runs/eval-placement
python -m course.pickorange.sac.simulate --runs runs/eval-placement --headless --enable_cameras
```

Residual-5 90k 原评测的第一槽为发布的 90k 权重，第二槽为 600 步组 92k 满幅权重。持有原伴随权重时，用下面三条命令分别启动两个策略和一个仿真。公开模型仓库提供第一槽权重；第二槽路径应指向含配置和 Safetensors 的原检查点目录。双槽短测可以验证协议，历史逐场成绩见[评测日志](https://huggingface.co/Nole326/LeIsaac-SO101-PickOrange/tree/main/logs)。

```bash
python -m course.pickorange.act.evaluate --profile Residual-5-90k-full --out runs/eval-90k
python -m course.pickorange.act.evaluate qpos --base models/pickorange/ACT/ACT-1/100k \
  --residual "$COMPANION_92K" --gripper-effort fixed-10Nm --synchronized-reset --required-slots 2 --out runs/eval-92k
python -m course.pickorange.sac.simulate --runs runs/eval-90k runs/eval-92k --headless --enable_cameras
```
