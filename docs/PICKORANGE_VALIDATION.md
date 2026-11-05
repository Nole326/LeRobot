# PickOrange：VLA检验与资源参考 / VLA validation and resource notes

本页汇总截至 **2026-11-05** 的PickOrange开发实验，供模型选择和计算预算参考。运行命令见[评测指南](../course/pickorange/README.md)，权重与固定版本见[资源清单](RESOURCES.md)，LiftCube状态见[任务对照](TASKS.md)。以下采用开发评测协议，不是课程正式评分。

This snapshot covers PickOrange development experiments through **2026-11-05**, not course benchmark scores. Follow the linked guides for evaluation commands, pinned assets and the separate LiftCube track.

## 协议与验收范围 / Protocol and acceptance scope

- 输入为front、wrist两路RGB（各480×640）、实测q6及完整任务文本。网络不接收物体真值；原生夹爪effort助手使用物体位置／质量，因此**整个控制栈属于特权辅助**，不作为无特权视觉baseline。
- 保持原生重力、关节物理限位、动作映射及“三颗入盘且回位”判据，时限60仿真秒。GR00T N1.5/N1.6为60 Hz动作、16步执行、4步去噪；π₀.₅任务导出为30 Hz动作、50步预测执行前16步、10步去噪，每个目标保持两个物理tick。
- 完整结果经过逐回合轨迹／SHA、原模型响应到实际目标、连续性、查询索引、原生终态及模型RNG握手审核。物理初态可配对；RGB不承诺逐像素一致，回合长度不同也会使模型随机流分岔。
- 成功终态RGB新鲜度与独立冷重放仍是另行的数据准入项；成功轨迹不自动成为BC演示。软件检查、GPU前向、物理闭环和可靠任务能力分别记录。

Policies receive dual RGB, measured q6 and task text, not object truth. The upstream gripper-effort helper does use truth, so these are **oracle-actuator-assisted** results. Native physics, limits and full-task success remain unchanged. GR00T and PI05 have different action clocks. Complete-episode audits do not by themselves admit trajectories for BC.

本页成绩来自已关闭并审核的独立实验。仓库中的可移植入口是对应接口的整理版本；后端版本检查及SwanLab收尾修正另做CPU回归，**尚未重新运行修正后代码的GPU整组评测**。不要把历史成绩归到一次未执行的新运行。

Results below come from completed independent runs. The portable source includes CPU-checked backend guards and tracking cleanup fixes; the revised code has not been re-benchmarked on GPU.

## 四类VLA的任务能力 / Task capability by VLA family

模型完整revision、视角和处理器来源统一维护在[资源清单](RESOURCES.md)。GR00T N1.5和N1.6分别列出。

| 模型 / Model | 已有验证 / Validation | 完整任务结果 / Task result | 当前用途 / Intended use |
| --- | --- | --- | --- |
| GR00T N1.5，Lightwheel任务权重 | 严格加载、原处理器、双RGB闭环及轨迹审核 | seed17–26：**7/10** | 可用开发基线；Residual SAC另见下节 / Development baseline |
| GR00T N1.6，wsagi任务权重 | 独立后端；相对臂／绝对夹爪解码；同初态、时限与执行器对照 | seed17–26：**3/10** | 版本对照；不据十场结论评价整个模型家族 / Version comparison |
| SmolVLA，Edge／wsagi任务权重 | 历史严格加载、GPU前向与双RGB闭环；采用旧30 Hz固定执行器协议 | Edge一次完整成功；wsagi所测长回合超时。**无分组成功率** | Edge双视角服务已接入并通过CPU检查；新原生辅助协议闭环待验 / New Edge service is CPU-checked; rollout pending |
| π₀.₅，Aikwed任务权重 | 严格加载、原分位数处理器、双RGB物理闭环与全组审核 | seed17–26：**0/10** | 已有任务微调导出，但当前协议下能力未通过 / Task checkpoint, not an accepted capable baseline |
| OpenVLA-OFT | 通用核心／处理器及SO101软件合同检查；双视角接口缺项拒绝 | **无可报告任务成功率** | 缺匹配受训核心、连续头、本体投影与统计 / Trained matching package required |

π₀.₅十场均完成60秒时限，部分有单颗入盘，但没有三颗同时入盘，失败不只发生在回位。该导出训练数据的动作目标范围还存在回位覆盖缺口；它提示数据／部署分布需要诊断，并不证明模型数学上不能外推。不能通过放宽评分或更换不匹配统计来改善成绩。

The PI05 task export already underwent task training, but failed this protocol. Partial object progress is not full-task success. Its training-target coverage also warrants review; no scoring or normalization workaround is implied.

历史N1.5的**52/100原生辅助**和**25/100固定执行器**是另外两组协议结果；它们与本页十场／三十场分开，不合并分母，也不解释为单因素因果差异。SmolVLA历史结果同样不能并入当前GR00T协议比较。

Historical 100-episode protocols and SmolVLA's older rollout settings are separate datasets, not additional samples for the comparisons above.

## Residual SAC小训练与配对 / Residual SAC pilot

冻结N1.5任务基座，仅训练外置六维残差；在原生解码的绝对motor目标上修正，边界为手臂±2、夹爪±1 motor单位。四个训练seed127–130，共6,641物理步、49轮actor／双Q更新；训练成功不计评测。奖励使用原生成功信号与盘区计数势函数，物体真值只用于奖励和既有辅助执行器，不输入策略或critic。

The VLA stays frozen. A small external actor learns bounded corrections to decoded motor targets. Training used four separate seeds and 49 SAC update rounds; training outcomes are not evaluation scores.

旧开发十对seed17–26为7/10→8/10。随后冻结同一最终actor，不加训，使用本项目此前训练及评测未用的seed2301–2330做三十对比较；不宣称可追溯并排除上游预训练全部场景。

| 指标 / Metric | 基座 / Base | 最终残差 / Residual |
| --- | ---: | ---: |
| 完整成功 / Complete success | 19/30 | 21/30 |
| 全部物理步 / Total physics steps | 61,438 | 52,931 |
| 模型查询 / Model queries | 3,848 | 3,318 |
| 13场共同成功的平均物理步 / Mean steps, 13 common successes | 949.85 | 1,030.46 |
| 模型＋仿真峰值 / Combined sampled peak | 12.03 GiB | 12.03 GiB |
| 回合物理步／墙钟秒 / Episode physics steps per wall second | 11.52 | 11.64 |
| 平均查询 / Mean query | 171.18 ms | 172.62 ms |

改善8场、退步6场、共同成功13场、共同失败3场。净增6.67个百分点，配对bootstrap 95%区间为−16.67至＋30.00个百分点，McNemar精确p=0.7905。**尚不能证明稳定提升**；共同成功场景反而平均多80.62步。总步数下降部分来自失败／成功构成改变，不能直接称动作更快。

Eight seeds improved and six regressed. The paired interval includes zero; there is no demonstrated stable success-rate gain, and common-success episodes took more steps on average. The prior ten pairs are not pooled with these thirty pairs.

两组共60回合、7,166次模型查询及本地SwanLab数值记录已对账。所测硬件为RTX 4090级24 GiB显存配置；吞吐包含模型、仿真和I/O，不是纯GPU计算速度。

All 60 episodes and 7,166 queries were audited. Measurements used a 24 GiB RTX 4090 configuration. Throughput includes simulation and I/O.

## RL显存选型参考 / RL memory reference

峰值为**同一NVML采样中模型与仿真进程的显存之和，再取最大值**，不含其他任务，也不把两个时刻的独立峰值相加。1 GiB=1,024 MiB。采样可能漏过短峰；这些是特定小预算的工程测量，不是长训练的显存保证。

Combined peaks sum the model and simulator in each NVML sample. Sampling can miss brief peaks. Do not extrapolate to larger batches, replay buffers, parallel environments or full-parameter training.

除另注的小训练／128步组，均为单环境64物理步、两轮真实参数更新。SAC包含actor和双Q更新；PPO和SAC工程探针的策略分布不同，不是严格算法消融。不同设备批次不作吞吐排名。

| 模型与更新范围 / Trainable scope | 算法／预算 / Algorithm and budget | 模型＋仿真峰值 / Combined GiB |
| --- | --- | ---: |
| N1.5冻结基座＋归一化残差 | PPO，两个独立设备批次 / two device configurations | 12.10–12.12 |
| N1.5完整动作头＋value，骨干冻结 | PPO，流程短测 / smoke | 28.09 |
| N1.5动作头＋value＋顶部8／10个文本层 | PPO，流程短测 / smoke | 34.56 / 35.98 |
| N1.5冻结基座＋归一化或motor残差 | SAC，分别验证的流程短测 / separate smokes | 12.17 |
| N1.5冻结基座＋motor残差 | SAC，4回合／49轮 / pilot | 12.06 |
| N1.5动作头q/k/v LoRA32，约655万参数 | SAC，流程短测 / smoke | 12.64 |
| N1.6冻结基座＋motor残差 | SAC，流程短测 / smoke | 13.30 |
| π₀.₅ SO101域基座＋外置残差 | PPO，流程短测 / smoke | 14.18 |
| π₀.₅ SO101域基座动作专家LoRA | PPO / SAC，流程短测 / smoke | 14.24 / 14.18–14.24 |
| π₀.₅ SO101域基座动作专家Adapter | PPO / SAC，流程短测 / smoke | 14.28 / 14.27 |
| π₀.₅ SO101域基座动作专家及投影直接更新 | PPO / SAC，流程短测 / smoke | 22.01 / 22.18 |
| π₀.₅ Aikwed任务权重＋motor残差 | SAC，128物理步／2轮 / smoke | 15.78 |
| OFT通用骨干＋新初始化头／投影，外置残差 | PPO，工程探针 / engineering probe | 22.09 |
| OFT通用骨干＋新初始化头／投影，语言骨干LoRA | SAC，工程探针 / engineering probe | 22.22 |
| OFT通用骨干＋新初始化头／投影，语言骨干Adapter | SAC，工程探针 / engineering probe | 22.31 |
| OFT通用骨干冻结，仅更新新头／投影 | SAC，工程探针 / engineering probe | 24.58 |

**OFT上述数字不是受训SO101策略的资源或能力验收。** 它们使用新初始化组件，仅验证训练工程。公共OFT推理接口继续拒绝缺少匹配受训包的加载，不提供随机头运行替代。

OFT's random-initialized components were used only in isolated memory probes. They are not a trained SO101 policy, and the public inference loader still requires a matching trained package.

π₀.₅“SO101域基座”与“Aikwed任务权重”是不同模型图／处理器，不互换成绩。SmolVLA尚无本摘要范围内已审计的RL峰值；四种模型均没有可引用的整模型全参RL完整测峰。28.09 GiB一行是完整**动作头**，不是整个GR00T全参。

The two PI05 checkpoints are not interchangeable. SmolVLA RL memory and whole-VLA full-parameter RL peaks remain unverified. Updating a complete action head is not full-model tuning.

### 搁置的DiT12实验 / Deferred DiT12 experiment

DiT12指最后12个动作DiT块及两个输出投影，共412,297,216可训练参数，其余冻结。短测19.28 GiB；4回合、6,613步／24轮小训练19.14 GiB；最终权重十对7/10→7/10（改善2、退步2）。五场共同成功平均751.6→638步，但没有成功率提升证据。

该工程SAC实现训练使用固定flow先验加高斯探索，评测使用原生fresh flow，存在分布差异。**当前搁置，不列入推荐训练路线，不打包其训练器或权重。** 完整DiT的另一次尝试在第二轮反向触及进程分配上限，未形成完整峰值；40–45 GiB档也未完成，不填入成功测峰。

DiT12 is deferred. Its training/deployment sampling mismatch and unchanged 7/10 score prevent treating it as an improved policy. The incomplete full-DiT attempt and unrun high-memory tiers are not accepted peak measurements.

## 代码交付与下一步 / Code delivery and next steps

- 本仓库提供GR00T N1.5／N1.6、π₀.₅及Edge SmolVLA独立推理服务，共用评测与CPU轨迹审核；OFT提供缺项拒绝接口。新增SmolVLA原生辅助协议仅通过CPU检查，不继承历史单例成绩。
- 实验RL训练器、学习后的残差／LoRA／Adapter／DiT参数及完整恢复包**尚未发布**。仓库命令加载原公开基座；复现训练后结果还需要对应训练器与学习参数。
- 模型与数据继续按[资源清单](RESOURCES.md)获取。后续公开学习参数需另有模型仓库、固定revision及许可审核；原始轨迹、机器配置和私人存储位置不随Git分发。
- 新运行按[SwanLab指南](SWANLAB.md)记录数值与白名单配置。本文结果依据本地回合记录和离线曲线审核；云端同步不在本页验收范围内。
- 后续优先做N1.5可归因的RL对照、SmolVLA统一协议基线、π₀.₅失败诊断，以及OFT受训包／BC方案选择。训练前按[数据与训练指南](TRAIN_AND_EVALUATE.md)完成准入；不通过改评分、裁剪标签或更换统计掩盖失败。

Public code covers inference, protocols and audits. Reproducing post-training results also requires the unreleased trainers and learned artifacts. Reported results are backed by local episode audits and offline curves; cloud synchronization is not established here.
