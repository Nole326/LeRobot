# SO-101 LiftCube

抓取桌面方块并抬升的开发起点，包含环境适配器、演示读取、BC、参考PPO和评测入口。完整多物体抓放使用后续发布的任务环境。

A starting point for picking up a tabletop cube, with an environment adapter, demonstration reader, BC, reference PPO and evaluation entry points. The complete multi-object task uses its separately released environment.

## 当前状态 / Status

环境与数据接口可用于开发；BC/PPO仍是参考实现，正式训练前需完成以下接入：

The environment and data interfaces are available for development. BC/PPO remain reference implementations with the following work pending before formal training:

| 环节 / Component | 待完成 / Pending work |
| --- | --- |
| 默认策略 / Default policy | SmallCNN当前只使用图像，本体融合尚未接入 / SmallCNN is image-only; proprioceptive fusion is not connected |
| BC→RL | 30Hz演示与60Hz环境之间的训练时序适配；当前不能直接加载BC初始化RL / Training-time adaptation between 30Hz demonstrations and the 60Hz environment; direct BC initialization of RL is currently rejected |
| 存档恢复 / Checkpoint recovery | 当前保存权重与接口配置；优化器、训练进度和完整RNG尚未保存 / Weights and interface specs are saved, but optimizer, training progress and complete RNG state are not |
| PPO超时 / PPO timeouts | 当前将截断与终止一并处理，需明确任务时域及bootstrap规则 / Truncation and termination currently share a mask; task-horizon and bootstrap semantics need to be defined |

演示动作映射也需先完成核对，数据要求见下方[起步说明](#起步--getting-started)。ACT/VLA接入后共用环境和评测接口。

Verify demonstration action mapping before training; see [Getting started](#起步--getting-started) below. ACT/VLA integrations will share the environment and evaluation interfaces.

## 运行环境 / Runtime

使用配套镜像 `isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src`：Isaac Sim 4.5.0、Isaac Lab v2.1.1、LeIsaac v0.1.2、Python 3.10.15、Torch 2.5.1+cu118。包元数据可能显示Isaac Lab 0.41.3、LeIsaac 0.1.0。仓库原有 `third_party/` 是另一组固定版本，不要混入这个环境的Python路径。

Use the matching image `isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src`: Isaac Sim 4.5.0, Isaac Lab v2.1.1, LeIsaac v0.1.2, Python 3.10.15 and Torch 2.5.1+cu118. Package metadata may report Isaac Lab 0.41.3 and LeIsaac 0.1.0. The repository's existing `third_party/` snapshots use different versions; keep them off this runtime's Python path.

从[交大云盘分享页](https://pan.sjtu.edu.cn/web/share/74e1cfacf999d941ad66860a03104554)下载 `isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar`，核对下方SHA256后在Docker宿主导入。该链接是分享页面，不是tar文件直链；访问按页面要求及课程授予的权限进行。仓库不存储模拟器镜像、场景资产或缓存。

Download `isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar` from the [SJTU cloud-drive share](https://pan.sjtu.edu.cn/web/share/74e1cfacf999d941ad66860a03104554), verify the SHA256 below, then load it on the Docker host. This is a share page, not a direct tar URL; follow its access requirements and the permissions granted by the course. Simulator images, scene assets and caches remain outside Git.

```bash
sha256sum /path/to/isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar
docker load -i /path/to/isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar
docker image inspect --format '{{.Id}}' isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src
```

Windows下载后可用PowerShell核验 / Verify a Windows download with PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar'
```

镜像归档SHA256 / Image archive SHA256：`83df9539e280d663681b876cda3aa4eff631164f743f0aad3d693e1e17d6b17f`。

导入后的image ID / Loaded image ID：`sha256:17d6d1a69fc4b361dffab3238ab170f72ec7f2f9b5cb6d58050e1c6ffd80d752`。

以上校验值对应当前配套镜像；若分发方更新文件，先核对发布说明和版本，再更新清单。

These checksums identify the current matching image. If the distributor replaces it, confirm the release/version before updating the inventory.

## 接口 / Interface

| 内容 / Field | 当前约定 / Contract |
| --- | --- |
| 图像 / Image | front RGB, uint8 HWC, 64×64×3 |
| 本体 / Proprioception | 6关节位置 / joint positions; 6速度补零 / zero velocity slots; 6上一执行命令 / previous applied commands; 可选3维末端位置补零 / optional 3 zero EE slots |
| 动作 / Action | 6维绝对电机目标 / absolute motor targets; arm −100…100, gripper 0…100 |
| 转换 / Conversion | 按关节限位映射到弧度 / per-joint mapping to radians |
| 频率 / Frequency | 环境60Hz，30Hz策略每两tick更新 / 60Hz environment, 30Hz policies hold for two ticks |
| 默认上限 / Default horizon | 640环境tick，约10.67秒 / 640 environment ticks, about 10.67 seconds |

默认评测为4回合、每回合最多640环境tick。这是起点配置，不是最终课程评测协议；正式任务的回合数、成功判据和时间上限以课程发布为准。

The default evaluation runs four episodes with at most 640 environment ticks each. These are starter defaults, not the final course protocol; use the course's published episode count, success criteria and time limit for formal evaluation.

训练和推理均使用64×64输入。物体真值只供训练奖励或调试，不能进入策略。更换动作语义、频率或输入结构时，需重新核对数据和模型兼容性；关节动作不能直接当成受限末端动作。

Use 64×64 inputs during both training and inference. Object ground truth is reserved for training rewards or debugging, never policy inputs. Changes to action semantics, frequency or observations require renewed data/model compatibility checks; joint targets are not constrained end-effector actions.

## 起步 / Getting started

先创建符合所在平台规范的GPU容器，将本仓库挂载到 `/repo`，演示目录单独挂载到 `/data/lift_cube`。保留镜像内的 `/isaaclab`、`/leisaac`，缓存不要写入数据集目录。共享服务器在已批准的容器内执行以下命令，避免使用自动Docker启动路径。

Create a GPU container following your platform's rules, mount this repository at `/repo` and demonstrations separately at `/data/lift_cube`. Preserve the image's `/isaaclab` and `/leisaac` directories and keep caches outside datasets. On shared servers, run these commands inside an approved container rather than using automatic Docker startup.

```bash
cd /repo/course/liftcube
export PYTHONPATH="$PWD/src:/leisaac/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
/isaac-sim/python.sh -m starter.scripts.smoke_test --config configs/lift_cube.yaml --steps 16
```

离线读取还需要PyArrow和PyAV；先检查镜像已有依赖，避免替换预装Torch/CUDA。数据可参考 [H2Ozone/lift_cube](https://huggingface.co/datasets/H2Ozone/lift_cube)，但须先核对单位、零位、限位和帧同步。v3加载器默认拒绝超限动作；`allow_action_clipping`仅为显式诊断选项。BC配置的数据路径应设为容器内实际挂载路径。

Offline reading requires PyArrow and PyAV; check existing packages and preserve bundled Torch/CUDA. [H2Ozone/lift_cube](https://huggingface.co/datasets/H2Ozone/lift_cube) is a candidate dataset, subject to unit, zero-point, limit and alignment checks. The v3 reader rejects out-of-range actions by default; `allow_action_clipping` is an explicit diagnostic option. Set the BC data path to the actual container mount.

`training/train_bc.py`、`training/train_rl.py`为参考训练入口，`evaluation/evaluate.py`通过`--policy-class package.module:ClassName`支持自定义策略加载；默认训练预算仅作示例。

`training/train_bc.py` and `training/train_rl.py` are reference training entry points. `evaluation/evaluate.py` supports custom loaders through `--policy-class package.module:ClassName`; default training budgets are examples.

## 来源与修改 / Provenance and changes

[来源清单](../../manifests/liftcube_source.json)记录原始归档及逐文件原始／当前SHA。[接口补丁](patches/team-interface.patch)记录全部源码修改：显式相机、单环境检查、动作历史、缩放一致、实际频率、SO101补零、终态捕获、渲染准备和数据限幅检查。评测配置保持原包内容。

The [source manifest](../../manifests/liftcube_source.json) records the original archive and original/current file hashes. The [patch](patches/team-interface.patch) contains all changes: camera/scalar checks, action history, consistent resizing, actual timing, SO101 padding, terminal capture, render preparation and clipping checks. The evaluation configuration retains its original contents.

原包 `pyproject.toml` 声明MIT，未附独立LICENSE文件；保留原声明和作者字段，见[第三方说明](../../THIRD_PARTY_NOTICES.md)。镜像、数据和权重分别遵循其来源条款。

The original `pyproject.toml` declares MIT without a standalone LICENSE file. Its declaration and author field are retained; see [third-party notices](../../THIRD_PARTY_NOTICES.md). Images, datasets and weights follow their respective source terms.
