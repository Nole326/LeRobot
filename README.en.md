<p align="center">
  <a href="assets/lerobot-tabletop-pushing-bilingual.svg">
    <img src="assets/lerobot-tabletop-pushing-bilingual.svg" width="960" alt="Bilingual LeRobot tabletop manipulation poster">
  </a>
</p>

<h1 align="center">LeRobot: Reinforcement Learning Project</h1>
<h3 align="center">Visual Tabletop Manipulation: From Simulation to SO-101 Deployment</h3>

<p align="center">
  Visual pick-and-place, imitation learning and reinforcement learning with LeRobot, LeIsaac and the SO-101.
</p>

<p align="center">
  <strong>English</strong> | <a href="README.md">简体中文</a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#repository-layout">Repository layout</a> ·
  <a href="#development-path">Development path</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

The current stage covers LiftCube and PickOrange as separate policy, data and evaluation tracks. The later course multi-object environment and hardware transfer are deferred.

## Current scope

- **LiftCube:** environment adapters, demonstration loading, BC, reference PPO, a visual rule policy and shared evaluation. See the [LiftCube guide](course/liftcube/README.md) for the matching image and runtime.
- **PickOrange:** pinned environment, dual-view GR00T/SmolVLA/PI05 services, shared evaluation and trajectory audits. The [privileged state control script](course/pickorange/expert/README.md) adds demonstration collection and independent action-trajectory replay verification. Control execution, independent replay and demonstration collection have been jointly validated in simulation. The new SmolVLA entry has CPU checks; its GPU rollout validation remains pending. See the [runner guide](course/pickorange/README.md).
- **VLA interfaces:** the [two-task interface guide](docs/VLA_INTERFACES.md) links model backends for both tasks: single-front wrappers for LiftCube and dual-view services for PickOrange. OFT still requires matching trained heads, projections and statistics.
- **Rules and control scripts:** the [two-task script guide](docs/SCRIPTED_POLICIES.md) links LiftCube visual/state policies, the official PickOrange state control script and upstream references. A [PickOrange visual rule policy](course/pickorange/visual/README.md) offers separate model-assisted and RGB/q6/known-camera-calibration variants under named 55-second protocols; privileged state control scripts remain separate.
- **Later scope:** the formal multi-object task, cross-object generalization and hardware deployment. The unreleased environment is not a prerequisite for current development.

See the [task and adaptation matrix](docs/TASKS.md) for runtimes, expert scripts, demonstrations and checkpoint/validation status across the four VLA families. A shared SO-101 robot does not make the interfaces interchangeable; keep configurations, checkpoints, statistics and scores separate.

See the [PickOrange validation summary](docs/PICKORANGE_VALIDATION.md) for VLA results, paired Residual SAC evaluation, and memory requirements. The [model resources](docs/RESOURCES.md#pickorange-act-权重--pickorange-act-checkpoints) link to ACT checkpoints, inference settings, and evaluation results. Experimental RL trainers and residual checkpoints remain unreleased.

Obtain weights and demonstrations from [resources](docs/RESOURCES.md), then follow the [training and evaluation guide](docs/TRAIN_AND_EVALUATE.md). PickOrange's native-assisted evaluation uses object truth for gripper control and is reported separately from non-privileged visual baselines.

## Quick start

Clone and check the repository (Git and Python 3.11+ required):

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

This check starts neither simulation nor training. Use the matching image from the [LiftCube runtime guide](course/liftcube/README.md#运行环境--runtime), or the [separate PickOrange deployment](docs/PICK_ORANGE.md). Do not mix their dependencies or launchers.

The rule baseline needs no demonstrations or trained weights, but does require matching robot/camera calibration. Start with the offline calibration checks in the [rule-runner guide](docs/SCRIPTED_LIFTCUBE.md). Prepare data and interfaces separately for BC/PPO using the [training guide](course/liftcube/TRAINING.md).

See [baseline notes](docs/BASELINE.md) and the [changelog](CHANGELOG.md) for versions. For development, branch from current `main` and follow the [contribution guide](CONTRIBUTING.md).

## Repository layout

LiftCube integration lives in `course/liftcube/src/starter/`. PickOrange execution lives in `course/pickorange/`, with diagnostics and pinned upstream task sources kept separately:

```text
.
├── course/liftcube/
│   ├── src/starter/
│   │   ├── interfaces/        # Observation, action, policy and model contracts
│   │   ├── envs/              # Simulator, control and terminal handling
│   │   ├── data/              # Demonstration reading, conversion, validation and splits
│   │   ├── algorithms/        # BC, RL and reference PPO
│   │   ├── training/          # Configuration-driven training entry points
│   │   ├── evaluation/        # Shared evaluation and policy loading
│   │   ├── rewards/           # Training rewards
│   │   └── logging/           # Metric recording
│   ├── configs/              # Current task configurations
│   └── patches/              # Reversible changes to the course starter
├── course/scripts/           # PickOrange environment, replay and ACT-bridge diagnostics
├── course/pickorange/        # Dual-view VLA services, evaluation and trajectory audits
├── configs/                  # Public experiment-plan templates
├── scripts/                  # Repository, provenance and plan checks
├── tests/                    # First-party regression tests
├── third_party/              # Pinned upstream sources
├── manifests/                # Source, version inventories
├── docs/                     # Architecture, interfaces, runtime and reproducibility
└── assets/                   # Poster and project artwork
```

See [architecture](docs/ARCHITECTURE.md) for layer ownership and model integration. Obtain images, datasets and weights separately; their sources and versions are listed in [dependencies](docs/DEPENDENCIES.md).

## Development path

1. Freeze each task's runtime, observations, actions and success criteria, then complete its entry points.
2. Retain LiftCube rule/data and ACT/compact-VLA tracks, while focusing current work on PickOrange VLA baselines, attributable RL comparisons and model-capability validation.
3. Use accepted data and models for fine-tuning, RL and relevant ablations, recording complete episodes and failures.
4. Resume cross-object transfer, generalization and hardware alignment after the formal multi-object environment is released.

BC and pretrained VLAs provide initialization or comparisons; RL is a separate training stage. Report success rate, completion time, wrong/missed grasps and placement error. Task environments define the success criteria and horizons.

See the [RLinf assessment](docs/RLINF.md) for VLA reinforcement-learning references. RLinf is not yet integrated into this project's runtime.

### Upstream versions

These are historical source snapshots. Current LiftCube separately uses the course's Isaac Sim 4.5.0 / Isaac Lab v2.1.1 / LeIsaac v0.1.2 runtime.

| Project | Historical source version | Pinned commit |
| --- | --- | --- |
| LeRobot | v0.4.2 | [58f70b6](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |
| LeIsaac | 0.4.0 | [24d3bcd](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| IsaacLab | v2.3.0 | [3c6e67b](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) |

See the [import inventory](docs/IMPORT_MANIFEST.md) for source provenance and integrity information.

## Documentation

- [SwanLab tracking](docs/SWANLAB.md): scalar curves and sanitized configuration; offline by default, private projects for online runs.

- [Development and stage acceptance](docs/GETTING_STARTED.md)
- [LiftCube/PickOrange resources and adaptation status](docs/TASKS.md)
- [PickOrange model validation, paired RL and memory reference](docs/PICKORANGE_VALIDATION.md)
- [Model weights and demonstration downloads](docs/RESOURCES.md)
- [Task-specific control script collection, training and evaluation](docs/TRAIN_AND_EVALUATE.md)
- [LiftCube source, image and runtime](course/liftcube/README.md)
- [PickOrange sources, separate runtime and diagnostics](docs/PICK_ORANGE.md)
- [Visual rule reference, calibration and migration](docs/SCRIPTED_LIFTCUBE.md)
- [Planar EE development interface and limitations](docs/PLANAR_EE_DEVELOPMENT.md)
- [Architecture, extension points and collaboration](docs/ARCHITECTURE.md)
- [RLinf benefits and integration boundaries](docs/RLINF.md)
- [Observation, action, data and recovery contracts](docs/INTERFACE_CONTRACTS.md)
- [Pre-training validation and limitations](docs/TRAINING_VALIDATION.md)
- [Dependencies and external assets](docs/DEPENDENCIES.md)
- [Deployment and reproducibility](docs/REPRODUCIBILITY.md)
- [Source import inventory](docs/IMPORT_MANIFEST.md)
- [Contributing and version management](CONTRIBUTING.md)
- [Agent development guide](AGENTS.md)
- [Poster and image notes](assets/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## References and acknowledgements

Thanks to [LeRobot](https://github.com/huggingface/lerobot), [LeIsaac](https://github.com/LightwheelAI/leisaac) and [IsaacLab](https://github.com/isaac-sim/IsaacLab). Engineering structure draws on [rl-garden](https://github.com/Nole326/rl-garden), and VLA reinforcement-learning research draws on [RLinf](https://github.com/RLinf/RLinf). See [third-party notices](THIRD_PARTY_NOTICES.md) for licenses.

The poster is a refined and adapted Chinese-English version of the course project poster. The robot image is sourced from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101).
