<p align="center">
  <a href="assets/lerobot-tabletop-pushing-bilingual.svg">
    <img src="assets/lerobot-tabletop-pushing-bilingual.svg?v=20261003-pdf-aligned" width="960" alt="Bilingual LeRobot tabletop manipulation poster">
  </a>
</p>

<h1 align="center">LeRobot</h1>
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

<p align="center">
  <a href="assets/lerobot-tabletop-pushing-bilingual.svg">Scalable poster</a> ·
  <a href="assets/lerobot-tabletop-pushing-bilingual.png">4K PNG</a>
</p>

---

Start with PickOrange to learn the data/control pipeline, establish training and evaluation on LiftCube, then align with the course multi-object task and real-robot deployment.

## Current scope

- **LiftCube:** the course starter, runtime guide, BC/reference PPO and evaluation entry points are included. It is a cube-grasping/lifting development task, not the full multi-object pick-and-place task.
- **Shared interfaces:** current LiftCube uses front RGB, six joint positions and absolute motor targets. Compatibility slots for velocity and end-effector position are zero-padded, not measured feedback. See the [task contract](course/liftcube/README.md#接口--interface) for the full input and timing.
- **Model directions:** ACT, SmolVLA, OpenVLA-OFT, π₀.₅ and GR00T are integration/validation candidates, not a claim that usable course-task checkpoints are provided.
- **Before formal training:** demonstration action mapping, model adapters and complete checkpoint recovery still require acceptance. Executable reference trainers do not establish training, generalization or hardware performance.

Formal course tasks, controllers and scoring rules take precedence over artwork and upstream examples. Policies use images and permitted proprioception, not object/target ground truth; training rewards may use privileged information where allowed. The final-task interface and current LiftCube joint control are not interchangeable.

## Quick start

Check repository contents without installing a simulator, allocating a GPU or starting training:

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

Checks require Git and Python 3.11+. The clone options apply only to the new repository, preserving upstream line endings and supporting deep Windows paths. Automated checks cover source integrity, syntax, documentation links, tests and poster dimensions—not simulator or training acceptance.

For development, follow the [LiftCube guide](course/liftcube/README.md), including the [matching image download, SHA256 and import instructions](course/liftcube/README.md#运行环境--runtime). Run code in an isolated environment approved by your platform, preserve bundled Torch/CUDA, and keep historical snapshots off its Python path.

The original source baseline is `v0.1.1-baseline`; see [baseline notes](docs/BASELINE.md). Branch from current `main` for subsequent team development. Record changes in the [changelog](CHANGELOG.md).

## Repository layout

Follow rl-garden's separation of responsibilities while reusing the existing `starter` package rather than creating another model/environment framework:

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
│   │   ├── rewards/           # Training rewards, not formal scoring
│   │   └── logging/           # Metric recording
│   ├── configs/              # Current task configurations
│   └── patches/              # Reversible changes to the course starter
├── course/scripts/           # Standalone diagnostics
├── configs/                  # Public experiment-plan templates
├── scripts/                  # Repository, provenance and plan checks
├── tests/                    # First-party regression tests
├── third_party/              # Pinned upstream sources, not experimental scratch space
├── manifests/                # Source, version and hash inventories
├── docs/                     # Architecture, interfaces, runtime and reproducibility
└── assets/                   # Approved project artwork
```

The [architecture guide](docs/ARCHITECTURE.md) defines responsibilities, extension points and collaboration. Merged model implementations coexist on `main`; branches isolate code changes, not permanent algorithm variants, seeds or checkpoints.

Images, datasets, weights, runtime logs and private deployment records stay outside Git. Record provenance, licenses, fixed versions and verification instructions instead.

## Development path

1. Align observations, actions, image processing and timing; verify terminal, timeout and safety behavior.
2. Accept a small complete demonstration set: units, joint order, synchronization, replay and success labels; split by episode.
3. Establish rule-based, ACT and compact-VLA baselines, then compare candidates such as OpenVLA-OFT and π₀.₅.
4. Establish a plain-RL comparison before direct VLA fine-tuning or residuals; account separately for demonstrations, interactions and compute.
5. Evaluate under the same tasks, scenes, seeds and success criteria, with ablations and failure analysis; pass deployment checks before hardware execution.

BC and pretrained VLAs provide initialization or comparisons, not a substitute for the required RL training. Distinguish seen, compositional and entirely unseen conditions, recording success rate, time, wrong/missed grasps and placement error. Success criteria and horizons follow formal releases, not other examples' defaults.

RLinf provides references for VLA SFT→RL, flow-policy optimization, chunk execution and evaluation, not an out-of-the-box course SO-101 trainer. See the [RLinf assessment](docs/RLINF.md) for reusable components, version differences and adaptation gaps. It is not currently a runtime dependency.

### Upstream versions

These are historical source snapshots. Current LiftCube separately uses the course's Isaac Sim 4.5.0 / Isaac Lab v2.1.1 / LeIsaac v0.1.2 runtime.

| Project | Historical source version | Pinned commit |
| --- | --- | --- |
| LeRobot | v0.4.2 | [58f70b6](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |
| LeIsaac | 0.4.0 | [24d3bcd](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| IsaacLab | v2.3.0 | [3c6e67b](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) |

Complete imports retain upstream licenses and inventories. All 45 LFS test fixtures are stored as regular Git files; no extra LFS download or submodule initialization is required.

## Documentation

- [Development and stage acceptance](docs/GETTING_STARTED.md)
- [LiftCube source, image and runtime](course/liftcube/README.md)
- [Architecture, extension points and collaboration](docs/ARCHITECTURE.md)
- [RLinf benefits and integration boundaries](docs/RLINF.md)
- [Observation, action, data and recovery contracts](docs/INTERFACE_CONTRACTS.md)
- [Pre-training validation and limitations](docs/TRAINING_VALIDATION.md)
- [Dependencies and external assets](docs/DEPENDENCIES.md)
- [Deployment and reproducibility](docs/REPRODUCIBILITY.md)
- [Source import inventory](docs/IMPORT_MANIFEST.md)
- [Contributing and version management](CONTRIBUTING.md)
- [Poster and image notes](assets/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## References and acknowledgements

Thanks to [LeRobot](https://github.com/huggingface/lerobot), [LeIsaac](https://github.com/LightwheelAI/leisaac) and [IsaacLab](https://github.com/isaac-sim/IsaacLab) for the open-source foundations. Engineering structure draws on [rl-garden](https://github.com/Nole326/rl-garden), and VLA reinforcement-learning research draws on [RLinf](https://github.com/RLinf/RLinf). Follow each source's license; integration does not change upstream ownership.

The poster is a refined and adapted Chinese-English version of the course project poster. The robot image is sourced from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101).
