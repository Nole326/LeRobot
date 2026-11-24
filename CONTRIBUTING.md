# 修改与验证 / Contributing and validation

日常开发使用当前 main。复现初始源码快照可切换到 `v0.1.x-baseline`。集成上游代码时优先使用独立适配器，算法、奖励、观测、控制器和评测协议的改动分别记录和验证。

Use the current main branch for development and `v0.1.x-baseline` to reproduce the initial source snapshot. Prefer standalone adapters for upstream integrations. Document and validate changes to algorithms, rewards, observations, controllers and evaluation protocols separately.

## 智能体协作 / Working with agents

[AGENTS.md](AGENTS.md)负责仓库导航与共同规则。[LiftCube](course/liftcube/AGENTS.md)、[PickOrange 模型评测与采集](course/pickorange/AGENTS.md)和[独立诊断工具](course/scripts/AGENTS.md)分别说明运行入口、依赖与验收条件。修改跨目录接口时，检查受影响目录的约定。[PR 模板](.github/pull_request_template.md)用于记录改动、验证与公开内容复核。

The [root guide](AGENTS.md) covers repository navigation and shared conventions. The [LiftCube](course/liftcube/AGENTS.md), [PickOrange evaluation and collection](course/pickorange/AGENTS.md), and [standalone diagnostics](course/scripts/AGENTS.md) guides cover entry points, dependencies, and acceptance criteria. Review the applicable guides when a change crosses directory boundaries. Use the [PR template](.github/pull_request_template.md) to document changes, validation, and publication checks.

## 分支与评审 / Branches and review

LiftCube、PickOrange 和共享工具的变更从最新 `main` 建立短期分支。PR 标题写明主要改动，描述列出受影响的入口、接口或文档，并说明实际运行的检查。新增运行入口时，说明公共入口、外部依赖和输入输出。可复用的启动方法写在对应的运行指南中。

Start short-lived branches from current `main` for LiftCube, PickOrange, and shared tooling. Use the PR title to identify the primary change. In the description, list affected entry points, interfaces, or documents and the checks you ran. For a new runtime entry point, identify the public entry point, dependencies, and inputs and outputs. Put reusable launch instructions in the runtime guide.

基线、SFT 和 RL 微调通过配置与权重来源区分。公共数据读取、动作映射和评测修复可以单独提交，便于其他开发分支同步。合并后清理已完成的开发分支。

Use configurations and checkpoint provenance to distinguish baselines, SFT, and RL fine-tuning. Shared data, action-mapping, and evaluation fixes can be submitted separately so other branches can adopt them. Retire completed development branches after merging.

Git保存代码、配置、数据/权重清单。大镜像通过原渠道或获准的容器仓库分发，数据与权重使用有权限控制的存储或模型/数据仓库；记录固定版本、许可和获取方式，避免将运行产物加入Git历史。

Keep code, configurations and data/weight manifests in Git. Distribute large images through their original channel or an approved container registry, and datasets/weights through access-controlled storage or model/data repositories. Record fixed versions, licenses and retrieval instructions rather than adding runtime artifacts to Git history.

## 版本记录 / Version history

每次 commit 用简短标题说明改动，必要时在正文补充原因和验证结果。错字、格式等小修正通常只需提交说明；新增功能、重要修复、依赖升级和兼容性变化同时记入 [CHANGELOG.md](CHANGELOG.md) 的 `Unreleased`，按中文一段、英文一段记录。

Give each commit a concise description, adding rationale and validation results in the body when needed. Typos and formatting fixes usually need only a commit message. Add features, significant fixes, dependency upgrades and compatibility changes to `Unreleased` in [CHANGELOG.md](CHANGELOG.md), using Chinese-then-English entries.

阶段性发布时，将 `Unreleased` 的内容整理到带版本号和日期的新条目下，再保留一个新的 `Unreleased`。同步 README 和相关版本文档，运行检查、提交后，在该提交上创建带说明的 Git tag，并推送提交和标签。已发布标签不覆盖；GitHub Release 可复用同一份版本摘要，无需每版另建 Markdown 文件。

For a milestone release, move the accumulated entries into a versioned, dated section and retain a fresh `Unreleased` section. Update the README and relevant version docs, run checks, commit, then create an annotated Git tag on that commit and push both. Never overwrite published tags. A GitHub Release can reuse the same summary; a separate Markdown file per version is unnecessary.

README 面向组员和首次访问的开发者：先说明项目用途、可用功能和起步方法，再链接到详细文档。简短区分已实现与计划中的功能；安装细节、验证记录和协作规则放到对应文档。图片来源保留在致谢或素材说明中，沟通过程与内部操作记录不写入首页。

Write for teammates and first-time developers: explain the purpose, available features and first steps, then link to details. Distinguish implemented and planned features briefly; keep setup details, validation records and collaboration rules in their own documents. Retain image credits in acknowledgements or asset notes, leaving conversations and internal operations out of the homepage.

提交前执行统一检查：

Run the unified checks before submitting:

```bash
python scripts/check_project.py
git diff --check
git diff --cached --stat
```

涉及动作、数据、训练或评测接口时，同时更新[接口约定](docs/INTERFACE_CONTRACTS.md)和配置检查测试；私有实验配置不提交。运行 `python scripts/inspect_plan.py configs/experiment-plan.toml --print-config` 可查看模板缺项，退出码 3 是未填写模板的预期结果，不是训练失败。具体阶段分工见[开发起步指南](docs/GETTING_STARTED.md)。

For action, data, training or evaluation interface changes, update the [contracts](docs/INTERFACE_CONTRACTS.md) and plan-inspection tests; do not commit private experiment configurations. Run `python scripts/inspect_plan.py configs/experiment-plan.toml --print-config` to list template gaps. Exit code 3 is expected for the unfilled template, not a training failure. See the [development guide](docs/GETTING_STARTED.md) for stage ownership.

仓库检查验证源码一致性、Python 语法、Markdown 本地链接及海报尺寸，并运行软件测试。提交时列出跳过的测试，以及另行执行的仿真或硬件验证。

Automated checks validate source consistency, public utility Python syntax, local Markdown links, poster dimensions, and software tests. Report skipped tests and any simulator or hardware validation separately.

PR 的 `source-check` 和 `vla-cpu` 见 [CI 工作流](.github/workflows/source-check.yml)。修改模型服务、控制脚本或日志时，按受影响目录的说明运行对应测试。历史实验应注明所用入口、模型与数据版本、动作时钟、场景集、时限和实际完成场数。通过 CPU 检查后，闭环仿真成绩仍以逐场记录为准。

The [CI workflow](.github/workflows/source-check.yml) defines `source-check` and `vla-cpu`. Run the relevant checks for changes to model services, controllers, or logging. For historical results, identify the runner, model and data versions, action rate, scene set, time limit, and completed episode count. Use episode records as evidence for simulator results.

根目录的 [中文 README](README.md) 和 [English README](README.en.md) 同步维护。新增项目文档按中文在前、英文在后的顺序编写，上游文档保留原文。

Keep the Chinese and English READMEs in sync. Write new project documentation in Chinese followed by English, and preserve upstream documents in their original language.

禁止提交凭据、原始课程材料、服务器信息、私有配置、数据集、模型、运行日志及未获准的截图。忽略规则不能替代人工复核；不要在公开 issue 中粘贴这些内容。海报仅限已获准的脱敏版本。

公开提交前确认Git作者/提交者的姓名与邮箱，避免系统默认身份携带本机账号或设备名；不希望公开邮箱时，使用经本人确认的GitHub no-reply地址。正文、文件内容和Git元数据分别检查，修改工作区文件不会移除历史元数据。

Do not commit credentials, original course materials, server details, private configuration, datasets, models, runtime logs or unapproved screenshots. Ignore rules do not replace manual review; do not paste such content into public issues. Only the approved redacted poster is public.

Before publishing, confirm author/committer names and emails rather than relying on system defaults that may expose local accounts or device names. Use a personally confirmed GitHub no-reply address when email privacy is needed. Review content, messages and Git metadata separately; editing working files does not remove historical metadata.

上游及资产的许可证、版权和引用必须保留。不要重写基准历史或强制覆盖基准标签；升级依赖应作为独立、可审查的变更。

Preserve licenses, copyright notices and citations for upstream code and assets. Do not rewrite baseline history or force-update its tag; dependency upgrades should be separate, reviewable changes.
