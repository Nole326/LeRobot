# 修改与验证 / Contributing and validation

新开发从最新`main`创建短期功能分支；`v0.1.x-baseline`只用于复现初始源码快照，不包含后续LiftCube功能。用户指定现有分支时沿用该分支。优先新增独立适配器，避免直接修改已冻结的上游源码；算法、奖励、观测、控制器和评估口径的变化需要分别记录和验证。

Branch new work from current `main`; use `v0.1.x-baseline` only to reproduce the initial source snapshot, which predates LiftCube development. Keep an existing branch when explicitly requested. Prefer standalone adapters over modifying frozen upstream source. Changes to algorithms, rewards, observations, controllers and evaluation protocols need explicit records and validation.

## 智能体协作 / Working with agents

将 [AGENTS.md](AGENTS.md) 作为智能体的仓库入口，交接时给出任务范围、适用授权和验收条件。每个 PR 复核其中的目录、接口边界、命令和开发约定；相关变化与代码同 PR 更新，无影响则说明原因。详细规范只保留一处，其余文档链接过去；运行进度与私有部署记录不写入该文件。

Use [AGENTS.md](AGENTS.md) as the repository entry point for agents, supplying task scope, applicable authority and acceptance criteria. Review its layout, contract boundaries, commands and conventions in each PR; update affected guidance alongside code or explain why no change is needed. Keep detailed rules in one place and link to them; runtime progress and private deployment records do not belong there.

[AGENTS.md](AGENTS.md)负责任务分流与共同规则，[LiftCube](course/liftcube/AGENTS.md)和[独立诊断工具](course/scripts/AGENTS.md)的子级文件负责各自入口、运行前提与验收。它们是智能体指令，使用单份中文，不重复面向读者的双语介绍。[PR 模板](.github/pull_request_template.md)包含文档同步、验证范围和公开内容复核项；开发者或智能体复核根目录及受影响子级文件，无需为无关改动机械更新。

这些指令面向所有使用者，不包含个人服务器、机构名称、设备绑定、账号或个人运维规则；使用者的平台专属配置留在仓库外。

The [root guide](AGENTS.md) routes tasks and defines shared rules; nested guides for [LiftCube](course/liftcube/AGENTS.md) and [standalone diagnostics](course/scripts/AGENTS.md) define entry points, prerequisites and acceptance. These are agent instructions maintained in Chinese, not duplicate bilingual introductions. The [PR template](.github/pull_request_template.md) covers documentation, validation scope and publication checks. Review the root and applicable nested guides; unrelated edits do not require artificial instruction changes.

These instructions apply to any user and exclude personal infrastructure, institutional identifiers, device bindings, accounts and individual operations policies. Keep platform-specific user configuration outside the repository.

## LiftCube并行开发 / Parallel LiftCube development

`feature/liftcube`用于导入公共起点，合入main后结束。ACT、SmolVLA、OpenVLA-OFT、π₀.₅和RL等工作从最新main分别创建短期功能分支，例如`feature/liftcube-act`、`feature/liftcube-smolvla`、`feature/liftcube-openvla-oft`、`feature/liftcube-pi05`、`feature/liftcube-rl`，通过PR将可复用实现合入main。这里只约定命名，不预先建立空分支。

Use `feature/liftcube` to import shared foundations, then close it after merging into main. Start short-lived branches such as `feature/liftcube-act`, `feature/liftcube-smolvla`, `feature/liftcube-openvla-oft`, `feature/liftcube-pi05` and `feature/liftcube-rl` from current main, merging reusable implementations through PRs. These are naming examples, not pre-created branches.

分支隔离开发中的代码改动，各模型实现合入后在main中共存。基线、SFT和RL微调通过配置与权重来源区分，seed和checkpoint不另开分支。公共数据读取、动作映射和评测修复尽早单独合入，其他模型分支及时同步。只有确实需要多项未合入改动联合验证时，才临时增加`dev/liftcube`集成分支。

Branches isolate code under development; model implementations coexist on main after merging. Distinguish baselines, SFT and RL through configurations and weight provenance, not branches per seed or checkpoint. Merge shared data/action/evaluation fixes separately and synchronize model branches. Add a temporary `dev/liftcube` integration branch only when multiple pending changes require joint validation.

Git保存代码、配置、数据/权重清单和哈希。大镜像通过原渠道或获准的容器仓库分发，数据与权重使用有权限控制的存储或模型/数据仓库；记录固定版本、许可和获取方式，避免将运行产物加入Git历史。

Keep code, configurations and data/weight manifests with hashes in Git. Distribute large images through their original channel or an approved container registry, and datasets/weights through access-controlled storage or model/data repositories. Record fixed versions, licenses and retrieval instructions rather than adding runtime artifacts to Git history.

## 版本记录 / Version history

每次 commit 用简短标题说明改动，必要时在正文补充原因和验证结果。错字、格式等小修正通常只需提交说明；新增功能、重要修复、依赖升级和兼容性变化同时记入 [CHANGELOG.md](CHANGELOG.md) 的 `Unreleased`，替换其中的占位文字，按中文一段、英文一段记录。

Give each commit a concise description, adding rationale and validation results in the body when needed. Typos and formatting fixes usually need only a commit message. Add features, significant fixes, dependency upgrades and compatibility changes to `Unreleased` in [CHANGELOG.md](CHANGELOG.md), replacing its placeholder with Chinese-then-English entries.

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

自动检查核验源码哈希、公开工具 Python 语法、通用 Markdown 的本地文件链接及海报尺寸。它不运行上游代码，不验证外部网页可用性，也不是完整安全审计或 GPU 测试。

Automated checks validate source hashes, public utility Python syntax, local file links in generic Markdown and poster dimensions. They do not execute upstream code or verify external websites, and are not a complete security audit or GPU test.

根README分为中文 `README.md` 和英文 `README.en.md`，顶部互链；修改目标、用法、范围与文档入口时同步两版。GitHub中这是普通文档链接，不需要网页脚本或额外部署。其他新写项目说明仍按中文一段、英文一段组织。上游原文不强行翻译。提交说明应包含目的、受影响接口、检查结果与尚未验证的部分。

The root README has linked Chinese (`README.md`) and English (`README.en.md`) editions; synchronize goals, usage, scope and navigation. These are ordinary GitHub document links, requiring no scripts or deployment. Other new project documentation uses Chinese-then-English paragraphs. Do not rewrite upstream documents solely for translation. Describe purpose, affected interfaces, checks and remaining validation limits.

禁止提交凭据、原始课程材料、服务器信息、私有配置、数据集、模型、运行日志及未获准的截图。忽略规则不能替代人工复核；不要在公开 issue 中粘贴这些内容。海报仅限已获准的脱敏版本。

公开提交前确认Git作者/提交者的姓名与邮箱，避免系统默认身份携带本机账号或设备名；不希望公开邮箱时，使用经本人确认的GitHub no-reply地址。正文、文件内容和Git元数据分别检查，修改工作区文件不会移除历史元数据。

Do not commit credentials, original course materials, server details, private configuration, datasets, models, runtime logs or unapproved screenshots. Ignore rules do not replace manual review; do not paste such content into public issues. Only the approved redacted poster is public.

Before publishing, confirm author/committer names and emails rather than relying on system defaults that may expose local accounts or device names. Use a personally confirmed GitHub no-reply address when email privacy is needed. Review content, messages and Git metadata separately; editing working files does not remove historical metadata.

上游及资产的许可证、版权和引用必须保留。不要重写基准历史或强制覆盖基准标签；升级依赖应作为独立、可审查的变更。

Preserve licenses, copyright notices and citations for upstream code and assets. Do not rewrite baseline history or force-update its tag; dependency upgrades should be separate, reviewable changes.
