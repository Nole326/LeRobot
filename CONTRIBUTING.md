# 修改与验证 / Contributing and validation

首次开发可从当前基准标签 `v0.1.1-baseline` 创建功能分支；后续协作通常从最新 `main` 分支开始，避免遗漏已合入的改动。优先新增独立适配器，避免直接修改已冻结的上游源码；算法、奖励、观测、控制器和评估口径的变化需要分别记录和验证。

Start initial development from `v0.1.1-baseline`; subsequent collaborative work should normally branch from the latest `main` to include merged changes. Prefer standalone adapters over modifying frozen upstream source. Changes to algorithms, rewards, observations, controllers and evaluation protocols need explicit records and validation.

## 版本记录 / Version history

每次 commit 用简短标题说明改动，必要时在正文补充原因和验证结果。错字、格式等小修正通常只需提交说明；新增功能、重要修复、依赖升级和兼容性变化同时记入 [CHANGELOG.md](CHANGELOG.md) 的 `Unreleased`，替换其中的占位文字，按中文一段、英文一段记录。

Give each commit a concise description, adding rationale and validation results in the body when needed. Typos and formatting fixes usually need only a commit message. Add features, significant fixes, dependency upgrades and compatibility changes to `Unreleased` in [CHANGELOG.md](CHANGELOG.md), replacing its placeholder with Chinese-then-English entries.

阶段性发布时，将 `Unreleased` 的内容整理到带版本号和日期的新条目下，再保留一个新的 `Unreleased`。同步 README 和相关版本文档，运行检查、提交后，在该提交上创建带说明的 Git tag，并推送提交和标签。已发布标签不覆盖；GitHub Release 可复用同一份版本摘要，无需每版另建 Markdown 文件。

For a milestone release, move the accumulated entries into a versioned, dated section and retain a fresh `Unreleased` section. Update the README and relevant version docs, run checks, commit, then create an annotated Git tag on that commit and push both. Never overwrite published tags. A GitHub Release can reuse the same summary; a separate Markdown file per version is unnecessary.

README 面向项目读者，介绍目标、用法和当前进度。图片来源放在致谢或素材说明中；沟通过程、编辑指令及内部操作记录不写入 README。

Write the README for project readers: describe goals, usage and current progress. Keep image attribution in acknowledgements or asset notes, and leave conversations, editing instructions and internal operations out of the README.

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

README 和新写的项目说明按中文一段、英文一段组织。上游原文不强行翻译。提交说明应包含目的、受影响接口、检查结果与尚未验证的部分。

Use Chinese-then-English paragraphs in the README and new project documentation. Do not rewrite upstream documents solely for translation. Describe the purpose, affected interfaces, checks performed and remaining validation limits in each contribution.

禁止提交凭据、原始课程材料、服务器信息、私有配置、数据集、模型、运行日志及未获准的截图。忽略规则不能替代人工复核；不要在公开 issue 中粘贴这些内容。海报仅限已获准的脱敏版本。

Do not commit credentials, original course materials, server details, private configuration, datasets, models, runtime logs or unapproved screenshots. Ignore rules do not replace manual review; do not paste such content into public issues. Only the approved redacted poster is public.

上游及资产的许可证、版权和引用必须保留。不要重写基准历史或强制覆盖基准标签；升级依赖应作为独立、可审查的变更。

Preserve licenses, copyright notices and citations for upstream code and assets. Do not rewrite baseline history or force-update its tag; dependency upgrades should be separate, reviewable changes.
