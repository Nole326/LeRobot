# 修改与验证 / Contributing and validation

请从基准标签创建功能分支，保持改动范围清楚。优先新增独立适配器，避免直接修改已冻结的上游源码；算法、奖励、观测、控制器和评估口径的变化需要分别记录和验证。

Start a feature branch from the baseline tag and keep changes scoped. Prefer standalone adapters over modifying frozen upstream source. Changes to algorithms, rewards, observations, controllers and evaluation protocols need explicit records and validation.

提交前执行统一检查：

Run the unified checks before submitting:

```bash
python scripts/check_project.py
git diff --check
git diff --cached --stat
```

自动检查核验源码哈希、公开工具 Python 语法、通用 Markdown 的本地文件链接及海报尺寸。它不运行上游代码，不验证外部网页可用性，也不是完整安全审计或 GPU 测试。

Automated checks validate source hashes, public utility Python syntax, local file links in generic Markdown and poster dimensions. They do not execute upstream code or verify external websites, and are not a complete security audit or GPU test.

README 和新写的项目说明按中文一段、英文一段组织。上游原文不强行翻译。提交说明应包含目的、受影响接口、检查结果与尚未验证的部分。

Use Chinese-then-English paragraphs in the README and new project documentation. Do not rewrite upstream documents solely for translation. Describe the purpose, affected interfaces, checks performed and remaining validation limits in each contribution.

禁止提交凭据、原始课程材料、服务器信息、私有配置、数据集、模型、运行日志及未获准的截图。忽略规则不能替代人工复核；不要在公开 issue 中粘贴这些内容。海报仅限已获准的脱敏版本。

Do not commit credentials, original course materials, server details, private configuration, datasets, models, runtime logs or unapproved screenshots. Ignore rules do not replace manual review; do not paste such content into public issues. Only the approved redacted poster is public.

上游及资产的许可证、版权和引用必须保留。不要重写基准历史或强制覆盖基准标签；升级依赖应作为独立、可审查的变更。

Preserve licenses, copyright notices and citations for upstream code and assets. Do not rewrite baseline history or force-update its tag; dependency upgrades should be separate, reviewable changes.
