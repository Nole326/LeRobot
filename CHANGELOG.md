# 版本记录 / Changelog

## Unreleased

依据当前项目任务说明重写README与海报文案，突出视觉多物体抓放、遥操与BC/VLA基线、强化学习、泛化评测和安全部署。海报保留原有字体字号、排版、纯色图形及官方照片，重新生成原生4K PNG和矢量文字SVG。

Rewrite the README and poster copy around the current task specification: visual multi-object pick-and-place, teleoperation and BC/VLA baselines, reinforcement learning, generalization evaluation and safe deployment. Retain the poster's typefaces, sizes, layout, flat-color artwork and official photograph, regenerating the native-4K PNG and outlined-text SVG.

将当前项目说明对齐为视觉抓放，保留早期海报，明确海报不定义控制接口。新增开发起步指南、观测/动作/数据/恢复接口约定和无依赖的实验计划检查器：检查 schema、动作单位/关节顺序、控制频率、chunk 执行长度、数据划分和预算单位。未确认参数保留空白，检查结果不替代训练验收。

Align the project description with visual pick-and-place while retaining the earlier poster and distinguishing it from the control specification. Add a development guide, observation/action/data/recovery contracts and a dependency-free plan inspector covering schemas, action units/joint order, control timing, chunk execution, dataset splits and budget units. Leave unconfirmed values blank; static checks do not replace training acceptance.

仓库检查同时覆盖未暂存且未忽略的项目新文件，避免新增源码和文档遗漏语法/链接检查；上游文件与公开工具哈希清单保持原样。

Repository checks now include non-ignored new first-party files before staging, preventing new source and documentation from missing syntax/link checks. Upstream files and public-utility hash manifests remain unchanged.

新增训练前验证工具：末帧与超时转移检查、ACT padding、匹配的随机数/采样位置存档及独立进程恢复对照。工具独立于上游源码，尚未接入正式训练入口，使用范围见[训练前验证](docs/TRAINING_VALIDATION.md)。

Add pre-training checks for final observations and timeout transitions, ACT padding, matching RNG/sample-position checkpoints and fresh-process recovery. The tools are separate from upstream sources and not yet integrated into a production trainer; see [training validation](docs/TRAINING_VALIDATION.md) for scope.

## v0.1.1-baseline — 2026-10-02

更新开发基准说明与 README，明确版本记录流程。上游源码和依赖版本保持不变，保留 `v0.1.0-baseline` 供历史对照。

Update the baseline guide and README, and document the version-history workflow. Upstream sources and dependency versions are unchanged; `v0.1.0-baseline` remains available for historical comparison.

统一全海报文字层级，修复01/02/03对应字号、字重、行距及字形宽窄不一致；长句换行而不压缩。加入字体层级与等比缩放回归检查，背景及官方照片不变。

Unify typography across the poster, correcting inconsistent sizes, weights, line heights and glyph proportions in sections 01/02/03; wrap long sentences instead of compressing them. Add regression checks for type roles and isotropic scaling, leaving the background and official photograph unchanged.

海报文字改为按已确认双语版位置重绘的字体轮廓，提供原生4K文字PNG与可缩放文字SVG；背景及图形配色校准到最初课程海报的纯色区域，移除后续生成的纹理、渐变及新增装饰，保留官方照片，统一小标题中英文颜色，并核对官方英文术语。

Rebuild poster lettering as font outlines at the approved bilingual positions, supplying native-4K text in PNG and scalable text in SVG; calibrate flat background and graphic colors to the first course poster, remove subsequently generated texture, gradients and added decorations, retain the official photograph, match bilingual subheading colors, and review official English terminology.

将海报中的机器人照片替换为官方 SO-101 清晰原图，仅等比缩放并裁切背景留白；同步更新双语致谢和图片来源。`v0.1.0-baseline` 标签不变。

Replace the robot photograph with the official clear SO-101 image, using proportional resizing and background-margin cropping only; update bilingual acknowledgements and image attribution. The `v0.1.0-baseline` tag remains unchanged.

## v0.1.0-baseline — 2026-10-02

建立项目源码整合初始基准：固定三个上游项目的完整快照、保留原始许可、物化 LFS 测试素材，并加入内容清单与校验、双语项目介绍和4K插值海报。

Establish the initial source-integration baseline: pin complete snapshots of three upstream projects, preserve their licenses, materialize LFS test fixtures, and include manifests, checks, a bilingual overview and an interpolated 4K poster.

增加统一无 GPU 检查入口、检查器单元测试、GitHub Actions 自动检查、基准使用方法和贡献规范。当前版本不宣称正式任务训练或真机部署完成。

Add a unified GPU-free check entry point, checker unit tests, GitHub Actions checks, baseline usage instructions and contribution guidelines. This version does not claim completed formal-task training or hardware deployment.
