# 版本记录 / Changelog

## Unreleased

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
