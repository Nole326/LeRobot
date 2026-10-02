# 项目海报 / Project poster

当前海报为课程项目海报的双语版，画布为3840×2160。机器人照片直接使用 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101) 的清晰原图，不再使用生成或手工修补的机器人图像。

The current poster is a bilingual version of the course project poster on a 3840×2160 canvas. The robot photograph is taken directly from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101), replacing the generated or retouched robot image.

原始照片保存在 [so101-follower-official.webp](so101-follower-official.webp)，尺寸2048×1536，文件保持下载原样。海报中仅等比缩小并裁去少量上下背景留白，未修改机器人或贴纸内容。

The original photograph is stored as [so101-follower-official.webp](so101-follower-official.webp), at 2048×1536, unchanged from the download. In the poster it is proportionally downscaled with a small crop of the top and bottom background margins; the robot and sticker contents are not retouched.

- 官方下载地址 / Official download: [SO101_Follower.webp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp)
- SHA-256: `855809851ecf2ac5a28b2f0050b4baca3adc5a18c5175908399f9c6a52dd6877`

## 清晰文字与原图配色 / Sharp typography and original colors

[4K PNG](lerobot-tabletop-pushing-bilingual.png) 的中文、英文与数字均由真实字体轮廓按3840×2160直接渲染，不再放大低分辨率文字。[SVG版](lerobot-tabletop-pushing-bilingual.svg) 将文字保存为矢量轮廓，无需安装字体；文字放大仍保持清晰。背景和图形亦为矢量形状，仅官方照片是细节有限的位图。

Chinese, English and numerical text in the [4K PNG](lerobot-tabletop-pushing-bilingual.png) is rendered directly from font outlines at 3840×2160, rather than enlarged from low-resolution lettering. The [SVG version](lerobot-tabletop-pushing-bilingual.svg) stores text as scalable outlines and needs no installed fonts. Backgrounds and graphics are vector shapes; only the official photograph remains raster and has finite detail.

配色依据最初提供的课程海报，而不是后续生成版。原图的大面积区域经取样确认为纯色：主背景`#F6F5F0`，照片底板`#E7ECE9`，流程底板`#E8ECE4`，深绿区域`#173B36`，物块`#39778B`，目标框底色`#D9E8D8`。背景、色块和图形已重建为纯色矢量形状，不含生成版的渐变、噪声、纹理、阴影、箭头和页脚竖线。保留已确认双语版的文字外框位置及照片显示区域，图形尺寸适配双语布局；不宣称与最初中文单语版布局逐像素一致。

Colors are sampled from the first supplied course poster, not later generated versions. Large regions in that original are flat: canvas `#F6F5F0`, photograph panel `#E7ECE9`, diagram panel `#E8ECE4`, dark green `#173B36`, blocks `#39778B`, and goal-box fill `#D9E8D8`. Backgrounds and graphics are rebuilt as flat vector shapes, without generated gradients, noise, texture, shadows, an arrow or a footer divider. The approved bilingual text bounds and photograph display area are retained, with shape dimensions adapted to the bilingual layout; this is not claimed to be pixel-identical to the first Chinese-only layout.

SVG嵌入完整2048×1536官方照片的无损PNG编码，解码像素与下载的WebP原图一致；显示时仅等比缩放并裁切背景留白。官方WebP文件保持原样。

The SVG embeds a losslessly encoded PNG of the complete official 2048×1536 photograph, whose decoded pixels match the downloaded WebP. Display uses proportional scaling and background-margin cropping only. The original WebP file remains unchanged.

中文使用接近原图的微软雅黑，英文和数字使用Calibri，主标题拉丁字符使用Times New Roman Bold；原图没有附带字体信息，因此字体为视觉匹配，不声称识别出原始字体。01/02/03之后的小标题分别从原中文采样颜色，同行英文与中文使用同一填色。不分发字体文件。

Chinese uses visually matched Microsoft YaHei; English and numerals use Calibri, with Times New Roman Bold for the Latin main title. The original image contains no font metadata, so these are visual matches, not identified original fonts. Each bilingual subheading after 01/02/03 uses one fill color sampled from its original Chinese text. No font files are distributed.

## 术语核对 / Terminology review

硬件名称“SO-101 follower”沿用[SO-101官方文档](https://huggingface.co/docs/lerobot/so101)；“end-effector”沿用[动作表示文档](https://huggingface.co/docs/lerobot/action_representations)。目标区域统一译为“goal region”，参考官方[HIL-SERL任务示例](https://huggingface.co/docs/lerobot/hilserl)中的推物描述。PPO/SAC保留算法缩写；任务步骤、小标题和课程安排为本项目译文，不宣称是官方任务定义或官方课程文案。

“SO-101 follower” follows the [official hardware documentation](https://huggingface.co/docs/lerobot/so101), and “end-effector” follows the [action-representation documentation](https://huggingface.co/docs/lerobot/action_representations). The poster consistently uses “goal region,” following the pushing example in the official [HIL-SERL guide](https://huggingface.co/docs/lerobot/hilserl). PPO/SAC remain algorithm abbreviations. Task steps, section headings and course arrangements are project translations, not official task definitions or official course copy.

图片权利归属不因本项目使用而改变。

Use in this project does not change the image's ownership.

原始课程PDF、联系人信息及私有课程记录不公开。海报展示项目目标，不是训练结果或功能验收证明。

The original course PDF, contact information and private course records are not published. The poster presents project objectives, not training results or proof of functional acceptance.
