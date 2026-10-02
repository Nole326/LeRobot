# 项目海报 / Project poster

当前海报为课程项目海报的双语版，画布为3840×2160。机器人照片直接使用 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101) 的清晰原图，不再使用生成或手工修补的机器人图像。

The current poster is a bilingual version of the course project poster on a 3840×2160 canvas. The robot photograph is taken directly from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101), replacing the generated or retouched robot image.

原始照片保存在 [so101-follower-official.webp](so101-follower-official.webp)，尺寸2048×1536，文件保持下载原样。海报中仅等比缩小并裁去少量上下背景留白，未修改机器人或贴纸内容。

The original photograph is stored as [so101-follower-official.webp](so101-follower-official.webp), at 2048×1536, unchanged from the download. In the poster it is proportionally downscaled with a small crop of the top and bottom background margins; the robot and sticker contents are not retouched.

- 官方下载地址 / Official download: [SO101_Follower.webp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp)
- SHA-256: `855809851ecf2ac5a28b2f0050b4baca3adc5a18c5175908399f9c6a52dd6877`

## 清晰文字与原背景 / Sharp typography and preserved background

[4K PNG](lerobot-tabletop-pushing-bilingual.png) 的中文、英文与数字均由真实字体轮廓按3840×2160直接渲染，不再放大低分辨率文字。[SVG版](lerobot-tabletop-pushing-bilingual.svg) 将文字保存为矢量轮廓，无需安装字体；文字放大仍保持清晰。照片和背景仍是位图，SVG并不意味着它们可以无限增加细节。

Chinese, English and numerical text in the [4K PNG](lerobot-tabletop-pushing-bilingual.png) is rendered directly from font outlines at 3840×2160, rather than enlarged from low-resolution lettering. The [SVG version](lerobot-tabletop-pushing-bilingual.svg) stores text as scalable outlines and needs no installed fonts. The photograph and background remain raster images; SVG does not add unlimited detail to them.

保留原海报的背景、渐变、细微纹理、图形和照片区域；只修复旧文字及其插值模糊边缘，再于测量所得的37处原文字外框内重绘。文字修复与重绘区域之外的PNG像素与前版一致，机器人照片区域逐像素一致；SVG另嵌入完整2048×1536官方原图，沿用同一显示区域及裁切方式。旧字覆盖处的底纹依据邻近像素修复，并不声称能恢复被旧字遮住的未知纹理。

The previous background, gradients, subtle texture, graphics and photograph region are retained. Only old lettering and its interpolation halos are repaired, then text is redrawn within 37 measured original text bounds. PNG pixels outside the text-repair/replacement regions match the previous version exactly, as does the photograph region. The SVG additionally embeds the full official 2048×1536 original, using the same display area and crop. Texture beneath old lettering is reconstructed from nearby pixels, not claimed to recover unknown occluded detail.

中文使用接近原图的微软雅黑，英文和数字使用Calibri，主标题拉丁字符使用Times New Roman Bold；原图没有附带字体信息，因此字体为视觉匹配，不声称识别出原始字体。01/02/03之后的小标题分别从原中文采样颜色，同行英文与中文使用同一填色。不分发字体文件。

Chinese uses visually matched Microsoft YaHei; English and numerals use Calibri, with Times New Roman Bold for the Latin main title. The original image contains no font metadata, so these are visual matches, not identified original fonts. Each bilingual subheading after 01/02/03 uses one fill color sampled from its original Chinese text. No font files are distributed.

## 术语核对 / Terminology review

硬件名称“SO-101 follower”沿用[SO-101官方文档](https://huggingface.co/docs/lerobot/so101)；“end-effector”沿用[动作表示文档](https://huggingface.co/docs/lerobot/action_representations)。目标区域统一译为“goal region”，参考官方[HIL-SERL任务示例](https://huggingface.co/docs/lerobot/hilserl)中的推物描述。PPO/SAC保留算法缩写；任务步骤、小标题和课程安排为本项目译文，不宣称是官方任务定义或官方课程文案。

“SO-101 follower” follows the [official hardware documentation](https://huggingface.co/docs/lerobot/so101), and “end-effector” follows the [action-representation documentation](https://huggingface.co/docs/lerobot/action_representations). The poster consistently uses “goal region,” following the pushing example in the official [HIL-SERL guide](https://huggingface.co/docs/lerobot/hilserl). PPO/SAC remain algorithm abbreviations. Task steps, section headings and course arrangements are project translations, not official task definitions or official course copy.

图片权利归属不因本项目使用而改变。

Use in this project does not change the image's ownership.

原始课程PDF、联系人信息及私有课程记录不公开。海报展示项目目标，不是训练结果或功能验收证明。

The original course PDF, contact information and private course records are not published. The poster presents project objectives, not training results or proof of functional acceptance.
