# 项目海报 / Project poster

当前海报介绍视觉 Pick & Place：多物体抓放、模仿学习与强化学习、泛化评测及安全真机部署。内容依据项目任务说明，画布为3840×2160，沿用原双语海报的版式、配色、图形和字体层级。机器人照片使用 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101) 的清晰原图。

The current poster presents visual Pick & Place: multi-object manipulation, imitation and reinforcement learning, generalization evaluation and safe real-robot deployment. Its content follows the project task specification on a 3840×2160 canvas, retaining the previous bilingual poster's layout, colors, graphics and type hierarchy. The robot photograph comes from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101).

原始照片保存在 [so101-follower-official.webp](so101-follower-official.webp)，尺寸2048×1536，文件保持下载原样。海报中仅等比缩小并裁去少量上下背景留白，未修改机器人或贴纸内容。

The original photograph is stored as [so101-follower-official.webp](so101-follower-official.webp), at 2048×1536, unchanged from the download. In the poster it is proportionally downscaled with a small crop of the top and bottom background margins; the robot and sticker contents are not retouched.

- 官方下载地址 / Official download: [SO101_Follower.webp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/SO101_Follower.webp)
- SHA-256: `855809851ecf2ac5a28b2f0050b4baca3adc5a18c5175908399f9c6a52dd6877`

## 清晰文字与原图配色 / Sharp typography and original colors

[4K PNG](lerobot-tabletop-pushing-bilingual.png) 的中文、英文与数字均由真实字体轮廓按3840×2160直接渲染，不再放大低分辨率文字。[SVG版](lerobot-tabletop-pushing-bilingual.svg) 将文字保存为矢量轮廓，无需安装字体；文字放大仍保持清晰。背景和图形亦为矢量形状，仅官方照片是细节有限的位图。

Chinese, English and numerical text in the [4K PNG](lerobot-tabletop-pushing-bilingual.png) is rendered directly from font outlines at 3840×2160, rather than enlarged from low-resolution lettering. The [SVG version](lerobot-tabletop-pushing-bilingual.svg) stores text as scalable outlines and needs no installed fonts. Backgrounds and graphics are vector shapes; only the official photograph remains raster and has finite detail.

配色依据最初提供的课程海报，而不是后续生成版。原图的大面积区域经取样确认为纯色：主背景`#F6F5F0`，照片底板`#E7ECE9`，流程底板`#E8ECE4`，深绿区域`#173B36`，物块`#39778B`，目标框底色`#D9E8D8`。背景、色块和图形已重建为纯色矢量形状，不含生成版的渐变、噪声、纹理、阴影、箭头和页脚竖线。保留已确认双语布局与照片显示区域；文字改用统一层级和自然字形比例，长句换行，不再强行适配每行旧外框。不宣称与最初中文单语版布局逐像素一致。

Colors are sampled from the first supplied course poster, not later generated versions. Large regions in that original are flat: canvas `#F6F5F0`, photograph panel `#E7ECE9`, diagram panel `#E8ECE4`, dark green `#173B36`, blocks `#39778B`, and goal-box fill `#D9E8D8`. Backgrounds and graphics are rebuilt as flat vector shapes, without generated gradients, noise, texture, shadows, an arrow or a footer divider. The approved bilingual arrangement and photograph area are retained. Text now uses consistent type roles and natural glyph proportions; long sentences wrap instead of being fitted to individual old bounds. This is not claimed to be pixel-identical to the first Chinese-only layout.

SVG嵌入完整2048×1536官方照片的无损PNG编码，解码像素与下载的WebP原图一致；显示时仅等比缩放并裁切背景留白。官方WebP文件保持原样。

The SVG embeds a losslessly encoded PNG of the complete official 2048×1536 photograph, whose decoded pixels match the downloaded WebP. Display uses proportional scaling and background-margin cropping only. The original WebP file remains unchanged.

中文使用接近原图的微软雅黑，英文和数字使用Calibri，主标题拉丁字符使用Times New Roman Bold；原图没有附带字体信息，因此字体为视觉匹配，不声称识别出原始字体。对应层级使用相同字号、字重、颜色与行距，所有字形等比缩放；01/02/03小标题的中英文同色。不分发字体文件。

Chinese uses visually matched Microsoft YaHei; English and numerals use Calibri, with Times New Roman Bold for the Latin main title. The original image contains no font metadata, so these are visual matches, not identified original fonts. Corresponding roles share size, weight, color and line height, with isotropic glyph scaling. Chinese and English in the 01/02/03 subheadings share one color. No font files are distributed.

在3840×2160画布上，正文编号为72px，小标题54px，中文粗体主句68px、对应英文58px，中英文说明均为46px。它们是字体em字号，不是逐句可见像素高度；字母上下伸部不同不会改变字号。第三段长英文说明按语义换行。仓库测试检查对应层级一致、无非等比拉伸且长句不缩小。

On the 3840×2160 canvas, section numbers use 72px, subheadings 54px, bold Chinese main statements 68px, their English translations 58px, and both Chinese and English explanations 46px. These are font-em sizes, not sentence-specific visible pixel heights; ascenders and descenders do not change font size. The long English explanation in section 03 wraps at a semantic boundary. Repository tests check consistent roles, isotropic scaling and wrapping without shrinking.

## 术语核对 / Terminology review

硬件名称“SO-101 follower”沿用[SO-101官方文档](https://huggingface.co/docs/lerobot/so101)；“end-effector”沿用[动作表示文档](https://huggingface.co/docs/lerobot/action_representations)。任务名称使用“Pick & Place”；其余说明为对应中文的项目译文。海报中的模仿学习是初始化与对照，不能替代项目要求的强化学习；算法不限定为PPO/SAC，受限末端动作也不预设为二维。

“SO-101 follower” follows the [official hardware documentation](https://huggingface.co/docs/lerobot/so101), and “end-effector” follows the [action-representation documentation](https://huggingface.co/docs/lerobot/action_representations). The task is named “Pick & Place”; other descriptions are project translations of the Chinese text. Imitation learning provides initialization and baselines rather than replacing the required reinforcement learning. Algorithms are not limited to PPO/SAC, and constrained end-effector actions are not presumed to be two-dimensional.

图片权利归属不因本项目使用而改变。

Use in this project does not change the image's ownership.

原始课程PDF、联系人信息及私有课程记录不公开。海报展示项目目标，不是训练结果或功能验收证明。

The original course PDF, contact information and private course records are not published. The poster presents project objectives, not training results or proof of functional acceptance.
