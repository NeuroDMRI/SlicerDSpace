<div align="center">

# 项目网站

**静态网站素材与维护指南**

[项目](../../README.zh-CN.md) | [网站](https://neurodmri.github.io/SlicerDSpace/) | [原始记录](UPSTREAM_README.md) | [English](README.md)

</div>

---

## 来源与贡献者

网站由 [xiaji](https://github.com/xiaji) 贡献，原 Git 作者为
`x1aji <jwjwei@gmail.com>`。通过普通 merge 保留 `935a5d9`、`444d8db`、
`5b09bc2` 三条原始提交，不做 squash/rebase，也不改写作者、提交者、时间及
共同作者记录。这样原贡献记录仍属于贡献者，不会被替换成维护者的新提交。

网站 HTML、样式、脚本、字体、图片、字幕和 47,838,648 字节宣传片放在
`docs/site/`。合并仅调整存放位置与测试/部署配置，不重新设计网页或转码视频。
[FACTS.md](FACTS.md)、[SPEC.md](SPEC.md)、[原 README](UPSTREAM_README.md)
保留贡献者的历史依据、设计及审核记录；它们不是当前安装指南，也不会公开部署。

---

## 性能部分

性能对照表位于主页 `#performance`，紧随 `#runtime` 阶段时间图，默认显示全部 14 项计时对比。
顶部导航与页脚的性能入口直接定位到表格，不添加整表折叠。只有每项方法与验证详情默认收起。
点击表格内的加号按钮或算法名称，在对应行下方展开方法与验证详情，再次点击可收起。
表格下不另列方法列表。按钮支持原生键盘操作；详情行直接带有 HTML `hidden` 属性，
脚本或样式未加载时也不会自动展开。脚本初始化后启用按钮；关闭脚本时仍可查看
优化报告与 JSON 数据。刷新页面后所有详情恢复收起，旧网址片段不会自动展开条目。
不另设性能子页。
`node Testing/Website/browser_check.mjs --performance` 专项检查默认状态、加减号、
键盘操作及脚本／样式加载失败时的行为。
证据分别展示等价计算实现、原生多线程、工作流调度及 Python 移植版对照，注明基线、
测量范围、输出一致性和限制，不宣称整条工作流的加速率。

10 月 30 日整理新增 10 月 29 日 MK-Curve 信号复用实验，基线为先前经过核验的
Python QR 路径。展开说明和 JSON 保留两种求解器的结果；batched 候选无稳定收益，
不改为默认路线。目录复核日期与每一条实测记录的日期分别生成。

`python3 scripts/build_performance_site.py` 从既有证据生成表格，写入 `index.html` 中
`measured-evidence` 标记之间；`--check` 核对来源哈希并拦截过期 HTML／JSON。只有筛选后的
`assets/data/performance.json` 进入公开网站，不复制原始快照、私有文件路径、被试数据或内部报告；
网页与生成工具不进入插件运行包。

主页不再展示性能筛选器、测量计数、两张重点柱状卡片或优化表格图片。
完整表格保留各条目的方法、测量范围、计时与优化类型，科研原始记录和可编辑优化图仍保留在仓库。

---

## 首页、方法表与参考文献

插图和视频不再添加圆角外框。结果区采用透明、无边框的图文排版，以细浅弧线分隔，
方法标签不加框。桌面三列、平板两列、手机单列；科研图像文件及其裁切、完整适配规则保持不变。

首页 Get started（`#start`）内介绍安装与使用 Agent Skill，提供 Skill、使用指南、CLI 模板
链接，以及可复制的安装、分析两个 Prompt，不再展示三个能力说明模块。
Agent 路线沿用手动安装的左侧编号竖线：查看 Skill、安装与启用、使用已有安装，
两个路线独立从 1 开始编号，Prompt 和复制功能不变。
命令行和队列批处理说明默认展开。中英文根 README 及论文的安装与使用
部分也提供对应示例；Skill 仅保留面向 Agent 的操作规范。下载前由用户
确认存储路径，分析前确认执行计划。Skill 文件留在仓库中，由用户自己的 Agent 执行。
旧 `#skills` 链接仍会定位到 Get started 内的 Prompt。网页助手通过本地预设的中英文资料回答并跳转到相关内容，
不发起网络请求、不持久化问题。所有屏幕尺寸均保留悬浮、拖动、眼动和对话框，不再停靠页眉。
对话框适配屏幕与键盘；`node Testing/Website/guide_bot_check.mjs` 检查交互、拖动、边界、键盘及无脚本回退。

首页主字标和 Watch the film、Get started 两个按钮居中，下方直接展示视频。
移除首页副标题、介绍段落和状态说明。移除独立的
The idea、Workflows 和 Status 部分，结果简介不再反复强调被试编号，来源记录继续保留。
“What is integrated” 位于主页最后。方法总图继续保留，六种产品组合示例
及独立 In Slicer 区块不再展示。页眉仅保留居中的导航，不展示 Logo 和预发布标记。
主区域与页脚使用透明矢量字标 `assets/img/slicerdspace-wordmark-light.svg`；
`python3 scripts/check_website_assets.py --check` 检查字标。
浏览器图标仍由 `paper-latex:docs/figures/logo-ref.svg` 导出；
`python3 scripts/check_website_assets.py --check` 检查这些图标。
Get started 提供安装路径、输入准备、
按需环境安装、方法选择及 CLI 示例。

- `assets/img/fig-product-map.png`（`#choose`）和 `#runtime` 中的
  `assets/img/fig-runtime-resources.png`
  由 `paper-latex:docs/figures/` 中可编辑的 SVG 生成，SVG 不公开发布，重建方法见 `paper-latex:docs/figures/README.md`。
  每个 `<img>` 的宽高与文件像素一致，并链接到原图。
- WMQL 卡片改用 `assets/img/res-wmql-ukf.jpg`，与 Results 图的官方多束规则 UKF
  素材逐字节一致，来源及哈希见 `paper-latex:docs/figures/src/results/wmql_sources.json`。
  `python3 scripts/check_website_assets.py --check` 校验已发布文件；更新时从独立论文仓库重新生成并导入。
  完整侧位图按比例显示，不裁切边缘；旧 CSD/ACT AF/CST 缩略图不再发布。
- 独立的 `#integrated` 部分用“What is integrated”表列出 35 项具体方法及分析工具，
  覆盖去噪、脑掩膜、解剖、微结构、追踪、束分析、连接组和选区分析，每项均有来源引用。
  表格逐项说明实现、运行依赖和调用入口，
  不再用某次测试的橙色状态标签概括整种方法。WMA 的推荐 UKF 输入、部分连接组权重的
  Python API 入口及所需外部组件继续明确列出；具体测试与科学审核记录保留在原有链接中。
- 方法表直接提供论文 DOI、作者软件或官方文档链接，鼠标悬停可查看完整引用；
  不再单列参考文献章节，也不显示数字引用索引。
  `docs/website/references.json` 保留已核对的来源目录；新增 DOI
  的出版者元数据保存在 `reference_verification_20261026.json`。
  `python3 scripts/build_website_catalog.py` 生成方法表及来源直链，`--check` 检查
  HTML 是否过期、源码入口是否存在、方法是否缺引用。文献链接由该审核目录加入允许列表；
  其他外部链接仍须加入 `Testing/Website/check_site.py` 的 `ALLOWED_URLS`。

页脚恢复与页眉一致的 2 px RGB 渐变细线，放大可点击字标，并与两列导航保持视觉平衡。
底部独立排列版权、许可声明及返回顶部入口，不加卡片框，按平板和手机宽度调整布局。
`node Testing/Website/browser_check.mjs --footer` 检查中英文、响应式布局、彩虹线与导航。
移动端底部留出悬浮助手的位置，避免遮挡页脚链接。
“关于宣传片”的制作和数据来源
记录保留在维护文档中，不在访客页脚展示。页面采用发布后的展示文案，不重复提示仓库访问权限。

---

## 发布方式

发布源为独立 [`gh-pages` 分支](https://github.com/NeuroDMRI/SlicerDSpace/tree/gh-pages)
的根目录，仅包含本地检查通过的 **`docs/site/` 快照**，包括 `index.html`、`zh.html`
和 `.nojekyll`。软件源码、内部文档、模型及独立的 `paper-latex/` 仓库不进入
发布内容；网站文件也不进入 Slicer 运行包。

发布分支从全新的根提交开始，不恢复旧的软件或论文历史，不要将这个生成分支合并回 `main`。

仓库所有者需先在 [Settings > Pages](https://github.com/NeuroDMRI/SlicerDSpace/settings/pages)
选择 **Source: Deploy from a branch**、**Branch: gh-pages**、**Folder: /(root)**，
然后保存。这个一次性的仓库设置与推送分支是两个步骤；当前仓库连接支持内容操作，
未开放 Pages 管理接口。

后续更新先运行下方的本地检查，再将审阅后的网站同步到 `gh-pages` 的独立检出目录，
保留其中的 `.git` 并清理过期发布文件。核对分支只含网站文件，再提交并推送；
不要在发布时夹带 `main` 的其他开发修改。

项目自定义的 `Website` 工作流保留手动触发，本次分支发布不需要启用或运行它。
GitHub 自身仍会执行 Pages 的构建和部署，可能显示在 Actions 界面。
以公网地址成功响应且内容与快照一致作为上线依据。

预期地址为 `https://neurodmri.github.io/SlicerDSpace/`。分享标签、canonical
URL 与 404 返回链接沿用该地址。不要设置 `main /docs`，否则会公开发布不属于
网站的内部文档。更换域名时按原始记录同步更新这些地址。

宣传片保留为普通 Git 文件，便于静态播放。仓库大文件检查仅对这一公开路径、
精确字节数和 SHA256 提供例外，不放行其他大视频或模型文件；替换视频须另行审核。

---

## 本地测试

导航右侧的 Eng / 中文胶囊沿用个人主页的样式，不影响中间导航居中。
`zh.html` 为静态中文版，与 `index.html` 共用版式、科学插图、视频和文献。
关闭 JavaScript 也能切换并阅读；启用脚本时切换保留当前章节位置。

中文文案维护在 `zh-CN.json`。修改英文首页后，运行
`python scripts/build_website_locales.py` 更新中文页；未翻译的新文案会阻止生成，
`--check` 检查两版是否同步。命令、算法标识与文献原始题名保留不变，
Agent Prompt、交互控件和项目助手随页面语言显示。

```bash
python scripts/build_website_locales.py --check
python scripts/audit_pages_site.py
python -I Testing/Website/check_site.py
npm ci --prefix Testing/Website
node Testing/Website/browser_check.mjs
```

浏览器测试默认调用已安装的 Google Chrome，也可通过 `CHROME_EXECUTABLE`
指定兼容的 Chrome for Testing。可选 `PLAYWRIGHT_CORE` 指向已有模块；不设置时
使用本地锁定版本。仍须通过视频编解码与播放检查；没有 ffprobe 时，相关可选
检查明确记录为 SKIP，不冒充已通过。

截图与浏览器结果保存在忽略的 `Testing/Website/out/`。浏览器依赖仅用于测试，
不进入网站或插件包。发布文件白名单拒绝内部文档和符号链接。

---

## 媒体加载

视频使用封面和 `preload="none"`，用户点击播放后才请求 MP4。封面位于首屏附近，
保持正常加载。图像使用原生懒加载、异步解码、明确的宽高及低请求优先级；
Results 仍保留完整矢量 SVG。

方法图与运行时间图通过 `<picture>` 优先提供无损 WebP，保留原 PNG 作为回退和
高清链接。两张图分别从 186,955 / 267,350 字节减至 73,288 / 81,220 字节，
解码后的 RGBA 像素和尺寸完全一致。

视频保留 1920×1080、30 fps、全部 1,944 帧、原有时长（64.8 秒）和原始 AAC 音轨。
使用 H.264 CRF 24 与 `+faststart` 后，从 45,452,363 字节减至 23,356,794 字节，
完整逐帧 SSIM 为 0.993841。原始母版保存在发布目录之外，不对压缩版反复转码。

```bash
python scripts/optimize_website_images.py --check
python scripts/optimize_website_video.py --source /path/to/original-master.mp4 \
  --ffmpeg /path/to/ffmpeg
```

视频命令加 `--verify-existing` 可只复核当前文件，不重新编码。凭据记录源文件与
输出哈希、尺寸、完整帧数、音轨哈希及画质检查。更换视频后须同步
`scripts/release_audit.py` 和 `Testing/Website/check_site.py` 的精确大小/哈希限制。

浏览器测试检查播放前没有 MP4 请求、点击播放与重播、移动端布局、WebP 格式选择和
禁用 JavaScript 后的原生控件。集成方法表以数字引用直接链接论文，完整题名保留在提示中。

参考：[web.dev 视频加载](https://web.dev/articles/lazy-loading-video)、
[MDN 图片加载](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/img)、
[MDN 图片格式回退](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/picture)。

---

## 证据边界

页面介绍已展示的工作流与审核证据，不表示所有算法在每个被试、平台都通过。
正式发布前保留科学与发布状态说明。视频元数据、音轨及压缩检查与原始母版的
画面和音频复核配合使用。

可编辑插图、原始素材与生成脚本由独立的 [paper-latex 仓库](https://github.com/ChengjinLii/paper-latex) 管理。本软件仓库保留网站展示图片及 SHA256 凭据。
