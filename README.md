# AI Project Genesis

从概率基础到 Agentic RL 的实验驱动课程。总纲见 [docs/syllabus.md](docs/syllabus.md)。

## 目录结构

```
docs/                    内容层：讲义 + manifest.yml（书籍结构单一事实源）
├── manifest.yml         分部与章节顺序，渲染层由此生成目录配置
├── preface.md           前言
├── syllabus.md          课程总纲
└── stage1..6-*/         六个阶段的讲义（第一阶段含 5 份实验讲义，其余为阶段导览）
code/                    内容层：实验代码，每个实验一个独立 uv 项目
book_render/                    渲染层（唯一入口 build-book.sh）
├── themes/               主题 = 标准 Quarto 扩展，自包含可独立发布
│   ├── genesis-book/     litera 底座 + 无衬线正文 + 阅读模式
│   └── paper-book/       journal 底座 + 衬线正文（纸书观感）
├── vendor/               自托管 MathJax
└── scripts/              gen-quarto-yml.py / gen-vitepress.py / build-book.sh
_quarto.yml / _book/ / _extensions/   生成物，不入库
```

## 书籍构建

内容与渲染分离：`docs/manifest.yml` 描述书籍结构，渲染器只消费该清单。

```bash
./build.sh                    # 构建书籍（默认 genesis-book 主题）
./build.sh book indigo-atlas  # 换主题构建（paper-book / indigo-atlas）
./build.sh serve              # 本地预览（http://localhost:8765）
./build.sh check              # 全量检查：构建校验 + 实验代码红灯
./build.sh install            # quarto add 重装线上主题（发布闭环）
./build.sh release "msg"      # 主题发布（同步 + push 到发布仓库）
```
渲染原语（manifest 生成、MathJax 本地化、resources 处理）在 `book_render/scripts/`；`build.sh` 是工作流编排入口。

换渲染工具（如 VitePress、mdBook）时：在 book_render/scripts/ 新增一个 manifest 生成器即可，`docs/` 与 `code/` 零改动。

主题已独立发布：https://github.com/dontreadthisline/quarto-book-themes
外部项目 `quarto add dontreadthisline/quarto-book-themes` 后 `format: genesis-book-html`（或 `paper-book-html`）即用。
主题同步流程：本仓库 `book_render/themes/` 为单一事实源，GitHub zip 安装要求 `_extensions/<org>/<name>/` 布局，同步并推送：

    rsync -a --delete --exclude '.DS_Store' book_render/themes/genesis-book \
        ~/my_proj/quarto-book-themes/_extensions/dontreadthisline/genesis-book
    rsync -a --delete --exclude '.DS_Store' book_render/themes/paper-book \
        ~/my_proj/quarto-book-themes/_extensions/dontreadthisline/paper-book
    cd ~/my_proj/quarto-book-themes && git add -A && git commit -m '...' && git push

## 图表规范（indigo-atlas 手绘体系）

- 每张图独立成一个单独的 HTML 页面（sketch.js 作为绘图语言），正文以链接/嵌入方式引用，不在正文中拼绘图代码。
- 图中出现的公式与文字一律用 LaTeX 单独渲染（MathJax），**永远不用 unicode 拼接**（不手写下标字符、不用全角上下标模拟公式）。

## 实验环境

讲义（docs/）已全部完成；实验代码骨架（code/）第一阶段就绪，第二至第六阶段待建——每个实验的讲义§「实现任务」即为其代码骨架的规约。

```bash
cd code/lab01-pdf-and-covariance
uv sync                 # 安装依赖
uv run run_lab01.py     # 先跑自测（实现前会 NotImplementedError），再跑对照实验
```

讲义与代码的对应关系见各实验目录下的 README.md。实验产出（图表、metrics.json）写入各项目的 output/，已被 git 忽略。

跨实验约定：lab02 复用 lab01 完成的分布函数、lab05 复用一维高斯密度。需要复制时，模块内注释已标明来源。
