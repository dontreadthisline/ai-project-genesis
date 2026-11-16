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
assets/mathjax/          渲染层资源：自托管 MathJax
scripts/                 渲染层：build-book.sh（唯一构建入口）、gen-quarto-yml.py
_quarto.yml / _book/     生成物，不入库
```

## 书籍构建

内容与渲染分离：`docs/manifest.yml` 描述书籍结构，渲染器只消费该清单。

```bash
./scripts/build-book.sh   # manifest -> _quarto.yml -> quarto render -> MathJax 本地化 + 校验
open _book/index.html     # 本地阅读，断网也可正常渲染公式
```

换渲染工具（如 VitePress、mdBook）时：在 scripts/ 新增一个 manifest 生成器即可，`docs/` 与 `code/` 零改动。

## 实验环境

```bash
cd code/lab01-pdf-and-covariance
uv sync                 # 安装依赖
uv run run_lab01.py     # 先跑自测（实现前会 NotImplementedError），再跑对照实验
```

讲义与代码的对应关系见各实验目录下的 README.md。实验产出（图表、metrics.json）写入各项目的 output/，已被 git 忽略。

跨实验约定：lab02 复用 lab01 完成的分布函数、lab05 复用一维高斯密度。需要复制时，模块内注释已标明来源。
