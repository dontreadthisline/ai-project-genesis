# AI Project Genesis

从概率基础到 Agentic RL 的实验驱动课程。总纲见 [docs/syllabus.md](docs/syllabus.md)。

## 目录结构

```
docs/                    课程讲义
├── syllabus.md          课程总纲
└── stage1-probability/  第一阶段讲义（Lab 01-05）
code/                    实验代码：每个实验一个独立 uv 项目
├── lab01-pdf-and-covariance/
├── lab02-joint-and-conditional/
├── lab03-monte-carlo/
├── lab04-entropy-kl/
└── lab05-mle/
```

## 使用方式

每个实验项目自包含（独立的 pyproject.toml 与 .venv），依赖仅 numpy 与 matplotlib：

```bash
cd code/lab01-pdf-and-covariance
uv sync                 # 安装依赖
uv run run_lab01.py     # 先跑自测（实现前会 NotImplementedError），再跑对照实验
```

讲义与代码的对应关系见各实验目录下的 README.md。实验产出（图表、metrics.json）写入各项目的 output/，已被 git 忽略。

跨实验约定：lab02 复用 lab01 完成的分布函数、lab05 复用一维高斯密度。需要复制时，模块内注释已标明来源。
