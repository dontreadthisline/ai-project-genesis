"""Lab 20 入口: selftest -> 对照实验 -> output/metrics.json + figures."""
from selftest import check_implementation


def e1(metrics: dict) -> None:
    """对照实验 E1: 讲义 §3 表格该行。"""
    raise NotImplementedError("TODO: E1")

def e2(metrics: dict) -> None:
    """对照实验 E2: 讲义 §3 表格该行。"""
    raise NotImplementedError("TODO: E2")

def e3(metrics: dict) -> None:
    """对照实验 E3: 讲义 §3 表格该行。"""
    raise NotImplementedError("TODO: E3")

def e4(metrics: dict) -> None:
    """对照实验 E4: 讲义 §3 表格该行。"""
    raise NotImplementedError("TODO: E4")


def main() -> None:
    check_implementation()
    metrics: dict = {}
    e1(metrics)
    e2(metrics)
    e3(metrics)
    e4(metrics)
    import json
    from pathlib import Path
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
