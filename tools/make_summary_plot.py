"""
Liten skript som jag använder för att få en helhetsbild av alla mina körningar
istället för att klicka runt mellan TensorBoards.

Vänster panel: arkitekturjämförelsen i Part 2 (FFN -> CNN -> CNN+reg).
Höger panel: hyperparameter-svepet — hur varje knapp jag vred på påverkade test_acc.

Resultat hamnar i outputs/summary_all_runs.png. Kör med:
    python3 tools/make_summary_plot.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MPL_DIR = ROOT / ".mplconfig"
_MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
SUMMARY_FILE = ROOT / "tools" / "experiments_summary.json"
TUNING_FILE = ROOT / "outputs" / "tuning_results.json"
OUTPUT_FILE = ROOT / "outputs" / "summary_all_runs.png"


def load_main_runs() -> list[dict]:
    data = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    return data["part2_main_runs"]


def load_tuning_runs() -> list[dict]:
    if not TUNING_FILE.exists():
        return []
    runs = json.loads(TUNING_FILE.read_text(encoding="utf-8"))
    return sorted(runs, key=lambda r: r["test_acc"], reverse=True)


def color_for_model(model_type: str) -> str:
    return {
        "FFN": "#94a3b8",
        "CNN-2": "#60a5fa",
        "CNN-3": "#3b82f6",
        "CNN-3+reg": "#1d4ed8",
    }.get(model_type, "#cbd5e1")


def plot_main_runs(ax: plt.Axes, runs: list[dict]) -> None:
    names = [r["name"] for r in runs]
    accs = [r["test_acc"] * 100.0 for r in runs]
    colors = [color_for_model(r["model_type"]) for r in runs]

    bars = ax.barh(names, accs, color=colors, edgecolor="white")
    ax.set_xlim(96.0, 100.0)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Part 2 – arkitekturjämförelse (MNIST)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.04,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%",
            va="center",
            fontsize=9,
        )


def plot_tuning_runs(ax: plt.Axes, runs: list[dict]) -> None:
    if not runs:
        ax.text(0.5, 0.5, "tuning_results.json saknas", ha="center", va="center")
        ax.set_axis_off()
        return

    names = [r["name"] for r in runs]
    accs = [r["test_acc"] * 100.0 for r in runs]

    best_idx = max(range(len(runs)), key=lambda i: runs[i]["test_acc"])
    colors = ["#0ea5e9" if i == best_idx else "#cbd5e1" for i in range(len(runs))]

    bars = ax.barh(names, accs, color=colors, edgecolor="white")
    ax.set_xlim(98.5, 99.5)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Part 2 – hyperparameter tuning (3-layer CNN + reg)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%",
            va="center",
            fontsize=9,
        )


def main() -> None:
    main_runs = load_main_runs()
    tuning_runs = load_tuning_runs()

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    plot_main_runs(ax_left, main_runs)
    plot_tuning_runs(ax_right, tuning_runs)

    fig.suptitle(
        "Assignment 1 – sammanfattning av alla körningar",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Sparade {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
