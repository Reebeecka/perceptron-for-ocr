import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "outputs_catsAndDogs"


def load_config(run_dir: Path) -> dict | None:
    p = run_dir / "training_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def guess_tag(cfg: dict | None) -> str:
    if not cfg:
        return "UNKNOWN"
    mode = cfg.get("mode", "UNKNOWN")
    img_size = cfg.get("img_size", None)
    max_pool = cfg.get("max_train_pool", None)

    parts: list[str] = ["catsAndDogs", str(mode)]
    if img_size is not None:
        parts.append(f"img{img_size}")
    if max_pool is not None:
        parts.append(f"pool{max_pool}")
    return "__".join(parts)


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "confusion_matrix_percent.png").exists() and (run_dir / "curves_loss_acc.png").exists()


def main():
    if not RUNS_DIR.exists():
        raise SystemExit(f"Missing: {RUNS_DIR}")

    for run_dir in sorted(RUNS_DIR.glob("run_*")):
        cfg = load_config(run_dir)
        tag = guess_tag(cfg)
        status = "OK" if is_complete(run_dir) else "INCOMPLETE"

        # Extract old timestamp from run_YYYYMMDD_HHMMSS
        old_name = run_dir.name
        ts = old_name.replace("run_", "")

        new_name = f"{tag}__{status}__run_{ts}"
        new_dir = run_dir.parent / new_name
        if new_dir.exists():
            continue
        run_dir.rename(new_dir)
        print(f"{old_name} -> {new_name}")


if __name__ == "__main__":
    main()

