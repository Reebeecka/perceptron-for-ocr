"""
Liten skript som plockar upp mina redan körda experiment och skickar in dem i
MLflow — utan att träna om något. Anledningen är att jag vill ha en dashboard
där alla mina runs ligger sida vid sida, men samtidigt behålla mina egna
run-mappar som primär källa.

Läser från:
  - tools/experiments_summary.json   (huvudkörningarna i Part 2)
  - outputs/tuning_results.json      (mitt hyperparameter-svep)

Det blir två separata experiments i MLflow så de inte blandas ihop:
  - "assignment1_part2_main"   — arkitekturjämförelsen
  - "assignment1_part2_tuning" — tuning-svepet

För varje run loggas hyperparametrar + run_dir som params, key metrics
(test_loss, test_acc, best_epoch) som metrics, och PNG-artefakterna som artifacts
om de finns kvar i run-mappen.

Kör så här (med .venv aktiverad):
    python3 tools/log_to_mlflow.py
    mlflow ui --backend-store-uri ./mlruns      # dashboard på http://127.0.0.1:5000

Eller utan att aktivera .venv:
    .venv/bin/python3 tools/log_to_mlflow.py
    .venv/bin/mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_FILE = ROOT / "tools" / "experiments_summary.json"
TUNING_FILE = ROOT / "outputs" / "tuning_results.json"
MLRUNS_DIR = ROOT / "mlruns"

ARTIFACT_FILENAMES = (
    "curves_loss_acc.png",
    "confusion_matrix_percent.png",
    "examples_correct.png",
    "examples_incorrect.png",
    "training_config.json",
)


def log_artifacts_from_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    for name in ARTIFACT_FILENAMES:
        artifact = run_dir / name
        if artifact.exists():
            mlflow.log_artifact(str(artifact))


def log_main_runs() -> None:
    data = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    runs = data["part2_main_runs"]

    mlflow.set_experiment("assignment1_part2_main")
    for run in runs:
        run_dir = ROOT / run["run_dir"]
        with mlflow.start_run(run_name=run["name"]):
            mlflow.log_param("run_dir", run["run_dir"])
            mlflow.log_param("model_type", run["model_type"])
            mlflow.log_param("augmentation", run["augmentation"])
            mlflow.log_param("regularization", run["regularization"])
            mlflow.log_param("epochs", run["epochs"])

            mlflow.log_metric("test_loss", run["test_loss"])
            mlflow.log_metric("test_acc", run["test_acc"])
            mlflow.log_metric("best_epoch", run["best_epoch"])

            log_artifacts_from_run_dir(run_dir)
            print(f"  logged main: {run['name']}")


def log_tuning_runs() -> None:
    if not TUNING_FILE.exists():
        print("  tuning_results.json saknas – hoppar över tuning-svepet")
        return

    runs = json.loads(TUNING_FILE.read_text(encoding="utf-8"))
    mlflow.set_experiment("assignment1_part2_tuning")
    for run in runs:
        run_dir = ROOT / run["run_dir"]
        with mlflow.start_run(run_name=run["name"]):
            mlflow.log_param("run_dir", run["run_dir"])
            mlflow.log_param("lr", run["lr"])
            mlflow.log_param("dropout_p", run["dropout_p"])
            mlflow.log_param("weight_decay", run["weight_decay"])
            mlflow.log_param("gaussian_noise_std", run["gaussian_noise_std"])
            mlflow.log_param("conv_channels", str(run["conv_channels"]))
            mlflow.log_param("fc_hidden", run["fc_hidden"])
            mlflow.log_param("batch_size", run["batch_size"])
            mlflow.log_param("epochs", run["epochs"])

            mlflow.log_metric("test_loss", run["test_loss"])
            mlflow.log_metric("test_acc", run["test_acc"])
            mlflow.log_metric("best_val_loss", run["best_val_loss"])
            mlflow.log_metric("best_epoch", run["best_epoch"])
            mlflow.log_metric("total_train_seconds", run["total_train_seconds"])

            log_artifacts_from_run_dir(run_dir)
            print(f"  logged tuning: {run['name']}")


def main() -> None:
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    print(f"Tracking URI: file:{MLRUNS_DIR}")

    print("Loggar Part 2 huvudkörningar...")
    log_main_runs()

    print("Loggar Part 2 tuning-svep...")
    log_tuning_runs()

    print()
    print("Klart. Öppna dashboarden med:")
    print("    mlflow ui --backend-store-uri ./mlruns")
    print("Och gå till http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
