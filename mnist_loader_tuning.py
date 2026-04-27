"""
Hyperparameter-tuning för min 3-layer CNN med regularization.

Idén är att jag varierar en knapp i taget (dropout, lr, weight decay, brus, kanalbredd,
fc-storlek) och låter resten av configen vara densamma. Då blir det lättare att se
vad som faktiskt påverkar resultatet och vad som mest är slump.

Varje körning hamnar i outputs/run_<timestamp>_<name>/ och sammanställningen
landar i outputs/tuning_results.json så jag kan sortera dem efter test_acc i efterhand.
"""
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# ---------------------------------------------------------------------------
# Mina experiment — en rad = en körning.
# Lätt att lägga till en ny konfig om jag vill prova någonting senare.
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS: list[dict] = [
    {
        "name": "baseline",
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "dropout_p": 0.30,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.10,
        "conv_channels": [8, 16, 32],
        "fc_hidden": 128,
    },
    {
        "name": "lower_dropout",
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "dropout_p": 0.10,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.10,
        "conv_channels": [8, 16, 32],
        "fc_hidden": 128,
    },
    {
        "name": "higher_dropout",
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "dropout_p": 0.50,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.10,
        "conv_channels": [8, 16, 32],
        "fc_hidden": 128,
    },
    {
        "name": "lower_lr",
        "batch_size": 64,
        "epochs": 5,
        "lr": 5e-4,
        "dropout_p": 0.30,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.10,
        "conv_channels": [8, 16, 32],
        "fc_hidden": 128,
    },
    {
        "name": "higher_weight_decay",
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "dropout_p": 0.30,
        "weight_decay": 5e-4,
        "gaussian_noise_std": 0.10,
        "conv_channels": [8, 16, 32],
        "fc_hidden": 128,
    },
    {
        "name": "low_noise",
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "dropout_p": 0.30,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.05,
        "conv_channels": [8, 16, 32],
        "fc_hidden": 128,
    },
    {
        "name": "wider_channels",
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "dropout_p": 0.30,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.10,
        "conv_channels": [16, 32, 64],
        "fc_hidden": 128,
    },
    {
        "name": "bigger_fc",
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "dropout_p": 0.30,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.10,
        "conv_channels": [8, 16, 32],
        "fc_hidden": 256,
    },
]


# ---------------------------------------------------------------------------
# Modellen — samma 3-layer CNN som tidigare, men jag har gjort allt
# konfigurerbart (kanaler, fc-storlek, dropout) så att samma kod kan användas
# för alla mina experiment.
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, conv_channels: list[int], fc_hidden: int, dropout_p: float):
        super().__init__()
        c1, c2, c3 = conv_channels

        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        # 28x28 -> pool -> 14x14 -> pool -> 7x7
        self.fc1 = nn.Linear(c3 * 7 * 7, fc_hidden)
        self.bn_fc1 = nn.BatchNorm1d(fc_hidden)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(fc_hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.bn_fc1(self.fc1(x))))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Hjälpfunktioner som jag återanvänder mellan körningar
# ---------------------------------------------------------------------------
class AddGaussianNoise:
    def __init__(self, std: float):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return x + torch.randn_like(x) * self.std


def hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / max(1, len(loader)), correct / max(1, total)


def build_transforms(config: dict) -> tuple:
    """Returnerar (train_transform, eval_transform) utifrån config."""
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=15,
                translate=(0.10, 0.10),
                scale=(0.90, 1.10),
                shear=10,
                fill=0,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(std=config["gaussian_noise_std"]),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return train_transform, eval_transform


def make_curves_figure(history: dict) -> plt.Figure:
    xs = list(range(1, len(history["train_loss"]) + 1))
    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(xs, history["train_loss"], label="train")
    plt.plot(xs, history["val_loss"], label="val")
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(xs, history["train_acc"], label="train")
    plt.plot(xs, history["val_acc"], label="val")
    plt.title("Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# run_experiment — kör en enskild konfiguration och returnerar resultatraden
# ---------------------------------------------------------------------------
def run_experiment(config: dict, device: torch.device, base_out: str) -> dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_out, f"run_{run_id}_{config['name']}")
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("00_Info/config", json.dumps(config, indent=2, sort_keys=True), 0)
    writer.add_text("00_Info/device", str(device), 0)

    torch.manual_seed(config.get("seed", 42))

    train_transform, eval_transform = build_transforms(config)

    full_train_clean = datasets.MNIST(root="./data", train=True, download=True, transform=eval_transform)
    full_train_aug = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=eval_transform)

    val_size = min(config.get("val_size", 10_000), len(full_train_clean) - 1)
    train_size = len(full_train_clean) - val_size
    train_part, val_part = random_split(
        full_train_clean,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )

    train_set = Subset(full_train_aug, train_part.indices)
    val_set = Subset(full_train_clean, val_part.indices)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    model = Net(
        conv_channels=config["conv_channels"],
        fc_hidden=config["fc_hidden"],
        dropout_p=config["dropout_p"],
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    history: dict = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0

    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            train_loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            loss.backward()
            opt.step()

        train_loss_avg = train_loss_sum / max(1, len(train_loader))
        train_acc_avg = train_correct / max(1, train_total)
        val_loss_avg, val_acc_avg = eval_epoch(model, val_loader, loss_fn, device)
        epoch_seconds = time.time() - epoch_t0

        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc_avg)

        elapsed = hhmmss(time.time() - start_time)
        print(
            f"  [{config['name']}] epoch={epoch} time={elapsed} "
            f"train_loss={train_loss_avg:.4f} train_acc={train_acc_avg:.4f} "
            f"val_loss={val_loss_avg:.4f} val_acc={val_acc_avg:.4f}"
        )

        writer.add_scalar("01_Förlust (loss)/train", train_loss_avg, epoch)
        writer.add_scalar("01_Förlust (loss)/val", val_loss_avg, epoch)
        writer.add_scalar("02_Träffsäkerhet (accuracy)/train", train_acc_avg, epoch)
        writer.add_scalar("02_Träffsäkerhet (accuracy)/val", val_acc_avg, epoch)
        writer.add_scalar("03_Tid/epok (sekunder)", epoch_seconds, epoch)
        writer.add_scalar("04_Hastighet/bilder per sekund", train_total / max(1e-9, epoch_seconds), epoch)

        fig = make_curves_figure(history)
        writer.add_figure("05_Figurer/kurvor (loss + accuracy)", fig, global_step=epoch)
        plt.close(fig)

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
             "history": history, "config": config},
            os.path.join(out_dir, f"epoch_{epoch:03d}.pt"),
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "history": history, "config": config},
                os.path.join(out_dir, "best.pt"),
            )
            writer.add_scalar("00_KeyNumbers/bästa val-loss", best_val_loss, epoch)

    total_train_seconds = time.time() - start_time
    writer.add_scalar("00_KeyNumbers/total träningstid (sekunder)", total_train_seconds, 0)

    # Spara kurv-figur
    fig = make_curves_figure(history)
    fig.savefig(os.path.join(out_dir, "curves_loss_acc.png"), dpi=160)
    plt.close(fig)

    # Test-evaluering med bästa checkpoint
    best = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
    print(
        f"  [{config['name']}] KLAR – "
        f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
        f"tid={hhmmss(total_train_seconds)}"
    )

    writer.add_scalar("00_KeyNumbers/test loss", test_loss, 0)
    writer.add_scalar("00_KeyNumbers/test accuracy", test_acc, 0)
    writer.close()

    return {
        "name": config["name"],
        "run_dir": out_dir,
        "lr": config["lr"],
        "dropout_p": config["dropout_p"],
        "weight_decay": config["weight_decay"],
        "gaussian_noise_std": config["gaussian_noise_std"],
        "conv_channels": config["conv_channels"],
        "fc_hidden": config["fc_hidden"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "total_train_seconds": round(total_train_seconds, 1),
    }


# ---------------------------------------------------------------------------
# main — loopar över alla configs, kör dem en i taget och sparar sammanfattningen
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    base_out = "outputs"
    os.makedirs(base_out, exist_ok=True)

    all_results: list[dict] = []

    for i, config in enumerate(EXPERIMENT_CONFIGS, start=1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(EXPERIMENT_CONFIGS)}: {config['name']}")
        print(f"{'='*60}")
        result = run_experiment(config, device, base_out)
        all_results.append(result)

    # Sortera på test_acc (bäst överst) och skriv ut sammanfattning
    all_results.sort(key=lambda r: r["test_acc"], reverse=True)

    print(f"\n{'='*60}")
    print("SAMMANFATTNING (sorterad på test accuracy)")
    print(f"{'='*60}")
    col_w = 22
    print(
        f"{'Namn':<{col_w}} {'test_acc':>9} {'test_loss':>10} "
        f"{'val_loss':>9} {'best_ep':>7} {'tid (s)':>8}"
    )
    print("-" * (col_w + 9 + 10 + 9 + 7 + 8 + 5))
    for r in all_results:
        print(
            f"{r['name']:<{col_w}} {r['test_acc']:>9.4f} {r['test_loss']:>10.4f} "
            f"{r['best_val_loss']:>9.4f} {r['best_epoch']:>7} {r['total_train_seconds']:>8.1f}"
        )

    # Spara som JSON
    results_path = os.path.join(base_out, "tuning_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, sort_keys=False)
    print(f"\nResultat sparade i: {results_path}")


if __name__ == "__main__":
    main()
