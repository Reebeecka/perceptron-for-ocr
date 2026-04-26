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


class Net(nn.Module):
    def __init__(self, dropout_p: float = 0.30):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))          # 1x28x28 -> 8x28x28
        x = self.pool(self.act(self.bn2(self.conv2(x))))  # 8x28x28 -> 16x14x14
        x = self.pool(self.act(self.bn3(self.conv3(x))))  # 16x14x14 -> 32x7x7
        x = torch.flatten(x, 1)                        # 32x7x7 -> 1568
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def eval_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
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


class AddGaussianNoise:
    def __init__(self, std: float = 0.10):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return x + torch.randn_like(x) * self.std


def unnormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    mean = 0.1307
    std = 0.3081
    return (x * std) + mean


def log_augmentation_preview(
    writer: SummaryWriter,
    original_dataset: datasets.MNIST,
    dataset_with_aug: datasets.MNIST,
    base_indices: list[int],
    variants_per_image: int = 3,
    tag: str = "06_Augmentation/original + augmentationer",
) -> None:
    imgs: list[torch.Tensor] = []
    for base_idx in base_indices:
        original_x, _ = original_dataset[base_idx]
        imgs.append(unnormalize_mnist(original_x).clamp(0.0, 1.0))
        for _ in range(variants_per_image):
            aug_x, _ = dataset_with_aug[base_idx]
            imgs.append(unnormalize_mnist(aug_x).clamp(0.0, 1.0))

    grid = make_grid(torch.stack(imgs, dim=0), nrow=variants_per_image + 1, padding=2)
    writer.add_image(tag, grid, global_step=0)


def make_curves_figure(history: dict[str, list[float]]):
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


def save_curves(history: dict[str, list[float]], out_dir: str) -> None:
    fig = make_curves_figure(history)
    fig.savefig(os.path.join(out_dir, "curves_loss_acc.png"), dpi=160)
    plt.close(fig)


def make_confusion_matrix_percent_figure(y_true: list[int], y_pred: list[int]):
    cm = [[0 for _ in range(10)] for _ in range(10)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t <= 9 and 0 <= p <= 9:
            cm[p][t] += 1

    cm_pct = [[0.0 for _ in range(10)] for _ in range(10)]
    for true_label in range(10):
        col_sum = 0
        for pred_label in range(10):
            col_sum += cm[pred_label][true_label]
        if col_sum == 0:
            col_sum = 1
        for pred_label in range(10):
            cm_pct[pred_label][true_label] = (cm[pred_label][true_label] / col_sum) * 100.0

    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm_pct, cmap="Blues", interpolation="nearest")
    plt.title("Confusion matrix (% of TRUE class) on TEST")
    plt.xlabel("True label (x)")
    plt.ylabel("Predicted label (y)")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{cm_pct[i][j]:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    return fig


def save_confusion_matrix_percent(y_true: list[int], y_pred: list[int], out_dir: str) -> None:
    fig = make_confusion_matrix_percent_figure(y_true, y_pred)
    fig.savefig(os.path.join(out_dir, "confusion_matrix_percent.png"), dpi=160)
    plt.close(fig)


def make_examples_figure(
    xs: list[torch.Tensor],
    y_true: list[int],
    y_pred: list[int],
    want_correct: bool,
    max_examples: int = 12,
):
    picked = []
    for i in range(len(y_true)):
        is_correct = (y_true[i] == y_pred[i])
        if is_correct == want_correct:
            picked.append(i)
        if len(picked) >= max_examples:
            break

    if len(picked) == 0:
        return None

    cols = 4
    rows = (len(picked) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for k, idx in enumerate(picked, start=1):
        img = xs[idx].squeeze(0).numpy()
        plt.subplot(rows, cols, k)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"pred={y_pred[idx]}, true={y_true[idx]}")
    plt.tight_layout()
    return fig


def save_examples(
    xs: list[torch.Tensor],
    y_true: list[int],
    y_pred: list[int],
    out_dir: str,
    filename: str,
    want_correct: bool,
    max_examples: int = 12,
) -> None:
    fig = make_examples_figure(xs, y_true, y_pred, want_correct=want_correct, max_examples=max_examples)
    if fig is None:
        return
    fig.savefig(os.path.join(out_dir, filename), dpi=160)
    plt.close(fig)


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    training_config = {
        "seed": 42,
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
        "val_size": 10_000,
        "dropout_p": 0.30,
        "weight_decay": 1e-4,
        "gaussian_noise_std": 0.10,
    }
    print("training_config=" + json.dumps(training_config, indent=2, sort_keys=True))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"mnist__3layer_regularization__run_{run_id}")
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("00_Info/training_config", json.dumps(training_config, indent=2, sort_keys=True), 0)
    writer.add_text("00_Info/device", str(device), 0)
    writer.add_text(
        "00_Info/README",
        "\n".join(
            [
                "## Vad visar TensorBoard här?",
                "",
                "Den här körningen tränar en 3-layer CNN på MNIST med regularization.",
                "",
                "### Regularization som används",
                "- **Batch normalization** efter conv-lager och första FC-lagret",
                "- **Dropout** efter första FC-lagret",
                "- **Weight decay** i Adam-optimizern",
                "- **Noise injection** via Gaussian noise i train augmentation",
                "",
                "### Tolkning",
                "- Regularization används för att minska overfitting.",
                "- Train-resultat kan ibland bli lite sämre, men val/test kan bli bättre eller stabilare.",
            ]
        ),
        0,
    )
    with open(os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, sort_keys=True)

    torch.manual_seed(training_config["seed"])

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
            AddGaussianNoise(std=training_config["gaussian_noise_std"]),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full_train_for_split = datasets.MNIST(root="./data", train=True, download=True, transform=eval_transform)
    full_train_aug = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=eval_transform)

    val_size = min(training_config["val_size"], len(full_train_for_split) - 1)
    train_size = len(full_train_for_split) - val_size
    train_part, val_part = random_split(
        full_train_for_split,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(training_config["seed"]),
    )

    train_set = Subset(full_train_aug, train_part.indices)
    val_set = Subset(full_train_for_split, val_part.indices)

    preview_indices = [int(idx) for idx in train_part.indices[:4]]
    log_augmentation_preview(writer, full_train_for_split, full_train_aug, preview_indices, variants_per_image=3)

    writer.add_text(
        "00_Info/augmentation",
        "\n".join(
            [
                "Train augmentation:",
                "- RandomAffine: degrees=15, translate=0.10, scale=(0.90,1.10), shear=10",
                f"- Gaussian noise: std={training_config['gaussian_noise_std']:.2f}",
                "",
                "Regularization in model:",
                f"- Dropout: p={training_config['dropout_p']:.2f}",
                f"- Weight decay: {training_config['weight_decay']:.1e}",
                "- BatchNorm: används efter conv-lager och fc1",
                "",
                "Val/Test:",
                "- ToTensor + Normalize (ingen augmentation)",
            ]
        ),
        0,
    )

    train_loader = DataLoader(train_set, batch_size=training_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=training_config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=training_config["batch_size"], shuffle=False)

    model = Net(dropout_p=training_config["dropout_p"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0

    start_time = time.time()
    for epoch in range(1, training_config["epochs"] + 1):
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

        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc_avg)

        epoch_seconds = time.time() - epoch_t0

        elapsed = hhmmss(time.time() - start_time)
        print(
            f"epoch={epoch} time={elapsed} "
            f"train_loss={train_loss_avg:.4f} train_acc={train_acc_avg:.4f} "
            f"val_loss={val_loss_avg:.4f} val_acc={val_acc_avg:.4f}"
        )

        writer.add_scalar("01_Förlust (loss)/train", train_loss_avg, epoch)
        writer.add_scalar("01_Förlust (loss)/val", val_loss_avg, epoch)
        writer.add_scalar("02_Träffsäkerhet (accuracy)/train", train_acc_avg, epoch)
        writer.add_scalar("02_Träffsäkerhet (accuracy)/val", val_acc_avg, epoch)
        writer.add_scalar("03_Tid/epok (sekunder)", epoch_seconds, epoch)
        writer.add_scalar("03_Tid/körning hittills (sekunder)", time.time() - start_time, epoch)
        writer.add_scalar("04_Hastighet/bilder per sekund (train)", train_total / max(1e-9, epoch_seconds), epoch)

        fig = make_curves_figure(history)
        writer.add_figure("05_Figurer/kurvor (loss + accuracy)", fig, global_step=epoch)
        plt.close(fig)

        ckpt_epoch_path = os.path.join(out_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "history": history,
                "config": training_config,
            },
            ckpt_epoch_path,
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "history": history,
                    "config": training_config,
                },
                os.path.join(out_dir, "best.pt"),
            )
            writer.add_scalar("00_KeyNumbers/bästa val-loss (lägre är bättre)", best_val_loss, epoch)
            writer.add_scalar("00_KeyNumbers/bästa epok (epoch index)", float(best_epoch), epoch)
            writer.add_text(
                "00_KeyNumbers/best checkpoint",
                f"best_epoch={best_epoch}\n"
                f"best_val_loss={best_val_loss:.6f}\n"
                f"checkpoint=best.pt\n",
                epoch,
            )

    total_train_seconds = time.time() - start_time
    writer.add_scalar("00_KeyNumbers/total träningstid (sekunder)", total_train_seconds, 0)
    save_curves(history, out_dir)

    best = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
    print(f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    writer.add_scalar("00_KeyNumbers/test loss (lägre är bättre)", test_loss, 0)
    writer.add_scalar("00_KeyNumbers/test accuracy (högre är bättre)", test_acc, 0)

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    xs: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            xs.extend(x.cpu())

    save_confusion_matrix_percent(y_true, y_pred, out_dir)
    save_examples(xs, y_true, y_pred, out_dir, "examples_correct.png", want_correct=True, max_examples=12)
    save_examples(xs, y_true, y_pred, out_dir, "examples_incorrect.png", want_correct=False, max_examples=12)

    cm_fig = make_confusion_matrix_percent_figure(y_true, y_pred)
    writer.add_figure("05_Figurer/confusion matrix (% av TRUE klass)", cm_fig, global_step=0)
    plt.close(cm_fig)

    correct_fig = make_examples_figure(xs, y_true, y_pred, want_correct=True, max_examples=12)
    if correct_fig is not None:
        writer.add_figure("05_Figurer/exempel – korrekta prediktioner", correct_fig, global_step=0)
        plt.close(correct_fig)

    incorrect_fig = make_examples_figure(xs, y_true, y_pred, want_correct=False, max_examples=12)
    if incorrect_fig is not None:
        writer.add_figure("05_Figurer/exempel – felklassificeringar", incorrect_fig, global_step=0)
        plt.close(incorrect_fig)

    writer.close()


if __name__ == "__main__":
    main()
