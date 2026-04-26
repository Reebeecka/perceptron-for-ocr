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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))          # 1x28x28 -> 8x28x28
        x = self.pool(self.act(self.conv2(x)))  # 8x28x28 -> 16x28x28 -> 16x14x14
        x = self.pool(self.act(self.conv3(x)))  # 16x14x14 -> 32x14x14 -> 32x7x7
        x = torch.flatten(x, 1) # 32x7x7 -> 1568
        x = self.act(self.fc1(x))
        x = self.fc2(x) 
        return x

        #28x28 -> conv -> relu -> pool -> conv -> relu -> pool -> flatten -> linear

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
    # x: (1, H, W) normalized with mean=0.1307 std=0.3081
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
    # Each row shows one original image followed by multiple augmented versions
    # of the same image, which is easier for beginners to interpret.
    imgs: list[torch.Tensor] = []
    for base_idx in base_indices:
        original_x, _ = original_dataset[base_idx]
        imgs.append(unnormalize_mnist(original_x).clamp(0.0, 1.0))
        for _ in range(variants_per_image):
            aug_x, _ = dataset_with_aug[base_idx]  # stochastic transform is applied here
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
    # Build 10x10 confusion matrix where:
    # x-axis = true label, y-axis = predicted label
    cm = [[0 for _ in range(10)] for _ in range(10)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t <= 9 and 0 <= p <= 9:
            cm[p][t] += 1

    # Convert counts -> percent of TRUE class (column-wise since x=true).
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
        "device": str(device),
    }
    print("training_config=" + json.dumps(training_config, indent=2, sort_keys=True))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"mnist__3layer_cnn__run_{run_id}")
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
                "Den här körningen tränar en enkel neural network-modell på MNIST (hand-skrivna siffror).",
                "",
                "### Viktiga grafer",
                "- **01_Förlust (Loss)**: lägre är bättre. Visar hur 'fel' modellen har.",
                "- **02_Träffsäkerhet (Accuracy)**: högre är bättre. Andel rätt klassificeringar.",
                "- **Train vs Val**:",
                "  - **Train** = data modellen tränar på.",
                "  - **Val** = data som *inte* används för att uppdatera vikter (bra för att upptäcka overfitting).",
                "",
                "### Overfitting (övertränning) – tumregel",
                "- Om **train loss går ner** men **val loss går upp**: modellen kan börja överanpassa sig.",
                "- Därför sparar vi **best.pt** från epoken med lägst val-loss (inte nödvändigtvis sista epoken).",
                "",
                "### Nyckeltal",
                "- **00_KeyNumbers/** innehåller sammanfattning (best epoch, test-accuracy, total tid).",
                "- **03_Tid/** visar tid per epok och hur länge körningen har pågått.",
                "- **04_Hastighet/** visar ungefärlig throughput (bilder per sekund) under träning.",
            ]
        ),
        0,
    )
    with open(os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, sort_keys=True)

    torch.manual_seed(training_config["seed"])

    # Data augmentation should apply ONLY to training data.
    # Validation + test should use a clean, deterministic transform.
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
            AddGaussianNoise(std=0.10),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Build the split indices from a dataset with eval transforms to keep val deterministic.
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

    # Same indices, but different transforms:
    # - train uses augmentation
    # - val uses clean eval transform
    train_set = Subset(full_train_aug, train_part.indices)
    val_set = Subset(full_train_for_split, val_part.indices)

    # Log a visual preview of augmentation to TensorBoard (Images tab).
    # Each row shows: original | augmented | augmented | augmented
    preview_indices = [int(idx) for idx in train_part.indices[:4]]
    log_augmentation_preview(writer, full_train_for_split, full_train_aug, preview_indices, variants_per_image=3)

    writer.add_text(
        "00_Info/augmentation",
        "\n".join(
            [
                "Train augmentation:",
                "- RandomAffine: degrees=15, translate=0.10, scale=(0.90,1.10), shear=10",
                "- Gaussian noise: std=0.10",
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

    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=training_config["lr"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0

    start_time = time.time()
    #Jag valde att stanna vid 5 Epochs eftersom test loss fortsätter ner och test accuracy fortsätter stiga. Men stoppade eftersom förändringen blev liten från 4 - 5
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
            train_loss_sum += loss.item() #Gör om loss till ett float och adderar det till train_loss_sum
            preds = logits.argmax(dim=1)    #Vilken klass har högst "score"
            train_correct += (preds == y).sum().item() #Antal korrekta prediktioner
            train_total += y.size(0) #Antal bilder i batchen
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

        # TensorBoard scalars + performance
        writer.add_scalar("01_Förlust (loss)/train", train_loss_avg, epoch)
        writer.add_scalar("01_Förlust (loss)/val", val_loss_avg, epoch)
        writer.add_scalar("02_Träffsäkerhet (accuracy)/train", train_acc_avg, epoch)
        writer.add_scalar("02_Träffsäkerhet (accuracy)/val", val_acc_avg, epoch)
        writer.add_scalar("03_Tid/epok (sekunder)", epoch_seconds, epoch)
        writer.add_scalar("03_Tid/körning hittills (sekunder)", time.time() - start_time, epoch)
        writer.add_scalar("04_Hastighet/bilder per sekund (train)", train_total / max(1e-9, epoch_seconds), epoch)

        # Curves figure (nice dashboard); log once per epoch.
        fig = make_curves_figure(history)
        writer.add_figure("05_Figurer/kurvor (loss + accuracy)", fig, global_step=epoch)
        plt.close(fig)

        # Save checkpoint each epoch (simple + safe).
        ckpt_epoch_path = os.path.join(out_dir, f"epoch_{epoch:03d}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "history": history, "config": training_config}, ckpt_epoch_path)

        # Track best model by validation loss.
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "history": history, "config": training_config}, os.path.join(out_dir, "best.pt"))
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

    # Use ONLY the best validation-loss checkpoint for the final test evaluation.
    best = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
    print(f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    writer.add_scalar("00_KeyNumbers/test loss (lägre är bättre)", test_loss, 0)
    writer.add_scalar("00_KeyNumbers/test accuracy (högre är bättre)", test_acc, 0)

    # Collect predictions for confusion matrix + examples (TEST).
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

    # TensorBoard figures for evaluation artifacts.
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