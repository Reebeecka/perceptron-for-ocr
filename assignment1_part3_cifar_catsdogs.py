import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

# Tvinga matplotlib att skriva sin cache i workspacet — i vissa miljöer (t.ex. när jag
# kör utanför min vanliga shell) får den inte skriva i ~/.matplotlib och då dör hela
# importen tyst.
_MPL_DIR = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)
os.environ.setdefault("MPLBACKEND", "Agg")  # ingen GUI-backend, det kraschar bara på vissa setups

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.15
    test_frac: float = 0.15
    img_size: int = 64  # mindre = snabbare, jag kör på 8 GB RAM
    mode: str = "scratch"  # "scratch" | "resnet50"
    max_train_pool: int | None = None  # krymp datasetet om jag bara vill testa snabbt


def _get_mode(default: str = "scratch") -> str:
    # Jag väljer mode via env-var så jag slipper redigera koden mellan körningar:
    # MODE=scratch eller MODE=resnet50
    mode = os.environ.get("MODE", default).strip().lower()
    if mode not in {"scratch", "resnet50"}:
        raise ValueError(f"Invalid MODE={mode!r}. Use MODE=scratch or MODE=resnet50.")
    return mode


def _unnormalize_imagenet(x: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    # x kommer in normaliserad med ImageNets mean/std. Plockar tillbaka till [0,1] så bilderna
    # ser ut som bilder igen när jag visar dem.
    m = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    return (x * s) + m


def _imagenet_mean_std(weights: ResNet50_Weights) -> tuple[list[float], list[float]]:
    # Olika torchvision-versioner lägger mean/std på olika ställen — säkraste är att
    # gräva fram dem ur Normalize-transformen i weights.transforms().
    t = weights.transforms()
    for tr in getattr(t, "transforms", []):
        if isinstance(tr, transforms.Normalize):
            return list(tr.mean), list(tr.std)
    # Fallback om jag inte hittar dem — standardvärden för ImageNet.
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class FilterAndRemap(Dataset):
    """
    Liten wrapper som kurerar ett dataset:
    - behåller bara samples vars label finns i allowed_labels
    - mappar om labels till 0..K-1 så att klassindexen blir kontinuerliga
    """

    def __init__(self, base: Dataset, allowed_labels: list[int], label_names: list[str]):
        self.base = base
        self.allowed_labels = allowed_labels
        self.label_names = label_names
        self.label_to_new = {lab: i for i, lab in enumerate(allowed_labels)}
        self.indices: list[int] = []
        # Jag itererar INTE base[i] här — det skulle trigga transforms och göra startup
        # extremt långsam. Plockar råa labels från `targets` om det finns, annars
        # faller jag tillbaka på base[i] som fallback.
        targets = getattr(base, "targets", None)
        if targets is not None:
            for i, y in enumerate(targets):
                if int(y) in self.label_to_new:
                    self.indices.append(i)
        else:
            for i in range(len(base)):
                _, y = base[i]
                if int(y) in self.label_to_new:
                    self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        return x, self.label_to_new[int(y)]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 64x1x1 -> 64
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def eval_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
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


def make_curves_figure(history: dict[str, list[float]]):
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
    fig = make_curves_figure(history)
    fig.savefig(os.path.join(out_dir, "curves_loss_acc.png"), dpi=160)
    plt.close(fig)


def make_confusion_matrix_percent_figure(y_true: list[int], y_pred: list[int], class_names: list[str]):
    import matplotlib.pyplot as plt
    k = len(class_names)
    cm = [[0 for _ in range(k)] for _ in range(k)]  # [pred][true]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < k and 0 <= p < k:
            cm[p][t] += 1

    cm_pct = [[0.0 for _ in range(k)] for _ in range(k)]
    for true_label in range(k):
        col_sum = 0
        for pred_label in range(k):
            col_sum += cm[pred_label][true_label]
        if col_sum == 0:
            col_sum = 1
        for pred_label in range(k):
            cm_pct[pred_label][true_label] = (cm[pred_label][true_label] / col_sum) * 100.0

    fig = plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm_pct, cmap="Blues", interpolation="nearest")
    plt.title("Confusion matrix (% of TRUE class) on TEST")
    plt.xlabel("True label (x)")
    plt.ylabel("Predicted label (y)")
    plt.xticks(range(k), class_names, rotation=25, ha="right")
    plt.yticks(range(k), class_names)
    plt.colorbar()
    for i in range(k):
        for j in range(k):
            plt.text(j, i, f"{cm_pct[i][j]:.1f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def save_confusion_matrix_percent(y_true: list[int], y_pred: list[int], class_names: list[str], out_dir: str) -> None:
    import matplotlib.pyplot as plt
    fig = make_confusion_matrix_percent_figure(y_true, y_pred, class_names)
    fig.savefig(os.path.join(out_dir, "confusion_matrix_percent.png"), dpi=160)
    plt.close(fig)


def _pick_examples(y_true: list[int], y_pred: list[int], want_correct: bool, max_examples: int) -> list[int]:
    picked: list[int] = []
    for i in range(len(y_true)):
        is_correct = (y_true[i] == y_pred[i])
        if is_correct == want_correct:
            picked.append(i)
        if len(picked) >= max_examples:
            break
    return picked


def make_examples_figure(
    xs: list[torch.Tensor],
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    want_correct: bool,
    max_examples: int = 12,
):
    import matplotlib.pyplot as plt
    picked = _pick_examples(y_true, y_pred, want_correct=want_correct, max_examples=max_examples)
    if len(picked) == 0:
        return None

    cols = 4
    rows = (len(picked) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for k, idx in enumerate(picked, start=1):
        img = xs[idx].permute(1, 2, 0).numpy()
        img = img.clip(0.0, 1.0)
        plt.subplot(rows, cols, k)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"pred={class_names[y_pred[idx]]}\ntrue={class_names[y_true[idx]]}")
    plt.tight_layout()
    return fig


def save_examples(
    xs: list[torch.Tensor],
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_dir: str,
    filename: str,
    want_correct: bool,
    max_examples: int = 12,
) -> None:
    import matplotlib.pyplot as plt
    fig = make_examples_figure(
        xs, y_true, y_pred, class_names, want_correct=want_correct, max_examples=max_examples
    )
    if fig is None:
        return
    fig.savefig(os.path.join(out_dir, filename), dpi=160)
    plt.close(fig)


def _device() -> torch.device:
    # Defaultar till CPU för stabilitet — MPS kan abort:a på vissa setups.
    # Slå på MPS med USE_MPS=1 om jag vill köra på GPU.
    if os.environ.get("USE_MPS", "").strip() == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)


def _make_run_dirs(run_name: str) -> tuple[str, str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Allt som hör till del 3 hamnar inne i assignment1/, så jag inte rörar resten av repot.
    out_dir = os.path.join("assignment1", "outputs_catsAndDogs", f"{run_name}__run_{run_id}")
    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(out_dir, exist_ok=True)
    return run_id, out_dir, tb_dir


def _curate_cifar10_catsdogs(train: bool, transform) -> FilterAndRemap:
    # CIFAR-10 har 10 klasser, jag plockar ut bara katt (3) och hund (5).
    # Officiell label-lista finns på https://www.cs.toronto.edu/~kriz/cifar.html.
    # Datasetet sparas under assignment1/data så det ligger på samma ställe som MNIST från del 2.
    base = datasets.CIFAR10(root="assignment1/data", train=train, download=True, transform=transform)
    return FilterAndRemap(base, allowed_labels=[3, 5], label_names=["cat", "dog"])


def _build_model(mode: str, num_classes: int, device: torch.device) -> nn.Module:
    if mode == "scratch":
        return SmallCNN(num_classes=num_classes).to(device)

    if mode == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        # Byter ut sista fc-lagret mot ett som ger ut 2 klasser (cat/dog) i stället för 1000.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)

    raise ValueError(f"Unknown mode={mode!r}. Use 'scratch' or 'resnet50'.")


def _freeze_backbone_for_transfer(model: nn.Module) -> None:
    # Frys allt utom fc-lagret. Det är så jag faktiskt utnyttjar features som ResNet
    # redan lärt sig på ImageNet — annars skulle jag bara träna om hela nätet.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():  # type: ignore[attr-defined]
        p.requires_grad = True


def _count_trainable_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def main():
    mode = _get_mode()
    config = TrainConfig(
        mode=mode,
        lr=5e-4 if mode == "resnet50" else 1e-3,
        epochs=3 if mode == "resnet50" else 5,
        batch_size=32 if mode == "resnet50" else 64,
        max_train_pool=2500 if mode == "resnet50" else None,
    )
    device = _device()
    _seed_everything(config.seed)

    run_name = f"catsAndDogs__{config.mode}__img{48 if config.mode=='resnet50' else config.img_size}"
    if config.max_train_pool is not None:
        run_name += f"__pool{config.max_train_pool}"
    run_id, out_dir, tb_dir = _make_run_dirs(run_name=run_name)
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"run_id={run_id} out_dir={out_dir} device={device} mode={config.mode}")
    writer.add_text("00_Info/device", str(device), 0)
    writer.add_text("00_Info/training_config", json.dumps(asdict(config), indent=2, sort_keys=True), 0)

    with open(os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)

    class_names = ["cat", "dog"]
    num_classes = len(class_names)

    if config.mode == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        mean, std = _imagenet_mean_std(weights)
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(48),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        # Mindre eval-size än ImageNet-default — ResNet klarar varierande spatiala dims och det blir mycket snabbare på laptop.
        eval_transform = transforms.Compose(
            [
                transforms.Resize(56),
                transforms.CenterCrop(48),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        writer.add_text("00_Info/model", "ResNet50 (förtränad på ImageNet) med nytt fc-huvud — bara fc tränas hos mig", 0)
        writer.add_text("00_Info/image_size", "48x48 — väldigt snabb transfer learning på laptop", 0)
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(config.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
                transforms.ToTensor(),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(config.img_size),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
            ]
        )
        writer.add_text("00_Info/model", "SmallCNN — egen liten CNN tränad från grunden, ingen förträning", 0)
        writer.add_text("00_Info/image_size", f"{config.img_size}x{config.img_size}", 0)

    # Mitt kurerade dataset — bara katter och hundar från CIFAR-10
    print("Loading curated CIFAR-10 cats vs dogs...")
    full_train = _curate_cifar10_catsdogs(train=True, transform=train_transform)
    full_eval = _curate_cifar10_catsdogs(train=True, transform=eval_transform)  # samma bilder, men ren eval-transform
    test_set = _curate_cifar10_catsdogs(train=False, transform=eval_transform)
    print(f"Sizes: train_pool={len(full_eval)} test={len(test_set)}")

    # Krymper poolen om jag bara vill köra snabbt — användbart när jag testar saker på min laptop.
    if config.max_train_pool is not None and len(full_eval) > config.max_train_pool:
        g = torch.Generator().manual_seed(config.seed)
        perm = torch.randperm(len(full_eval), generator=g).tolist()
        keep = perm[: config.max_train_pool]
        full_train = Subset(full_train, keep)
        full_eval = Subset(full_eval, keep)
        print(f"Using subset for speed: train_pool={len(full_eval)}")

    # Splitten görs på full_eval (ren transform, deterministisk), och sen återanvänder
    # jag samma index på det augmenterade trainsetet — så jag är säker på att val-setet
    # aldrig ser augmentation och inte överlappar med train.
    n = len(full_eval)
    n_val = int(n * config.val_frac)
    n_train = n - n_val
    train_part, val_part = random_split(
        full_eval,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_set = Subset(full_train, train_part.indices)
    val_set = Subset(full_eval, val_part.indices)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    model = _build_model(config.mode, num_classes=num_classes, device=device)
    if config.mode == "resnet50":
        _freeze_backbone_for_transfer(model)

    writer.add_text("00_Info/trainable_params", str(_count_trainable_params(model)), 0)
    print(f"trainable_params={_count_trainable_params(model)}")

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        if epoch == 1:
            print(f"Starting training loop... batches_per_epoch={len(train_loader)}")
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
        # matplotlib är importerad lokalt i hjälpfunktionen, så jag stänger via fig API:t
        fig.clf()

        ckpt_epoch_path = os.path.join(out_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "history": history, "config": asdict(config)},
            ckpt_epoch_path,
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "history": history, "config": asdict(config)},
                os.path.join(out_dir, "best.pt"),
            )
            writer.add_scalar("00_KeyNumbers/bästa val-loss (lägre är bättre)", best_val_loss, epoch)
            writer.add_scalar("00_KeyNumbers/bästa epok (epoch index)", float(best_epoch), epoch)

    total_train_seconds = time.time() - start_time
    writer.add_scalar("00_KeyNumbers/total träningstid (sekunder)", total_train_seconds, 0)
    save_curves(history, out_dir)

    best = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
    print(f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    writer.add_scalar("00_KeyNumbers/test loss (lägre är bättre)", test_loss, 0)
    writer.add_scalar("00_KeyNumbers/test accuracy (högre är bättre)", test_acc, 0)

    # Samla prediktioner för confusion matrix + exempelbilder
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
            # Spara bilderna så jag kan visa dem.
            # I resnet50-läget normaliserar jag med ImageNet-mean/std, då måste jag plocka tillbaka
            # bilderna till [0,1] innan plot, annars ser de helt konstiga ut.
            x_cpu = x.cpu()
            if config.mode == "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V2
                mean, std = _imagenet_mean_std(weights)
                for img in x_cpu:
                    xs.append(_unnormalize_imagenet(img, mean=mean, std=std).clamp(0.0, 1.0))
            else:
                for img in x_cpu:
                    xs.append(img.clamp(0.0, 1.0))

    save_confusion_matrix_percent(y_true, y_pred, class_names, out_dir)
    save_examples(xs, y_true, y_pred, class_names, out_dir, "examples_correct.png", want_correct=True, max_examples=12)
    save_examples(xs, y_true, y_pred, class_names, out_dir, "examples_incorrect.png", want_correct=False, max_examples=12)

    cm_fig = make_confusion_matrix_percent_figure(y_true, y_pred, class_names)
    writer.add_figure("05_Figurer/confusion matrix (% av TRUE klass)", cm_fig, global_step=0)
    cm_fig.clf()

    correct_fig = make_examples_figure(xs, y_true, y_pred, class_names, want_correct=True, max_examples=12)
    if correct_fig is not None:
        writer.add_figure("05_Figurer/exempel – korrekta prediktioner", correct_fig, global_step=0)
        correct_fig.clf()

    incorrect_fig = make_examples_figure(xs, y_true, y_pred, class_names, want_correct=False, max_examples=12)
    if incorrect_fig is not None:
        writer.add_figure("05_Figurer/exempel – felklassificeringar", incorrect_fig, global_step=0)
        incorrect_fig.clf()

    writer.close()


if __name__ == "__main__":
    main()

