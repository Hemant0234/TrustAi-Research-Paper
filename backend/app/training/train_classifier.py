import os
import argparse
from typing import List

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# ============================================================
# NIH CHEST-XRAY14
# ============================================================

CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

NUM_CLASSES = len(CLASSES)


class NIHChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 multi-label dataset.

    Each image can have multiple findings.
    """

    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        transform=None,
        max_samples=None,
    ):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_file)

        # Keep only images that actually exist in this split.
        image_names = set(
            os.listdir(image_dir)
        )

        df = df[df["Image Index"].isin(image_names)].copy()

        if max_samples is not None:
            df = df.head(max_samples)

        self.df = df.reset_index(drop=True)

        print(
            f"[INFO] Dataset: {image_dir}"
        )
        print(
            f"[INFO] Samples found: {len(self.df)}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_name = row["Image Index"]
        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        # Convert grayscale X-ray to RGB.
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Multi-label target.
        labels = row["Finding Labels"].split("|")

        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        for label in labels:
            if label in CLASSES:
                target[CLASSES.index(label)] = 1.0

        return image, target


# ============================================================
# MODEL
# ============================================================

def build_model():

    print("[INFO] Loading DenseNet121...")

    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, NUM_CLASSES)
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
):

    model.train()

    running_loss = 0.0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

        if scaler is not None:

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()
            optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total += batch_size

        if batch_idx % 50 == 0:

            print(
                f"    Batch {batch_idx}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / max(total, 1)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total += batch_size

        if batch_idx % 50 == 0:

            print(
                f"    Validation Batch {batch_idx}/{len(loader)} "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    return running_loss / max(total, 1)



# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_pipeline(
    data_dir,
    epochs,
    batch_size,
    learning_rate,
    output_dir,
    max_train_samples=None,
    max_val_samples=None,
):

    print("=" * 70)
    print("       TRUSTXAI-MED NIH CHEST-XRAY14 TRAINING")
    print("=" * 70)

    print(f"Dataset:       NIH ChestX-ray14")
    print(f"Architecture:  DenseNet121")
    print(f"Epochs:        {epochs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}")

    if device.type == "cuda":

        print(
            f"[INFO] GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    csv_file = os.path.join(
        data_dir,
        "Data_Entry_2017_v2020.csv"
    )

    train_dir = os.path.join(
        data_dir,
        "train",
        "images"
    )

    val_dir = os.path.join(
        data_dir,
        "val",
        "images"
    )

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"CSV not found: {csv_file}"
        )

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training images not found: {train_dir}"
        )

    if not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Validation images not found: {val_dir}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("\n[INFO] Loading training dataset...")

    train_dataset = NIHChestXrayDataset(
        csv_file=csv_file,
        image_dir=train_dir,
        transform=train_transform,
        max_samples=max_train_samples,
    )

    print("\n[INFO] Loading validation dataset...")

    val_dataset = NIHChestXrayDataset(
        csv_file=csv_file,
        image_dir=val_dir,
        transform=val_transform,
        max_samples=max_val_samples,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset contains 0 images."
        )

    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation dataset contains 0 images."
        )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    num_workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"\n[INFO] Training batches: {len(train_loader)}"
    )

    print(
        f"[INFO] Validation batches: {len(val_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 3),
        eta_min=1e-6,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = None

    if device.type == "cuda":

        scaler = torch.amp.GradScaler("cuda")

        print(
            "[INFO] Mixed precision training enabled."
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    checkpoint_path = os.path.join(
        output_dir,
        "densenet121_nih_chestxray14_best.pth"
    )

    for epoch in range(epochs):

        print("\n" + "=" * 70)

        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        print("=" * 70)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        print(
            f"\nEpoch {epoch + 1} complete:"
        )

        print(
            f"  Train Loss: {train_loss:.4f}"
        )

        print(
            f"  Val Loss:   {val_loss:.4f}"
        )

        print(
            f"  LR:         {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "classes": CLASSES,
                },
                checkpoint_path,
            )

            print(
                f"[INFO] New best model saved:"
            )

            print(
                f"       {checkpoint_path}"
            )

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"[INFO] Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"[INFO] Checkpoint: {checkpoint_path}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/chexpert"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None
    )

    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
