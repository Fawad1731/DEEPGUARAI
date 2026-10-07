"""
DeepGuard AI — Training Script
================================
Supports the '140k Real and Fake Faces' Kaggle dataset.

Expected dataset folder structure (place inside project root):
  real-vs-fake/
    train/
      real/
      fake/
    valid/
      real/
      fake/
    test/
      real/
      fake/

Run:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import get_model

# ─────────────────────────────────────────────
#  CONFIGURATION  —  Edit paths if needed
# ─────────────────────────────────────────────
# Point this to wherever Kaggle extracted the dataset
# Common locations:
#   ./real-vs-fake          (extracted in project folder)
#   C:/Users/<you>/Downloads/real-vs-fake
DATASET_ROOT = "./real_vs_fake/real-vs-fake"   # <-- change if different

BATCH_SIZE    = 32        # Reduce to 16 if you get OOM errors
EPOCHS        = 5         # Fast Mode for quick submission
LEARNING_RATE = 1e-4
NUM_WORKERS   = 4         # Set 0 if you see DataLoader errors on Windows
IMAGE_SIZE    = 224
SAVE_PATH     = "best_model.pth"
TARGET_ACC    = 0.90
# ─────────────────────────────────────────────


def get_transforms():
    """
    Training: aggressive augmentation to prevent overfitting.
    Validation/Test: only resize + normalize (no augmentation).
    """
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # Slightly larger
        transforms.RandomCrop(IMAGE_SIZE),                       # Random crop to 224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Occlusion robustness
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf


def load_datasets(dataset_root, train_tf, val_tf):
    """
    Load train / valid / test splits using ImageFolder.
    ImageFolder automatically assigns:
      fake/ -> label 0
      real/ -> label 1
    We'll override the criterion in training to match our convention
    (fake=1, real=0) by flipping labels below.
    """
    train_dir = os.path.join(dataset_root, "train")
    valid_dir = os.path.join(dataset_root, "valid")
    test_dir  = os.path.join(dataset_root, "test")

    # Verify folders exist
    for split, d in [("train", train_dir), ("valid", valid_dir)]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"[ERROR] '{split}' folder not found at: {d}\n"
                f"Make sure DATASET_ROOT='{dataset_root}' is correct.\n"
                f"It should contain train/, valid/, test/ subdirectories."
            )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    valid_ds = datasets.ImageFolder(valid_dir, transform=val_tf)

    print(f"[Dataset] Train : {len(train_ds):,} images  |  Classes: {train_ds.classes}")
    print(f"[Dataset] Valid : {len(valid_ds):,} images")

    test_ds = None
    if os.path.isdir(test_dir):
        test_ds = datasets.ImageFolder(test_dir, transform=val_tf)
        print(f"[Dataset] Test  : {len(test_ds):,} images")

    # ImageFolder maps alphabetically: fake=0, real=1
    # We want: real=0 (negative), fake=1 (positive) for BCEWithLogitsLoss
    # So we need to check class_to_idx and handle accordingly
    class_to_idx = train_ds.class_to_idx
    print(f"[Dataset] Class mapping: {class_to_idx}")

    fake_label = class_to_idx.get('fake', 0)  # Will be 0 with ImageFolder
    real_label = class_to_idx.get('real', 1)  # Will be 1 with ImageFolder

    # --- FAST MODE (GHOST MODE) ---
    import random
    train_idx = random.sample(range(len(train_ds)), min(2000, len(train_ds)))
    valid_idx = random.sample(range(len(valid_ds)), min(500, len(valid_ds)))
    train_ds = torch.utils.data.Subset(train_ds, train_idx)
    valid_ds = torch.utils.data.Subset(valid_ds, valid_idx)
    print(f"\n[FAST MODE] Training on {len(train_ds)} images, validating on {len(valid_ds)} images.\n")
    # ------------------------------

    return train_ds, valid_ds, test_ds, fake_label, real_label


def get_label_tensor(labels_batch, fake_label_idx):
    """
    Convert ImageFolder labels (fake=0, real=1) to our convention (fake=1, real=0).
    This ensures BCEWithLogitsLoss treats fake images as positives.
    """
    # If ImageFolder assigned fake=0, flip: new_label = 1 - original
    # If ImageFolder assigned fake=1, keep: new_label = original
    if fake_label_idx == 0:
        return (1 - labels_batch).float().unsqueeze(1)
    else:
        return labels_batch.float().unsqueeze(1)


def train_one_epoch(model, loader, optimizer, criterion, device, fake_label_idx, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        targets = get_label_tensor(labels, fake_label_idx).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, fake_label_idx, split_name="Val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc=f"  [{split_name}]", leave=False):
        inputs = inputs.to(device)
        targets = get_label_tensor(labels, fake_label_idx).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train():
    print("=" * 60)
    print("  DeepGuard AI — Training Pipeline")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" + (" 🚀 GPU detected!" if device.type == 'cuda' else " (CPU — training will be slow)"))

    # Transforms & Datasets
    train_tf, val_tf = get_transforms()

    try:
        train_ds, valid_ds, test_ds, fake_label_idx, _ = load_datasets(DATASET_ROOT, train_tf, val_tf)
    except FileNotFoundError as e:
        print(e)
        return

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda')
    )

    # Model
    model = get_model(device, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    # Cosine annealing gives smooth LR decay → better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"\n[Config] Batch={BATCH_SIZE} | Epochs={EPOCHS} | LR={LEARNING_RATE} | Target={TARGET_ACC*100:.0f}%")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, fake_label_idx, epoch, EPOCHS
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, fake_label_idx, "Valid")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch:02d}/{EPOCHS}  LR={current_lr:.2e}")
        print(f"  Train  →  Loss: {train_loss:.4f}  |  Acc: {train_acc*100:.2f}%")
        print(f"  Valid  →  Loss: {val_loss:.4f}  |  Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✅ New best model saved! ({val_acc*100:.2f}%)")

        if val_acc >= TARGET_ACC:
            print(f"\n🎯 Target accuracy of {TARGET_ACC*100:.0f}% reached! Training complete.")
            break

        print("-" * 60)

    print(f"\n{'='*60}")
    print(f"Training finished. Best validation accuracy: {best_val_acc*100:.2f}%")

    # Final test evaluation
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        # Load best model for test
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, fake_label_idx, "Test ")
        print(f"Test  →  Loss: {test_loss:.4f}  |  Acc: {test_acc*100:.2f}%")

    print(f"Model saved to: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
