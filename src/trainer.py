"""训练循环。管理训练、评估、checkpoint 保存和日志输出。"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from .dataset import EmotionROIDataset
from .models import build_model
from .metrics import compute_metrics, plot_confusion_matrix
from .utils import (
    make_run_dir, save_config, save_class_to_idx, save_git_info, Logger,
)

# CLIP 官方归一化参数
CLIP_MEAN = (0.48145466, 0.4826274, 0.39049073)
CLIP_STD = (0.26862955, 0.26128729, 0.2779887)


def get_transforms(image_size=224):
    """构建 CLIP 标准 train / test transforms。"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    return train_transform, test_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """单 epoch 训练。返回 (train_loss, train_acc)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total * 100
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """单 epoch 评估，返回 loss 和全部预测/标签。"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_labels, all_preds


def train(config):
    """完整训练流程。"""
    # ---- 0. 环境准备 ----
    device = torch.device(
        config["train"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(config["experiment"]["seed"])

    run_dir = make_run_dir(config["experiment"]["name"])
    logger = Logger(os.path.join(run_dir, "train.log"))

    logger.log(f"Run directory: {run_dir}")
    logger.log(f"Device: {device}")

    # 保存项目元信息
    save_config(config, run_dir)
    save_class_to_idx(config["data"]["classes"], run_dir)
    save_git_info(run_dir)

    # ---- 1. 数据 ----
    train_transform, test_transform = get_transforms(config["data"]["image_size"])
    data_cfg = config["data"]

    train_dataset = EmotionROIDataset(
        data_root=data_cfg["data_root"],
        split_file=os.path.join(data_cfg["split_dir"], data_cfg["train_file"]),
        transform=train_transform,
    )
    test_dataset = EmotionROIDataset(
        data_root=data_cfg["data_root"],
        split_file=os.path.join(data_cfg["split_dir"], data_cfg["test_file"]),
        transform=test_transform,
    )

    logger.log(f"Train samples: {len(train_dataset)}")
    logger.log(f"Test  samples: {len(test_dataset)}")

    train_cfg = config["train"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- 2. 模型 ----
    model = build_model(config)
    model = model.to(device)
    logger.log(f"Model head: {config['model']['head']}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Trainable / Total params: {trainable_params:,} / {total_params:,}")

    # ---- 3. 优化器与损失 ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # ---- 4. 训练循环 ----
    best_acc = 0.0
    metrics_records = []
    all_labels = []
    all_preds = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, all_labels, all_preds = evaluate(model, test_loader, criterion, device)

        metrics, conf_matrix = compute_metrics(all_labels, all_preds, config["data"]["classes"])
        test_acc = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        # 日志输出 (控制台 + 文件)
        best_mark = " (new best)" if is_best else ""
        log_line = (
            f"[Epoch {epoch:03d}/{train_cfg['epochs']:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.2f} "
            f"macro_f1={macro_f1:.2f} "
            f"best_acc={best_acc:.2f}{best_mark}"
        )
        logger.log(log_line)

        # 记录指标
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 2),
            "macro_f1": round(macro_f1, 2),
            "best_acc": round(best_acc, 2),
        }
        for cls_name in config["data"]["classes"]:
            record[f"recall_{cls_name}"] = round(metrics[f"recall_{cls_name}"], 2)
        metrics_records.append(record)

        # 保存 checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "test_acc": test_acc,
            "macro_f1": macro_f1,
        }

        torch.save(checkpoint, os.path.join(run_dir, "last.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(run_dir, "best_acc.pth"))

    # ---- 5. 训练结束收尾 ----
    # 保存 metrics.csv
    df = pd.DataFrame(metrics_records)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # 保存最终混淆矩阵 (PNG + NPY)
    plot_confusion_matrix(
        conf_matrix, config["data"]["classes"],
        os.path.join(run_dir, "confusion_matrix.png"),
    )
    import numpy as np
    np.save(os.path.join(run_dir, "confusion_matrix.npy"), conf_matrix)

    logger.log(f"\nTraining finished. Best test_acc: {best_acc:.2f}")
    logger.close()

    return run_dir, best_acc
