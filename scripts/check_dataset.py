"""检查 EmotionROI 数据集结构、样本数和类别分布。

用法:
    python scripts/check_dataset.py --data-root /path/to/EmotionROI
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.dataset import EmotionROIDataset


def check_split(data_root, split_file, class_names, label="Split"):
    """检查单个 split，只遍历 samples 列表，不打开图片。"""
    dataset = EmotionROIDataset(
        data_root=data_root,
        split_file=split_file,
        transform=None,
    )

    print(f"{label} samples: {len(dataset)}")

    # 直接遍历 dataset.samples，避免 __getitem__ 打开所有图片
    label_counts = Counter()
    missing = 0
    for img_path, lbl in dataset.samples:
        label_counts[lbl] += 1
        if not os.path.exists(img_path):
            missing += 1

    for idx, name in enumerate(class_names):
        count = label_counts.get(idx, 0)
        print(f"  {name:10s}: {count}")

    if missing > 0:
        print(f"  WARNING: {missing} 个图片文件不存在")
    else:
        print(f"  全部图片文件均存在")

    # 检查前 3 个样本
    print(f"  First 3 sample paths:")
    for i in range(min(3, len(dataset.samples))):
        img_path, lbl = dataset.samples[i]
        exists = os.path.exists(img_path)
        status = "OK" if exists else "MISSING"
        print(f"    [{i}] {img_path}  [{status}]")

    print()


def main():
    config = load_config()
    data_cfg = config["data"]

    print(f"Data root: {data_cfg['data_root']}")
    print(f"Classes:   {data_cfg['classes']}")
    print()

    # Train
    train_split = os.path.join(data_cfg["split_dir"], data_cfg["train_file"])
    check_split(data_cfg["data_root"], train_split, data_cfg["classes"], label="Train")

    # Test
    test_split = os.path.join(data_cfg["split_dir"], data_cfg["test_file"])
    check_split(data_cfg["data_root"], test_split, data_cfg["classes"], label="Test")


if __name__ == "__main__":
    main()
