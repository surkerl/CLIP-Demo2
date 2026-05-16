"""EmotionROI 数据集。使用官方 training_testing_split 划分。"""

import os
from PIL import Image
from torch.utils.data import Dataset


class EmotionROIDataset(Dataset):
    """EmotionROI 数据集。

    目录结构要求:
        data_root/
        ├── images/
        │   ├── anger/
        │   ├── disgust/
        │   ├── fear/
        │   ├── joy/
        │   ├── sadness/
        │   └── surprise/
        └── training_testing_split/
            ├── training.txt
            └── testing.txt

    split 文件每行格式: class_name/image_filename.ext
    实际图片路径: data_root/images/class_name/image_filename.ext
    """

    CLASSES = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    def __init__(self, data_root, split_file, transform=None):
        """
        Args:
            data_root: 数据集根目录
            split_file: 相对于 data_root 的 split 文件路径
            transform: torchvision transforms
        """
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        split_path = os.path.join(data_root, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"split 文件不存在: {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 统一路径分隔符，支持 Windows / Linux
                line = line.replace("\\", "/")
                parts = line.split("/")
                if len(parts) < 2:
                    continue
                class_name = parts[-2]
                if class_name in self.CLASSES:
                    label = self.CLASSES.index(class_name)
                    img_path = os.path.join(data_root, "images", line)
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"split 文件中未找到有效样本: {split_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self):
        return len(self.CLASSES)
