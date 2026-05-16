"""EmotionROI Dataset 单元测试 (使用临时 fake 数据)。"""

import os
import sys
import tempfile
import unittest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import EmotionROIDataset


class TestEmotionROIDataset(unittest.TestCase):

    CLASSES = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    def setUp(self):
        """创建临时 fake 数据集目录结构 (匹配真实 images/ 子目录)。"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = self.temp_dir.name

        # 创建 images/ 下的类别子目录
        images_dir = os.path.join(self.data_root, "images")
        for cls_name in self.CLASSES:
            os.makedirs(os.path.join(images_dir, cls_name), exist_ok=True)

        # 创建 split 目录
        split_dir = os.path.join(self.data_root, "training_testing_split")
        os.makedirs(split_dir, exist_ok=True)

        # 生成 fake 图片和 split 文件
        self._create_split("training.txt", num_per_class=5)
        self._create_split("testing.txt", num_per_class=3)

    def _create_split(self, filename, num_per_class):
        """创建 split 文件并生成对应 fake 图片 (在 images/ 下)。"""
        split_path = os.path.join(self.data_root, "training_testing_split", filename)
        with open(split_path, "w") as f:
            for cls_name in self.CLASSES:
                for i in range(num_per_class):
                    img_name = f"{cls_name}_{i:04d}.jpg"
                    img_path = os.path.join(self.data_root, "images", cls_name, img_name)
                    img = Image.new("RGB", (224, 224), color=(i * 40, 100, 200))
                    img.save(img_path)
                    f.write(f"{cls_name}/{img_name}\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_length_training(self):
        dataset = EmotionROIDataset(
            data_root=self.data_root,
            split_file="training_testing_split/training.txt",
        )
        self.assertEqual(len(dataset), 6 * 5)

    def test_dataset_length_testing(self):
        dataset = EmotionROIDataset(
            data_root=self.data_root,
            split_file="training_testing_split/testing.txt",
        )
        self.assertEqual(len(dataset), 6 * 3)

    def test_labels_range(self):
        dataset = EmotionROIDataset(
            data_root=self.data_root,
            split_file="training_testing_split/testing.txt",
        )
        labels = [dataset[i][1] for i in range(len(dataset))]
        self.assertEqual(set(labels), set(range(6)))

    def test_image_tensor_shape(self):
        try:
            from torchvision import transforms
        except ImportError:
            self.skipTest("torchvision 未安装")
        transform = transforms.ToTensor()
        dataset = EmotionROIDataset(
            data_root=self.data_root,
            split_file="training_testing_split/training.txt",
            transform=transform,
        )
        img, label = dataset[0]
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertIsInstance(label, int)

    def test_num_classes_property(self):
        dataset = EmotionROIDataset(
            data_root=self.data_root,
            split_file="training_testing_split/training.txt",
        )
        self.assertEqual(dataset.num_classes, 6)

    def test_invalid_split_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            EmotionROIDataset(
                data_root=self.data_root,
                split_file="nonexistent.txt",
            )

    def test_class_order_matches_constant(self):
        dataset = EmotionROIDataset(
            data_root=self.data_root,
            split_file="training_testing_split/training.txt",
        )
        self.assertEqual(dataset.CLASSES, self.CLASSES)

    def test_label_assignment(self):
        """验证每个类别的 label 赋值正确。"""
        dataset = EmotionROIDataset(
            data_root=self.data_root,
            split_file="training_testing_split/training.txt",
        )
        for idx in range(len(dataset)):
            img_path, label = dataset.samples[idx]
            img_path = img_path.replace("\\", "/")
            # 从路径中提取类别名
            class_name = img_path.split("/")[-2]
            expected_label = self.CLASSES.index(class_name)
            self.assertEqual(label, expected_label)


if __name__ == "__main__":
    unittest.main()
