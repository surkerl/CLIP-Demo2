"""CEDC-CLIP 训练入口。

用法:
    python main.py                                              # 使用默认配置
    python main.py --data-root /path/to/EmotionROI              # 覆盖数据集路径
    python main.py --head mlp --epochs 80                       # 覆盖模型头和 epoch
    python main.py --experiment-name mlp_v1                     # 覆盖实验名称
    python main.py --config configs/emotionroi_baseline.yaml    # 指定配置文件
"""

from src.config import load_config
from src.trainer import train


def main():
    config = load_config()
    run_dir, best_acc = train(config)
    print(f"\nRun directory: {run_dir}")
    print(f"Best test_acc: {best_acc:.2f}")


if __name__ == "__main__":
    main()
