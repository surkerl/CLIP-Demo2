"""配置加载模块。读取 YAML 配置文件，支持命令行参数覆盖。"""

import argparse
import yaml


def load_config():
    parser = argparse.ArgumentParser(description="CEDC-CLIP Training")
    parser.add_argument(
        "--config", type=str, default="configs/emotionroi_baseline.yaml",
        help="YAML 配置文件路径"
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="覆盖数据集根路径")
    parser.add_argument("--head", type=str, default=None,
                        choices=["linear", "mlp"],
                        help="覆盖分类头类型")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="覆盖 batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="覆盖实验名称")
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖设备 (cuda / cpu)")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖 YAML 配置
    if args.data_root is not None:
        config["data"]["data_root"] = args.data_root
    if args.head is not None:
        config["model"]["head"] = args.head
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["learning_rate"] = args.lr
    if args.experiment_name is not None:
        config["experiment"]["name"] = args.experiment_name
    if args.device is not None:
        config["train"]["device"] = args.device

    return config
