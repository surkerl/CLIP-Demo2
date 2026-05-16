"""Baseline 模型: frozen CLIP ViT-B/16 + Linear / MLP 分类头。"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class LinearBaseline(nn.Module):
    """冻结 CLIP ViT-B/16 视觉编码器 + 单层 Linear 分类头。"""

    def __init__(self, clip_model_name="openai/clip-vit-base-patch16",
                 num_classes=6, dropout=0.3):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)

        # 冻结视觉编码器全部参数
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        hidden_dim = self.vision_encoder.config.hidden_size  # ViT-B/16 => 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            outputs = self.vision_encoder(x)
            features = outputs.pooler_output  # (B, hidden_dim)
        logits = self.classifier(features)
        return logits


class MLPBaseline(nn.Module):
    """冻结 CLIP ViT-B/16 视觉编码器 + 两层 MLP 分类头。"""

    def __init__(self, clip_model_name="openai/clip-vit-base-patch16",
                 num_classes=6, mlp_hidden=512, dropout=0.3):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)

        # 冻结视觉编码器全部参数
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        hidden_dim = self.vision_encoder.config.hidden_size  # ViT-B/16 => 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            outputs = self.vision_encoder(x)
            features = outputs.pooler_output
        logits = self.classifier(features)
        return logits


def build_model(config):
    """根据配置构建模型。"""
    model_cfg = config["model"]
    num_classes = len(config["data"]["classes"])

    head = model_cfg.get("head", "linear")
    if head == "linear":
        model = LinearBaseline(
            clip_model_name=model_cfg["clip_model"],
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.3),
        )
    elif head == "mlp":
        model = MLPBaseline(
            clip_model_name=model_cfg["clip_model"],
            num_classes=num_classes,
            mlp_hidden=model_cfg.get("mlp_hidden", 512),
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"未知的分类头类型: {head}，可选值为 linear / mlp")

    return model
