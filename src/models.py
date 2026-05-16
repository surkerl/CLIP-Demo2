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

    def forward(self, x, return_evidence_maps=False):
        with torch.no_grad():
            outputs = self.vision_encoder(x)
            features = outputs.pooler_output  # (B, hidden_dim)
        logits = self.classifier(features)
        if return_evidence_maps:
            return logits, None, None, None
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

    def forward(self, x, return_evidence_maps=False):
        with torch.no_grad():
            outputs = self.vision_encoder(x)
            features = outputs.pooler_output
        logits = self.classifier(features)
        if return_evidence_maps:
            return logits, None, None, None
        return logits


class CAEMBaseline(nn.Module):
    """Frozen CLIP ViT-B/16 + Class-aware Affective Evidence Mining.

    - num_classes 个 learnable class queries
    - 对每个类别用 class query 与所有 patch token 计算 attention
    - 加权求和得到 evidence token，经 class-specific weight 得到 evidence_logits
    - cls_token 经 global classifier 得到 global_logits
    - final_logits = global_logits + evidence_scale * evidence_logits
    """

    def __init__(self, clip_model_name="openai/clip-vit-base-patch16",
                 num_classes=6, evidence_dim=768, dropout=0.3,
                 evidence_scale=1.0):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        hidden_dim = self.vision_encoder.config.hidden_size  # ViT-B/16 => 768
        self.num_classes = num_classes
        self.evidence_dim = evidence_dim
        self.evidence_scale = evidence_scale
        self.scale = evidence_dim ** 0.5

        # 可学习 class queries [C, D]
        self.class_queries = nn.Parameter(torch.empty(num_classes, evidence_dim))
        nn.init.xavier_uniform_(self.class_queries)

        # Class-specific evidence weight [C, D] + bias [C]
        self.evidence_weight = nn.Parameter(torch.empty(num_classes, evidence_dim))
        nn.init.xavier_uniform_(self.evidence_weight)
        self.evidence_bias = nn.Parameter(torch.zeros(num_classes))

        # Global classifier (从 cls_token 得到 global_logits)
        self.global_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values, return_evidence_maps=False):
        outputs = self.vision_encoder(
            pixel_values=pixel_values, output_hidden_states=False,
        )
        tokens = outputs.last_hidden_state      # [B, 197, 768]
        cls_token = tokens[:, 0, :]              # [B, 768]
        patch_tokens = tokens[:, 1:, :]          # [B, 196, 768]

        B, N, D = patch_tokens.shape
        C = self.num_classes

        # Class-aware attention: [B, C, 196]
        attention = torch.einsum("cd,bnd->bcn", self.class_queries, patch_tokens)
        attention = attention / self.scale
        evidence_maps = attention.softmax(dim=-1)

        # Evidence tokens: weighted sum → [B, C, D]
        evidence_tokens = torch.einsum("bcn,bnd->bcd", evidence_maps, patch_tokens)

        # Evidence logits: 每个类用自己的 weight
        evidence_logits = torch.einsum("bcd,cd->bc", evidence_tokens, self.evidence_weight)
        evidence_logits = evidence_logits + self.evidence_bias

        # Global logits (residual)
        global_logits = self.global_classifier(cls_token)

        # Final = global + evidence_scale * evidence
        final_logits = global_logits + self.evidence_scale * evidence_logits

        if return_evidence_maps:
            return final_logits, evidence_maps, global_logits, evidence_logits
        return final_logits


def build_model(config):
    """根据配置构建模型。支持 head: linear / mlp / caem。"""
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
    elif head == "caem":
        model = CAEMBaseline(
            clip_model_name=model_cfg["clip_model"],
            num_classes=num_classes,
            evidence_dim=model_cfg.get("evidence_dim", 768),
            dropout=model_cfg.get("dropout", 0.3),
            evidence_scale=model_cfg.get("evidence_scale", 1.0),
        )
    else:
        raise ValueError(f"未知的分类头类型: {head}，可选值为 linear / mlp / caem")

    return model
