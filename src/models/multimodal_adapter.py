"""
Parameter-Efficient Fine-Tuning (PEFT) and Multimodal Adaptation Modules.

Includes:
  1. LoRALinear: Low-Rank Adaptation (Hu et al., 2021) for ViT self-attention projections.
  2. SARTXTEncoderCNN: Compact auxiliary CNN encoder for SAR, GLCM, Indices, and DEM.
  3. GatedCrossModalFusion: Gated cross-modal bottleneck with zero-initialization identity residual.
  4. LocalAuxiliaryEncoder & LocalResidualFusion: High-resolution residual fusion alternative.
  5. PrithviMultimodalAdapterModel: End-to-end multimodal architecture with optical warm-start.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import DenseRegressionDecoder

FUSION_TYPES = ('feature_v1', 'local_residual_v2')

# -------------------------------------------------------------------------
# 1. LoRA (Low-Rank Adaptation) Layer
# -------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    Parameter-Efficient LoRA layer for linear projections in ViT attention.
    Delta W = (B * A) * (alpha / r), where r << d.
    At initialization, B is initialized to zeros, ensuring the adapter produces
    exactly zero delta and perfectly preserves pre-trained or frozen base representations.
    """
    def __init__(self, in_features, out_features, r=8, lora_alpha=16.0, lora_dropout=0.05, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / r) if r > 0 else 1.0
        
        # Frozen base linear projection
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # Trainable low-rank decomposition matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            self.dropout = nn.Identity()

    @classmethod
    def from_linear(cls, linear: nn.Linear, r=8, lora_alpha=16.0, lora_dropout=0.05):
        """Construct LoRALinear wrapping an existing nn.Linear, preserving its weights."""
        has_bias = linear.bias is not None
        lora = cls(linear.in_features, linear.out_features, r=r,
                   lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=has_bias)
        with torch.no_grad():
            lora.linear.weight.copy_(linear.weight)
            if has_bias:
                lora.linear.bias.copy_(linear.bias)
        lora.linear.weight.requires_grad = False
        if has_bias:
            lora.linear.bias.requires_grad = False
        return lora

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        w_key = prefix + "weight"
        b_key = prefix + "bias"
        if w_key in state_dict and (prefix + "linear.weight") not in state_dict:
            state_dict[prefix + "linear.weight"] = state_dict.pop(w_key)
        if b_key in state_dict and (prefix + "linear.bias") not in state_dict:
            state_dict[prefix + "linear.bias"] = state_dict.pop(b_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        base_out = self.linear(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_out + lora_out
        return base_out


# -------------------------------------------------------------------------
# 2. Auxiliary Multimodal CNN Encoders
# -------------------------------------------------------------------------
class SARTXTEncoderCNN(nn.Module):
    """Compact auxiliary encoder: four stride-2 stages, 128x128 -> 8x8."""
    def __init__(self, in_channels=35, embed_dim=64):
        super().__init__()
        widths = (16, 32, 64, embed_dim)
        stages = []
        current_in = in_channels
        for width in widths:
            stages.extend([
                nn.Conv2d(current_in, width, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, width),
                nn.GELU(),
            ])
            current_in = width
        self.stages = nn.Sequential(*stages)

    def forward(self, x_aux):
        return self.stages(x_aux)


class GatedCrossModalFusion(nn.Module):
    """Identity-initialized residual cross-modal fusion through a bottleneck."""
    def __init__(self, embed_dim=768, aux_dim=64, bottleneck_dim=64):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(embed_dim + aux_dim, bottleneck_dim, 1),
            nn.GroupNorm(1, bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
            nn.GELU(),
        )
        self.output_proj = nn.Conv2d(bottleneck_dim, embed_dim, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, optical_feats, aux_feats):
        correction = self.output_proj(
            self.residual(torch.cat([optical_feats, aux_feats], dim=1))
        )
        return optical_feats + correction


class LocalAuxiliaryEncoder(nn.Module):
    """Compact multi-scale CNN for full-resolution auxiliary spatial features."""
    def __init__(self, in_channels, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class LocalResidualFusion(nn.Module):
    """Local residual correction network with tanh-bounded scaling."""
    def __init__(self, optical_dim=768, aux_dim=32, out_channels=2,
                 target_scales=None, limit=2.0, raw_aux_channels=0):
        super().__init__()
        self.limit = float(limit)
        self.optical_proj = nn.Conv2d(optical_dim, aux_dim, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(aux_dim * 2, aux_dim, 1),
            nn.GELU(),
            nn.Conv2d(aux_dim, 1, 1)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(aux_dim * 2, aux_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(aux_dim, aux_dim, 3, padding=1),
            nn.GELU()
        )
        self.output_proj = nn.Conv2d(aux_dim, out_channels, 1)
        self.linear_proj = (nn.Conv2d(raw_aux_channels, out_channels, 1)
                            if raw_aux_channels > 0 else None)
        if target_scales is not None:
            scales = torch.as_tensor(target_scales, dtype=torch.float32).view(1, -1, 1, 1)
        else:
            scales = torch.ones(1, out_channels, 1, 1)
        self.register_buffer('target_scale', scales)

    def forward(self, optical_feats, aux_feats, raw_aux=None):
        optical = F.interpolate(self.optical_proj(optical_feats), size=aux_feats.shape[-2:],
                                mode='bilinear', align_corners=False)
        gate = torch.sigmoid(self.gate(torch.cat((optical, aux_feats), dim=1)))
        hidden = self.residual(torch.cat((optical, gate * aux_feats), dim=1))
        logits = 0.1 * self.output_proj(hidden)
        if self.linear_proj is not None and raw_aux is not None:
            logits = logits + self.linear_proj(raw_aux)
        return self.target_scale * self.limit * torch.tanh(logits / self.limit)


# -------------------------------------------------------------------------
# 3. End-to-End Multimodal Adapter Model
# -------------------------------------------------------------------------
class PrithviMultimodalAdapterModel(nn.Module):
    """
    Proposed Architecture:
      - Prithvi-100M ViT Backbone for Optical bands (6 HLS channels) with LoRA or Frozen mode
      - Auxiliary CNN Encoder for SAR, GLCM, Indices, and DEM
      - Gated Cross-Modal Fusion
      - Dense Regression Decoder for independent targets [AGB, GSV]
      - Deterministic Carbon Stock calculation: Carbon = 0.47 * AGB (IPCC standard)
    """
    def __init__(self, backbone, total_channels=41, optical_channels=6,
                 embed_dim=768, out_channels=2, use_adapter=True,
                 aux_dim=64, fusion_dim=64, target_names=None, fusion_type='feature_v1',
                 fusion_width=32, residual_scales=None, residual_limit=2.0):
        super().__init__()
        self.target_names = target_names or ["AGB", "GSV"]
        self.backbone = backbone
        self.total_channels = total_channels
        self.optical_channels = optical_channels
        self.aux_channels = total_channels - optical_channels
        self.use_adapter = use_adapter and (self.aux_channels > 0)
        self.out_channels = out_channels
        self.fusion_type = fusion_type

        if self.use_adapter and fusion_type == 'local_residual_v2':
            self.aux_encoder = LocalAuxiliaryEncoder(self.aux_channels, fusion_width)
            self.fusion = LocalResidualFusion(embed_dim, fusion_width, out_channels,
                                             residual_scales, residual_limit, self.aux_channels)
        elif self.use_adapter:
            self.aux_encoder = SARTXTEncoderCNN(
                in_channels=self.aux_channels,
                embed_dim=aux_dim
            )
            self.fusion = GatedCrossModalFusion(
                embed_dim=embed_dim, aux_dim=aux_dim, bottleneck_dim=fusion_dim
            )

        self.decoder = DenseRegressionDecoder(
            in_dim=embed_dim,
            out_channels=out_channels
        )

    def load_optical_checkpoint(self, checkpoint_path, device="cpu"):
        """Load pre-trained optical backbone and decoder weights."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
        state = {k.removeprefix("module."): v for k, v in state.items()}

        backbone_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        decoder_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}

        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)
        if decoder_state:
            self.decoder.load_state_dict(decoder_state, strict=False)
        print(f"[+] Loaded optical warm-start weights from: {checkpoint_path}")

    def forward(self, x, return_components=False):
        """
        Input x: (B, total_channels, 128, 128)
        Output: (B, out_channels, 128, 128)
        """
        x_optical = x[:, :self.optical_channels, :, :] # (B, 6, 128, 128)
        _, optical_feats = self.backbone(x_optical)    # (B, 768, 8, 8)

        if self.use_adapter and self.fusion_type == 'local_residual_v2':
            base = self.decoder(optical_feats)
            raw_aux = x[:, self.optical_channels:]
            correction = self.fusion(optical_feats, self.aux_encoder(raw_aux), raw_aux)
            prediction = torch.clamp_min(base + correction, 0.0)
            return (prediction, correction) if return_components else prediction

        if self.use_adapter:
            x_aux = x[:, self.optical_channels:, :, :]
            aux_feats = self.aux_encoder(x_aux)
            fused_feats = self.fusion(optical_feats, aux_feats)
        else:
            fused_feats = optical_feats

        pred = self.decoder(fused_feats)
        return (pred, None) if return_components else pred

    def predict_full_targets(self, x):
        """
        Predict independent targets and deterministically append Carbon stock (0.47 * AGB).
        Output shape: (B, 3, 128, 128) -> [AGB, GSV, Carbon].
        """
        pred = self.forward(x)
        if "AGB" in self.target_names:
            agb_idx = self.target_names.index("AGB")
            carbon = pred[:, agb_idx:agb_idx + 1] * 0.47
            return torch.cat([pred, carbon], dim=1)
        return pred

    def get_trainable_params_count(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
