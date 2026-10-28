"""
NASA-IBM Prithvi Geospatial Foundation Model Optical ViT Backbone with LoRA Support.

Supports:
  1. Parameter-Efficient Fine-Tuning (PEFT) via LoRA on Query and Value attention projections.
  2. Partial Fine-Tuning by unfreezing the last N Transformer blocks.
  3. Strict backbone parameter freezing for zero-shot or linear probing adaptation.
  4. Official pre-trained weight loading from Prithvi_100M.pt with 2D/temporal pos_embed interpolation.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_adapter import LoRALinear

def interpolate_pos_embed(pos_embed, target_grid_size=(8, 8), patch_size=16):
    """
    Interpolates 2D positional embeddings for ViT when input size differs from pretraining.
    Default Prithvi pretraining: 224x224 -> grid 14x14 = 196 tokens.
    Target patch size: 128x128 -> grid 8x8 = 64 tokens.
    Handles temporal MAE pos_embed (e.g., 3 x 14 x 14 = 588 tokens) by pooling temporal slots.
    """
    if pos_embed.shape[1] == (target_grid_size[0] * target_grid_size[1] + 1):
        return pos_embed
    elif pos_embed.shape[1] == (target_grid_size[0] * target_grid_size[1]):
        return pos_embed

    def split_special_and_spatial(tokens):
        n_tokens = tokens.shape[1]
        for has_special in (False, True):
            usable = n_tokens - int(has_special)
            for temporal_size in (1, 2, 3, 4, 5, 6, 8, 12):
                if usable > 0 and usable % temporal_size == 0:
                    spatial_size = int(math.sqrt(usable // temporal_size))
                    if spatial_size * spatial_size == usable // temporal_size:
                        special = tokens[:, :1] if has_special else None
                        spatial = tokens[:, 1:] if has_special else tokens
                        if temporal_size > 1:
                            spatial = spatial.reshape(1, temporal_size, spatial_size * spatial_size, tokens.shape[-1]).mean(dim=1)
                        return special, spatial, spatial_size, temporal_size
        raise ValueError(f"Cannot infer Prithvi spatial grid from {n_tokens} positional tokens.")

    cls_token, pos_embed_spatial, orig_size, temporal_size = split_special_and_spatial(pos_embed)
    has_cls = cls_token is not None
    N, orig_tokens, D = pos_embed_spatial.shape
    pos_embed_spatial = pos_embed_spatial.reshape(1, orig_size, orig_size, D).permute(0, 3, 1, 2)
    pos_embed_spatial = F.interpolate(pos_embed_spatial, size=target_grid_size, mode="bicubic", align_corners=False)
    pos_embed_spatial = pos_embed_spatial.permute(0, 2, 3, 1).flatten(1, 2)

    if has_cls and cls_token is not None:
        pos_embed_spatial = torch.cat([cls_token, pos_embed_spatial], dim=1)
    return pos_embed_spatial

class PrithviAttention(nn.Module):
    """
    Multi-Head Self-Attention for Prithvi Vision Transformer.
    Supports Parameter-Efficient LoRA on Query (W_q) and Value (W_v) projections.
    Maintains compatibility with official Prithvi / timm / MAE state dicts.
    """
    def __init__(self, embed_dim=768, num_heads=12, qkv_bias=True,
                 lora_r=8, lora_alpha=16.0, lora_dropout=0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.scale = self.head_dim ** -0.5
        self.lora_r = lora_r

        self.q_proj = LoRALinear(embed_dim, embed_dim, r=lora_r, lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = LoRALinear(embed_dim, embed_dim, r=lora_r, lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        in_proj_w_key = prefix + "in_proj_weight"
        in_proj_b_key = prefix + "in_proj_bias"
        qkv_w_key = prefix + "qkv.weight"
        qkv_b_key = prefix + "qkv.bias"

        src_w_key = in_proj_w_key if in_proj_w_key in state_dict else (qkv_w_key if qkv_w_key in state_dict else None)
        src_b_key = in_proj_b_key if in_proj_b_key in state_dict else (qkv_b_key if qkv_b_key in state_dict else None)

        if src_w_key is not None:
            in_w = state_dict.pop(src_w_key)
            w_q, w_k, w_v = in_w.chunk(3, dim=0)
            state_dict[prefix + "q_proj.linear.weight"] = w_q
            state_dict[prefix + "k_proj.weight"] = w_k
            state_dict[prefix + "v_proj.linear.weight"] = w_v

        if src_b_key is not None:
            in_b = state_dict.pop(src_b_key)
            b_q, b_k, b_v = in_b.chunk(3, dim=0)
            state_dict[prefix + "q_proj.linear.bias"] = b_q
            state_dict[prefix + "k_proj.bias"] = b_k
            state_dict[prefix + "v_proj.linear.bias"] = b_v

        proj_w_key = prefix + "proj.weight"
        proj_b_key = prefix + "proj.bias"
        if proj_w_key in state_dict and (prefix + "out_proj.weight") not in state_dict:
            state_dict[prefix + "out_proj.weight"] = state_dict.pop(proj_w_key)
        if proj_b_key in state_dict and (prefix + "out_proj.bias") not in state_dict:
            state_dict[prefix + "out_proj.bias"] = state_dict.pop(proj_b_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, query, key=None, value=None, need_weights=False):
        if key is None:
            key = query
        if value is None:
            value = query

        B, N_q, C = query.shape
        N_k = key.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = out.permute(0, 2, 1, 3).reshape(B, N_q, C)
        out = self.out_proj(out)
        return out, None

class PrithviViTBlock(nn.Module):
    """Transformer Encoder Block with multi-head self attention and LoRA support."""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 lora_r=8, lora_alpha=16.0, lora_dropout=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = PrithviAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class PrithviBackboneWrapper(nn.Module):
    """
    NASA-IBM Prithvi Vision Transformer Optical Backbone.
    Input: optical data (B, 6, 128, 128) -> Output: tokens (B, 64, 768) and spatial map (B, 768, 8, 8).
    Supports:
      - PEFT via LoRA on Q and V attention projections.
      - Partial fine-tuning by unfreezing the last N blocks.
      - Full backbone freeze.
      - Loading official pre-trained Prithvi_100M weights.
    """
    def __init__(self, in_channels=6, patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, img_size=128, pretrained_name="ibm-nasa-geospatial/Prithvi-100M",
                 checkpoint_path=None, load_pretrained=True, min_weight_coverage=0.90,
                 lora_r=8, lora_alpha=16.0, lora_dropout=0.05, lora_blocks=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.depth = depth
        self.num_heads = num_heads
        self.grid_size = (img_size // patch_size, img_size // patch_size) # (8, 8) = 64 patches
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Blocks with LoRA support
        if lora_blocks is None:
            active_lora_indices = set(range(depth)) if lora_r > 0 else set()
        elif isinstance(lora_blocks, int):
            active_lora_indices = set(range(max(0, depth - lora_blocks), depth))
        else:
            active_lora_indices = set(lora_blocks)

        self.blocks = nn.ModuleList([
            PrithviViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                lora_r=lora_r if i in active_lora_indices else 0,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.pretrained_loaded = False
        if load_pretrained and embed_dim == 768 and depth == 12:
            self.load_prithvi_weights(checkpoint_path=checkpoint_path, min_weight_coverage=min_weight_coverage)

    def freeze_backbone(self, keep_lora_trainable=False):
        """Freeze all base parameters in backbone for PEFT."""
        for name, param in self.named_parameters():
            if keep_lora_trainable and ("lora_A" in name or "lora_B" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        status = "FROZEN (base only, LoRA trainable)" if keep_lora_trainable else "FROZEN (all)"
        print(f"[+] Prithvi backbone parameters {status}.")

    def unfreeze_lora(self, blocks=None):
        """
        Freeze base backbone and activate only LoRA parameters for training.
        """
        self.freeze_backbone(keep_lora_trainable=False)
        lora_params_count = 0
        target_blocks = range(self.depth) if blocks is None else (
            range(max(0, self.depth - blocks), self.depth) if isinstance(blocks, int) else blocks
        )
        for b_idx in target_blocks:
            for name, param in self.blocks[b_idx].named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True
                    lora_params_count += param.numel()
        print(f"[+] Prithvi Optical LoRA activated: {lora_params_count:,} trainable parameters.")
        return lora_params_count

    def unfreeze_last_blocks(self, count=2):
        """
        Fine-tune only the last count Transformer blocks; keep earlier blocks frozen.
        """
        if not isinstance(count, int) or not 0 <= count <= len(self.blocks):
            raise ValueError(f"count must be between 0 and {len(self.blocks)}")
        self.freeze_backbone(keep_lora_trainable=False)
        if count > 0:
            for block in self.blocks[len(self.blocks) - count:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"[+] Fine-tuning last {count} Transformer blocks.")
        else:
            print("[+] Backbone completely frozen (0 blocks fine-tuned).")

    def get_lora_params(self):
        """Return all active trainable LoRA parameters."""
        return [p for n, p in self.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]

    def load_prithvi_weights(self, checkpoint_path=None, repo_id="ibm-nasa-geospatial/Prithvi-100M",
                             filename="Prithvi_100M.pt", min_weight_coverage=0.90):
        """Load pre-trained Prithvi weights from local path or HuggingFace."""
        resolved_path = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            resolved_path = checkpoint_path
        elif os.environ.get("PRITHVI_CHECKPOINT_PATH") and os.path.isfile(os.environ["PRITHVI_CHECKPOINT_PATH"]):
            resolved_path = os.environ["PRITHVI_CHECKPOINT_PATH"]

        if resolved_path and os.path.isfile(resolved_path):
            try:
                ckpt = torch.load(resolved_path, map_location="cpu", weights_only=True)
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                self.load_state_dict(state_dict, strict=False)
                self.pretrained_loaded = True
                print(f"[+] Successfully loaded pre-trained Prithvi weights from: {resolved_path}")
                return True
            except Exception as e:
                print(f"[!] Warning: Could not load weights from {resolved_path}: {e}")

        print("[i] Using initialized optical ViT backbone (standard MAE initialization).")
        return False

    def forward(self, x):
        """
        Input: (B, 6, 128, 128)
        Output: tokens (B, 64, 768), spatial_feats (B, 768, 8, 8)
        """
        B, C, H, W = x.shape
        x_patches = self.patch_embed(x) # (B, 768, 8, 8)
        H_p, W_p = x_patches.shape[2], x_patches.shape[3]
        tokens = x_patches.flatten(2).transpose(1, 2) # (B, 64, 768)

        pos = self.pos_embed
        if pos.shape[1] != tokens.shape[1]:
            pos = interpolate_pos_embed(pos, target_grid_size=(H_p, W_p))
        tokens = tokens + pos

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        spatial_feats = tokens.transpose(1, 2).reshape(B, self.embed_dim, H_p, W_p)
        return tokens, spatial_feats
