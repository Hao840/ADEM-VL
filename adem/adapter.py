from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from clip.model import ResidualAttentionBlock
import adem


class Adapter(nn.Module):
    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            drop_ratio=0
    ):
        super().__init__()
        if hidden_dim > 0:
            self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
            self.fc2 = nn.Linear(hidden_dim, in_features, bias=False)
            self.hidden_dim = hidden_dim
            nn.init.zeros_(self.fc2.weight)
        self.dropout = nn.Dropout(0.1)
        self.drop_ratio = drop_ratio

    def forward(self, x, vis_weight):
        with autocast():
            if vis_weight is not None:
                image_embeds, adapter_emb1, adapter_emb2 = vis_weight
                x = (F.silu(x)) @ (F.silu(image_embeds + adapter_emb1).permute(0, 2, 1))
                if self.drop_ratio > 0:
                    score, _ = x.sort(dim=2)
                    threshold = score[:, :, int(self.drop_ratio * score.size(2))].unsqueeze(2)
                    mask = torch.ones_like(x)
                    mask[torch.where(x < threshold.expand_as(x))] = 0
                    x *= mask
                x = x @ (image_embeds + adapter_emb2)
            else:
                x = self.fc1(x)
                x = self.dropout(F.gelu(x))
                x = self.fc2(x)
        return x


def checkpoint(func, enable, training, *args, **kwargs):
    if enable and training:
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
    else:
        return func(*args, **kwargs)


def forward_llama(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                  vis_weight):
    x_norm = self.attention_norm(x)
    h = x + checkpoint(self.attention, self.gradient_checkpointing, self.training, x_norm, start_pos, freqs_cis, mask)
    h_norm = self.ffn_norm(h)
    out = h + checkpoint(self.feed_forward, self.gradient_checkpointing, self.training, h_norm) + self.adapter_mlp(
        h_norm, vis_weight) * self.s
    return out


def forward_clip(self, x: torch.Tensor):
    x = x + self.attention(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x)) + self.adapter_mlp(self.ln_2(x), None) * self.s
    return x


def set_Llama_Adapter(model, s=1, gradient_checkpointing=False, drop_ratio=0):
    for _ in model.children():
        if type(_) == adem.model.TransformerBlock:
            _.adapter_mlp = Adapter(_.dim, hidden_dim=0, drop_ratio=drop_ratio)
            _.s = s
            _.gradient_checkpointing = gradient_checkpointing
            bound_method = forward_llama.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Llama_Adapter(_, s, gradient_checkpointing=gradient_checkpointing, drop_ratio=drop_ratio)


def set_Clip_Adapter(model, dim=8, s=0.1):
    for _ in model.children():
        if type(_) == ResidualAttentionBlock:
            _.adapter_mlp = Adapter(1024, hidden_dim=dim)
            _.s = s
            bound_method = forward_clip.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Clip_Adapter(_, dim, s)
