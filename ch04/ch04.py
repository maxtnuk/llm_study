from typing import Optional
import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        dropout: float,
        attn_dtype: torch.dtype = torch.float32,
        qkv_bias=False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.attn_dtype = attn_dtype
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.inverse_d_k = np.reciprocal(np.sqrt(d_out))

        self.k_w = nn.Linear(d_in, d_out, dtype=self.attn_dtype, bias=qkv_bias)
        self.q_w = nn.Linear(d_in, d_out, dtype=self.attn_dtype, bias=qkv_bias)
        self.v_w = nn.Linear(d_in, d_out, dtype=self.attn_dtype, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out, dtype=self.attn_dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, context_len, _d_in = x.shape
        if mask is None:
            self.register_buffer(
                "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
            )

        weighted_q = self.k_w(x)
        weighted_k = self.q_w(x)
        weighted_v = self.v_w(x)

        # head view
        # [b, context_len, d_out] -> [b, num_heads, context_len, head_dim]
        keys = weighted_k.view(b, context_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        querys = weighted_q.view(
            b, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = weighted_v.view(
            b, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_scores = querys @ keys.transpose(2, 3)
        mask = (
            self.mask.bool().reshape(1, 1, context_len, context_len).tile((b, 1, 1, 1))
        )
        
        attention_scores.masked_fill_(mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores * self.inverse_d_k, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # [b, context_len, num_heads, head_dim]
        context_vectors = (attention_weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(b, context_len, self.d_out)
        context_vectors = self.out_proj(context_vectors)

        return context_vectors
