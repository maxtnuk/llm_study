from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


@dataclass
class GPTConfig:
    vocab_size: int
    max_context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_ratio: float
    qkv_bias: bool


GPT_124M_CONFIG = GPTConfig(
    vocab_size=50257,
    max_context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_ratio=0.1,
    qkv_bias=False,
)


def create_gpt_config_from_dict(config_dict):
    config = GPTConfig(
        vocab_size=50257,  # default value
        max_context_length=1024,  # default value
        emb_dim=768,  # default value
        n_heads=12,  # default value
        n_layers=12,  # default value
        drop_ratio=0.1,  # default value
        qkv_bias=False,  # default value
    )

    if "vocab_size" in config_dict:
        config.vocab_size = config_dict["vocab_size"]
    if "context_length" in config_dict:
        config.max_context_length = config_dict["context_length"]
    if "n_embd" in config_dict:
        config.emb_dim = config_dict["n_embd"]
    if "n_head" in config_dict:
        config.n_heads = config_dict["n_head"]
    if "n_layer" in config_dict:
        config.n_layers = config_dict["n_layer"]
    if "dropout" in config_dict:
        config.drop_ratio = config_dict["dropout"]
    if "qkv_bias" in config_dict:
        config.qkv_bias = config_dict["qkv_bias"]

    return config


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        # trainable prameter
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor):
        # unbiased true -> correction=1 -> Bessel's correction(https://en.wikipedia.org/wiki/Bessel%27s_correction)
        # unbiased false -> correction=0
        (var, mean) = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
        norm = (x - mean) * torch.reciprocal(torch.sqrt(var + self.eps))
        return norm * self.scale + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        intermediate_dim = cfg.emb_dim * 4
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, intermediate_dim),
            GELU(),
            nn.Linear(intermediate_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


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
        model_device = x.device
        if mask is None:
            self.register_buffer(
                "mask",
                torch.triu(
                    torch.ones(context_len, context_len, device=model_device),
                    diagonal=1,
                ),
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


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            num_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
            dropout=cfg.drop_ratio,
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_ratio)

    def forward(self, x: torch.Tensor):
        # attention
        short_cut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + short_cut

        # feedforward
        short_cut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + short_cut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.max_context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_ratio)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        _batch_size, sequence = x.shape
        tok_embbed = self.tok_emb(x)
        in_consecutive_idx = torch.arange(sequence, device=x.device)
        pos_embbed = self.pos_emb(in_consecutive_idx)
        x_embbed = tok_embbed * pos_embbed

        x_drop_out = self.drop_emb(x_embbed)
        x = self.trf_blocks(x_drop_out)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}


def print_sampled_tokens(probas):
    torch.manual_seed(123)  # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(
        0
    )  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if (
            idx_next == eos_id
        ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].attn.q_w.weight = assign(
            gpt.trf_blocks[b].attn.q_w.weight, q_w.T
        )
        gpt.trf_blocks[b].attn.k_w.weight = assign(
            gpt.trf_blocks[b].attn.k_w.weight, k_w.T
        )
        gpt.trf_blocks[b].attn.v_w.weight = assign(
            gpt.trf_blocks[b].attn.v_w.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].attn.q_w.bias = assign(gpt.trf_blocks[b].attn.q_w.bias, q_b)
        gpt.trf_blocks[b].attn.k_w.bias = assign(gpt.trf_blocks[b].attn.k_w.bias, k_b)
        gpt.trf_blocks[b].attn.v_w.bias = assign(gpt.trf_blocks[b].attn.v_w.bias, v_b)

        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
