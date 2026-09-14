"""Decoder-only Transformer language model -- the complete Model H / Model L network.

The brief calls for "a decoder-only (GPT-style) Transformer language model"
(SS2.1). Named TransformerLM rather than GPT to keep it unambiguous that every
layer here is built from primitive PyTorch modules -- nothing is imported from
HuggingFace and no pretrained weights are involved, per guideline 2.

    tokens (B, T)
      -> TokenPositionEmbedding      (B, T, d)
      -> N x TransformerBlock        (B, T, d)
      -> final LayerNorm             (B, T, d)
      -> LM head                     (B, T, vocab_size)   logits for next token
      -> cross-entropy against tokens shifted left by one

Weight tying. The LM head is ``nn.Linear(d_model, vocab_size, bias=False)`` whose
weight is the *same tensor* as the token-embedding matrix. Both map between token
identity and model space, so sharing them is a reasonable inductive bias and
saves ``vocab_size x d_model`` = 8000 x 512 = 4,096,000 parameters, about 14% of
this model. The tied matrix receives gradient from both uses.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import TokenPositionEmbedding
from src.model.transformer_block import TransformerBlock


# Target value excluded from the cross-entropy. Matches torch's default so
# pretraining behaviour is untouched; finetuning writes it over prompt spans.
IGNORE_INDEX = -100


@dataclass
class TransformerLMConfig:
    """Architecture hyperparameters. Mirrors the ``architecture:`` block of the YAML."""
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    ctx_len: int = 512
    dropout: float = 0.1
    tie_weights: bool = True
    pos_encoding: str = "learned"     # "learned" | "none" (bonus ablation)

    @classmethod
    def from_yaml(cls, path: str) -> "TransformerLMConfig":
        """Build a config from a model YAML file's ``architecture`` block."""
        import yaml
        blob = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        arch = blob["architecture"]
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in arch.items() if k in fields})

    def expected_params(self) -> int:
        """Closed-form trainable-parameter count, independent of the built model.

        Used as an assertion target so a wiring mistake cannot pass silently.
        """
        d, v, L, f = self.d_model, self.vocab_size, self.n_layers, self.d_ff
        emb = v * d + (self.ctx_len * d if self.pos_encoding == "learned" else 0)
        attn = 4 * (d * d + d)                      # Q,K,V,O projections + biases
        ffn = (d * f + f) + (f * d + d)             # two linears + biases
        lns = 2 * 2 * d                             # two LayerNorms (weight+bias)
        head = 0 if self.tie_weights else v * d
        return emb + L * (attn + ffn + lns) + 2 * d + head


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        config: A :class:`TransformerLMConfig`.
    """

    def __init__(self, config: TransformerLMConfig):
        super().__init__()
        self.config = config

        self.embed = TokenPositionEmbedding(
            config.vocab_size, config.d_model, config.ctx_len,
            config.dropout, learned_pos=(config.pos_encoding == "learned"))

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff,
                             config.ctx_len, config.dropout)
            for _ in range(config.n_layers)])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            # Shared storage, not a copy: both names point at one tensor.
            self.lm_head.weight = self.embed.token_emb.weight

        self.apply(self._init_weights)
        # GPT-2 scaled init: residual-path output projections are scaled down by
        # 1/sqrt(2*n_layers) so the variance of the residual stream does not grow
        # with depth.
        scale = 1.0 / math.sqrt(2 * config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module):
        """Normal(0, 0.02) for Linear/Embedding weights; zero biases."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                need_weights: bool = False):
        """Run the forward pass and optionally the causal-LM loss.

        Args:
            idx: Input token ids, (B, T).
            targets: Next-token ids, (B, T). The dataset already shifts these by
                one, so the loss is a plain element-wise cross-entropy.
            need_weights: Collect per-layer attention probabilities.

        Returns:
            (logits, loss, attentions):
                logits     (B, T, vocab_size)
                loss       scalar, or None when targets is None
                attentions list of n_layers tensors (B, h, T, T), or None
        """
        x = self.embed(idx)

        attentions = [] if need_weights else None
        for block in self.blocks:
            x, w = block(x, need_weights=need_weights)
            if need_weights:
                attentions.append(w)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # (B, T, V) -> (B*T, V) and (B, T) -> (B*T,).
            #
            # ignore_index is passed explicitly even though -100 is already
            # cross_entropy's default. Phase 3 finetuning masks every prompt
            # position to IGNORE_INDEX so the loss is taken on the answer span
            # alone; a contract that load-bearing should be visible in the code
            # rather than inherited silently from a library default. Pretraining
            # passes no -100 targets, so its loss is unchanged.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=IGNORE_INDEX)
        return logits, loss, attentions

    def num_params(self, trainable_only: bool = True) -> int:
        """Total parameter count. Tied weights are counted once."""
        seen, total = set(), 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def param_breakdown(self) -> dict:
        """Per-component parameter counts, for the report's architecture table."""
        emb_tok = self.embed.token_emb.weight.numel()
        emb_pos = self.embed.pos_emb.weight.numel() if self.embed.pos_emb else 0
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        return {
            "token_embedding": emb_tok,
            "position_embedding": emb_pos,
            "per_block": per_block,
            "blocks_total": per_block * len(self.blocks),
            "final_layernorm": sum(p.numel() for p in self.ln_f.parameters()),
            "lm_head": 0 if self.config.tie_weights else self.lm_head.weight.numel(),
            "total": self.num_params(),
            "expected": self.config.expected_params(),
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, greedy: bool = False,
                 eos_id: int = None) -> torch.Tensor:
        """Autoregressively extend a prompt.

        Args:
            idx: Prompt token ids, (B, T).
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature; ignored when greedy.
            greedy: Take the argmax instead of sampling.
            eos_id: Stop early once every sequence has emitted this id.

        Returns:
            (B, T + max_new_tokens) token ids, prompt included.
        """
        self.eval()
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # Learned absolute positions exist only up to ctx_len, so crop.
            window = idx[:, -self.config.ctx_len:]
            logits, _, _ = self(window)
            logits = logits[:, -1, :]                       # (B, V), last position
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None:
                done |= nxt.squeeze(1) == eos_id
                if bool(done.all()):
                    break
        return idx
