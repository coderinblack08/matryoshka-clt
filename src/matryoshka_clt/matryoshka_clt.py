from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import nn

# Use vendored circuit_tracer CLT implementation
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder


@dataclass
class MatryoshkaCLTConfig:
    n_layers: int
    d_model: int
    d_transcoder: int
    prefixes: List[int]  # e.g., [256, 512, 1024]
    activation_function: str = "relu"  # or "jump_relu"
    lr: float = 4e-4
    steps: int = 10_000
    batch_size: int = 4096
    l1_coef: float = 1.4e-4
    weight_decay: float = 0.0
    warmup_steps: int = 5_000
    b_dec_init: str = "mean"  # "mean" or "zero"
    device: str = "cpu"
    dtype: str = "float32"  # or "bfloat16"


def _dtype_from_str(s: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }[s]


class MatryoshkaCLTTrainer:
    def __init__(self, cfg: MatryoshkaCLTConfig):
        self.cfg = cfg
        dtype = _dtype_from_str(cfg.dtype)
        self.model = CrossLayerTranscoder(
            n_layers=cfg.n_layers,
            d_transcoder=cfg.d_transcoder,
            d_model=cfg.d_model,
            activation_function=cfg.activation_function,
            lazy_decoder=False,
            lazy_encoder=False,
            device=torch.device(cfg.device),
            dtype=dtype,
        )
        # Kaiming init for enc/dec
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.model.W_enc, a=0.0)
            for i in range(cfg.n_layers):
                nn.init.kaiming_uniform_(self.model.W_dec[i], a=0.0)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # Constant with warmup schedule
        def lr_lambda(step: int):
            if cfg.warmup_steps <= 0:
                return 1.0
            return min(1.0, (step + 1) / float(cfg.warmup_steps))
        self._global_step = 0
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    def _reconstruct_with_prefix(self, feats: torch.Tensor, p: int) -> torch.Tensor:
        """Decode using only features < p (by index) across all layers.

        feats: [n_layers, batch, d_transcoder]
        returns recon: [n_layers, batch, d_model]
        """
        masked = feats
        if p < feats.shape[-1]:
            masked = feats.clone()
            masked[..., p:] = 0
        # Convert to sparse per-layer for decoder path
        sparse_layers = []
        for l in range(masked.shape[0]):
            layer_feats = masked[l]  # [batch, d_transcoder]
            sparse_layers.append(layer_feats.to_sparse())
        sparse_feats = torch.stack(sparse_layers).coalesce()
        return self.model.decode(sparse_feats)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        # x: [n_layers, batch, d_model], y: [n_layers, batch, d_model]
        feats = self.model.encode(x)
        # Baseline reconstruction (no features): just decoder bias
        recon0 = self.model.b_dec[:, None, :]
        l2_terms = [ (recon0 - y).pow(2).mean() ]
        # Prefix reconstructions
        for p in self.cfg.prefixes:
            recon_p = self._reconstruct_with_prefix(feats, p)
            l2_terms.append( (recon_p - y).pow(2).mean() )
        l2_stack = torch.stack(l2_terms)
        mean_l2 = l2_stack.mean()
        # L1 on encoded activations (proxy for acts_topk in SAE code)
        l1_norm = feats.abs().sum(dim=-1).mean()
        l1_loss = self.cfg.l1_coef * l1_norm
        loss = mean_l2 + l1_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self._global_step += 1
        self.sched.step()
        return {
            "loss": float(loss.detach().cpu()),
            "l2_mean": float(mean_l2.detach().cpu()),
            "l2_min": float(l2_stack.min().detach().cpu()),
            "l2_max": float(l2_stack.max().detach().cpu()),
            "l1_norm": float(l1_norm.detach().cpu()),
        }


def train_matryoshka_clt(
    cfg: MatryoshkaCLTConfig,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[CrossLayerTranscoder, dict]:
    device = torch.device(cfg.device)
    dtype = _dtype_from_str(cfg.dtype)
    trainer = MatryoshkaCLTTrainer(cfg)
    trainer.model.to(device=device, dtype=dtype)

    X = torch.tensor(x, dtype=dtype, device=device)
    Y = torch.tensor(y, dtype=dtype, device=device)
    assert X.shape == Y.shape == (cfg.n_layers, X.shape[1], cfg.d_model), "x/y must be [n_layers, batch, d_model]"

    # Initialize decoder bias to mean of outputs, as in some SAE configs
    if cfg.b_dec_init == "mean":
        with torch.no_grad():
            # Mean over positions
            mean_by_layer = Y.mean(dim=1)  # [n_layers, d_model]
            trainer.model.b_dec.copy_(mean_by_layer)

    ds = torch.utils.data.TensorDataset(X.transpose(0,1), Y.transpose(0,1))  # [batch, n_layers, d_model]
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    logs = {}
    for step in range(cfg.steps):
        for bx, by in dl:
            # transpose back to [n_layers, batch, d_model]
            bx = bx.transpose(0,1).contiguous()
            by = by.transpose(0,1).contiguous()
            logs = trainer.step(bx, by)
        # simple last-iter logging only
    return trainer.model, logs


def save_clt_safetensors(model: CrossLayerTranscoder, out_dir: str) -> None:
    """Save encoder/decoder weights into per-layer safetensors like circuit-tracer expects."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for l in range(model.n_layers):
            save_file({f"W_enc_{l}": model.W_enc[l].cpu()}, os.path.join(out_dir, f"W_enc_{l}.safetensors"))
            save_file({f"W_dec_{l}": model.W_dec[l].cpu()}, os.path.join(out_dir, f"W_dec_{l}.safetensors"))
