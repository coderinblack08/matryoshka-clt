from __future__ import annotations

import argparse
import json
import os
import numpy as np

from .matryoshka_clt import MatryoshkaCLTConfig, train_matryoshka_clt, save_clt_safetensors


def main() -> None:
    ap = argparse.ArgumentParser(prog="matryoshka-clt")
    ap.add_argument("--data", required=True, help="NPZ file with arrays 'x' and 'y' shaped [n_layers, n_pos, d_model]")
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--features", type=int, help="Total CLT features (d_transcoder). If omitted, computed as expansion * d_model.")
    ap.add_argument("--expansion", type=int, default=32, help="Expansion factor to compute features if --features not set (default: 32)")
    ap.add_argument("--prefixes", nargs="+", type=int, required=True, help="Matryoshka feature prefixes (e.g. 256 512 1024)")
    ap.add_argument("--activation", default="relu", choices=["relu", "jump_relu"]) 
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--l1", type=float, default=1.4e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=5000)
    ap.add_argument("--b-dec-init", default="mean", choices=["mean", "zero"], help="Initialize decoder bias to mean or zeros (default: mean)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--save-safetensors", action="store_true", help="Export W_enc_*.safetensors and W_dec_*.safetensors")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    npz = np.load(args.data)
    if not ("x" in npz and "y" in npz):
        raise SystemExit("NPZ must contain arrays 'x' and 'y'")
    x = npz["x"]
    y = npz["y"]

    features = args.features if args.features is not None else args.expansion * args.d_model

    cfg = MatryoshkaCLTConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_transcoder=features,
        prefixes=list(sorted(args.prefixes)),
        activation_function=args.activation,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        l1_coef=args.l1,
        weight_decay=args.wd,
        warmup_steps=args.warmup_steps,
        b_dec_init=args.b_dec_init,
        device=args.device,
        dtype=args.dtype,
    )

    model, logs = train_matryoshka_clt(cfg, x, y)
    with open(os.path.join(args.out, "training_logs.json"), "w") as f:
        json.dump(logs, f)

    if args.save_safetensors:
        save_clt_safetensors(model, args.out)
    else:
        import torch
        torch.save(model.state_dict(), os.path.join(args.out, "clt.pt"))

    print(f"Done. Saved to {args.out}")


if __name__ == "__main__":
    main()
