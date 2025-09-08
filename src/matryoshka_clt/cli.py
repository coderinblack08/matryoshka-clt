from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
from typing import Iterable
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

from .matryoshka_clt import (
    MatryoshkaCLTConfig,
    train_matryoshka_clt,
    save_clt_safetensors,
    MatryoshkaCLTTrainer,
)


def main() -> None:
    ap = argparse.ArgumentParser(prog="matryoshka-clt")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data", help="NPZ with arrays x,y shaped [n_layers, n_pos, d_model]")
    mode.add_argument("--dataset", help="HF dataset name for streaming training (e.g., Skylion007/openwebtext)")

    # Common hyperparameters
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--features", type=int, default=None, help="Total CLT features; if omitted, uses expansion * d_model")
    ap.add_argument("--expansion", type=int, default=32)
    ap.add_argument("--prefixes", nargs="+", type=int, default=None, help="Matryoshka prefixes; default = [F/8,F/4,F/2,F]")
    ap.add_argument("--activation", default="relu", choices=["relu", "jump_relu"]) 
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--l1", type=float, default=1.4e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=5000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32","fp32","bfloat16","bf16","float16","fp16"]) 
    ap.add_argument("--out", default="out")
    ap.add_argument("--save-safetensors", action="store_true")

    # Logging and publishing
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb-project", default="matryoshka-clt", help="W&B project name")
    ap.add_argument("--wandb-entity", default=None, help="W&B entity/org (optional)")
    ap.add_argument("--hf-repo-id", default=None, help="Hugging Face repo to upload artifacts (e.g., user/repo)")
    ap.add_argument("--hf-branch", default="main", help="Target branch for upload")
    ap.add_argument("--hf-subdir", default=None, help="Optional subdirectory within the repo to place artifacts")

    # Streaming-only options
    ap.add_argument("--model", default="gpt2-small", help="TransformerLens model name (streaming mode)")
    ap.add_argument("--split", default="train", help="Dataset split (streaming mode)")
    ap.add_argument("--seq-len", type=int, default=128, help="Sequence length (streaming mode)")
    ap.add_argument("--batch-tokens", type=int, default=4096, help="Tokens per batch (streaming mode)")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Optional W&B init (lazy import)
    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        except Exception as e:
            print(f"[wandb] Failed to initialize logging: {e}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }[args.dtype]

    features = args.features or (args.expansion * args.d_model)
    prefixes = args.prefixes or [features // 8, features // 4, features // 2, features]

    if args.data:
        # Offline mode
        npz = np.load(args.data)
        if not ("x" in npz and "y" in npz):
            raise SystemExit("NPZ must contain arrays 'x' and 'y'")
        x = npz["x"]
        y = npz["y"]
        cfg = MatryoshkaCLTConfig(
            n_layers=args.n_layers,
            d_model=args.d_model,
            d_transcoder=features,
            prefixes=sorted(prefixes),
            activation_function=args.activation,
            lr=args.lr,
            steps=args.steps,
            batch_size=4096,  # ignored: full-batch from NPZ via DataLoader in train_matryoshka_clt
            l1_coef=args.l1,
            weight_decay=args.wd,
            warmup_steps=args.warmup_steps,
            b_dec_init="mean",
            device=str(device),
            dtype=args.dtype,
        )
        model, logs = train_matryoshka_clt(cfg, x, y)
        with open(os.path.join(args.out, "training_logs.json"), "w") as f:
            json.dump(logs, f)
        if args.save_safetensors:
            save_clt_safetensors(model, args.out)
        else:
            torch.save(model.state_dict(), os.path.join(args.out, "clt.pt"))
        # Save basic config for publishing
        with open(os.path.join(args.out, "clt_config.json"), "w") as f:
            json.dump({
                "n_layers": cfg.n_layers,
                "d_model": cfg.d_model,
                "d_transcoder": cfg.d_transcoder,
                "prefixes": cfg.prefixes,
                "activation": cfg.activation_function,
                "dtype": cfg.dtype,
            }, f, indent=2)
        print(f"Saved to {args.out}")
        # Optional upload to HF
        if args.hf_repo_id:
            _upload_to_hf(args.hf_repo_id, args.hf_branch, args.out, subdir=args.hf_subdir)
    else:
        # Streaming mode
        def iter_token_batches(ds_name: str, split: str, tokenizer, *, seq_len: int, batch_tokens: int) -> Iterable[torch.Tensor]:
            ds = load_dataset(ds_name, split=split, streaming=True)
            batch = max(1, batch_tokens // seq_len)
            buf = []
            for item in ds.shuffle(buffer_size=10_000):
                text = item["text"] if isinstance(item, dict) else str(item)
                toks = tokenizer.encode(text, add_special_tokens=False)
                for i in range(0, len(toks) - seq_len, seq_len):
                    buf.append(toks[i : i + seq_len])
                    if len(buf) >= batch:
                        arr = np.array(buf[:batch], dtype=np.int64)
                        buf = buf[batch:]
                        yield torch.tensor(arr, dtype=torch.long, device=device)

        def capture_pairs(model: HookedTransformer, tokens: torch.Tensor, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
            hooks_x = [f"blocks.{l}.hook_resid_mid" for l in range(n_layers)]
            hooks_y = [f"blocks.{l}.hook_mlp_out" for l in range(n_layers)]
            cache = {}
            def save(t, hook):
                cache[hook.name] = t.detach().to(dtype)
            fwd_hooks = [(name, save) for name in hooks_x + hooks_y]
            with torch.no_grad():
                _ = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            xs, ys = [], []
            for l in range(n_layers):
                x = cache[f"blocks.{l}.hook_resid_mid"]
                y = cache[f"blocks.{l}.hook_mlp_out"]
                b, p, d = x.shape
                xs.append(x.reshape(b * p, d))
                ys.append(y.reshape(b * p, d))
            return torch.stack(xs, 0), torch.stack(ys, 0)

        model = HookedTransformer.from_pretrained(args.model, device=device, dtype=dtype)
        tokenizer = model.tokenizer

        cfg = MatryoshkaCLTConfig(
            n_layers=args.n_layers,
            d_model=args.d_model,
            d_transcoder=features,
            prefixes=sorted(prefixes),
            activation_function=args.activation,
            lr=args.lr,
            steps=args.steps,
            batch_size=args.batch_tokens,
            l1_coef=args.l1,
            weight_decay=0.0,
            warmup_steps=args.warmup_steps,
            b_dec_init="zero",
            device=str(device),
            dtype=args.dtype,
        )
        trainer = MatryoshkaCLTTrainer(cfg)
        trainer.model.to(device=device, dtype=dtype)

        data_iter = iter_token_batches(args.dataset, args.split, tokenizer, seq_len=args.seq_len, batch_tokens=args.batch_tokens)
        pbar = tqdm(range(args.steps), desc="train")
        initialized = False
        for _ in pbar:
            toks = next(data_iter)  # [batch, seq]
            x, y = capture_pairs(model, toks, n_layers=args.n_layers)
            if not initialized:
                with torch.no_grad():
                    trainer.model.b_dec.copy_(y.mean(dim=1))
                initialized = True
            logs = trainer.step(x.to(device), y.to(device))
            pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))})
            if wandb is not None:
                wandb.log(logs)

        if args.save_safetensors:
            save_clt_safetensors(trainer.model, args.out)
        else:
            torch.save(trainer.model.state_dict(), os.path.join(args.out, "clt.pt"))
        # Save basic config for publishing
        with open(os.path.join(args.out, "clt_config.json"), "w") as f:
            json.dump({
                "n_layers": cfg.n_layers,
                "d_model": cfg.d_model,
                "d_transcoder": cfg.d_transcoder,
                "prefixes": cfg.prefixes,
                "activation": cfg.activation_function,
                "dtype": cfg.dtype,
                "model_name": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "seq_len": args.seq_len,
                "batch_tokens": args.batch_tokens,
            }, f, indent=2)
        print(f"Saved to {args.out}")
        # Optional upload to HF
        if args.hf_repo_id:
            _upload_to_hf(args.hf_repo_id, args.hf_branch, args.out, subdir=args.hf_subdir)

    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def _upload_to_hf(repo_id: str, branch: str, local_dir: str, *, subdir: str | None = None) -> None:
    """Upload artifacts in local_dir to a Hugging Face Hub repo (minimal).

    Requires HF token in environment (HUGGINGFACE_HUB_TOKEN or huggingface-cli login).
    Only uploads common artifact files; does not create README or other extras.
    """
    try:
        from huggingface_hub import HfApi, create_repo
        from huggingface_hub.utils import RepositoryNotFoundError
    except Exception as e:
        print(f"[hf] huggingface_hub not available: {e}")
        return

    api = HfApi()
    # Create repo if missing
    try:
        api.repo_info(repo_id)
    except RepositoryNotFoundError:
        create_repo(repo_id, repo_type="model", exist_ok=True)

    path_in_repo = subdir or ""
    print(f"[hf] Uploading {local_dir} -> {repo_id}:{branch}/{path_in_repo or '.'}")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        path_in_repo=path_in_repo,
        commit_message="Upload Matryoshka CLT artifacts",
        allow_patterns=["*.safetensors", "*.pt", "*.json"],
    )


if __name__ == "__main__":
    main()
