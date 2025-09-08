# Matryoshka-CLT

Train a Cross-Layer Transcoder with a Matryoshka loss to encourage more interpretable features and cleaner attribution graphs. Uses safety-research/circuit-tracer's `CrossLayerTranscoder` under the hood.

## Project layout

```
matryoshka-clt/
├── src/
│   ├── matryoshka_clt/
│   │   ├── cli.py                    # CLI entry (`matryoshka-clt`) to train/export CLTs
│   │   └── matryoshka_clt.py         # Matryoshka CLT trainer, config, and safetensors export
│   └── sitecustomize.py              # makes `vendor/circuit_tracer` importable automatically
├── vendor/
│   └── circuit_tracer/               # vendored circuit-tracer library (Transcoder implementation)
├── pyproject.toml                    # package and CLI definition (uv project)
└── README.md                         # this file
```

## Quick start

- Create environment and install deps:
  - `uv venv && uv sync`
- Prepare training data (`pairs.npz`):
  - File must contain arrays:
    - `x`: residual inputs at read hook, shape `[n_layers, n_pos, d_model]`
    - `y`: MLP outputs at write hook, shape `[n_layers, n_pos, d_model]`
  - Minimal dummy example to sanity-check the pipeline: - `python - << 'PY'
import numpy as np
n_layers, n_pos, d_model = 4, 512, 256
np.savez('pairs.npz', x=np.random.randn(n_layers,n_pos,d_model).astype('float32'), y=np.random.randn(n_layers,n_pos,d_model).astype('float32'))
PY`
- Train from precomputed pairs.npz (offline):
  - `uv run matryoshka-clt --data ./pairs.npz --n-layers 4 --d-model 256 --expansion 32 --prefixes 128 256 512 --steps 200 --lr 4e-4 --l1 1.4e-4 --warmup-steps 5000 --out ./out --save-safetensors`
- Stream dataset with GPT‑2 small (online):
  - `uv run matryoshka-clt --dataset Skylion007/openwebtext --split train --model gpt2-small --n-layers 12 --d-model 768 --expansion 32 --seq-len 128 --batch-tokens 4096 --prefixes 3072 6144 12288 24576 --steps 2000 --lr 4e-4 --l1 1.4e-4 --warmup-steps 5000 --out out_gpt2_clt --save-safetensors`
- Load the trained CLT in circuit-tracer (example):
  - `python - << 'PY'
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
clt = load_clt('./out')
print('Loaded CLT with', clt.n_layers, 'layers and', clt.d_transcoder, 'features')
PY`

## Logging & Upload

- Enable W&B logging (optional):
  - `wandb login` (once per machine)
  - Add flags: `--wandb --wandb-project matryoshka-clt [--wandb-entity your_team]`
- Upload artifacts to Hugging Face (optional):
  - `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`
  - Add flags: `--hf-repo-id yourname/matryoshka-clt --hf-branch main [--hf-subdir gpt2-small]`
  - Uploads files from `--out` matching: `*.safetensors`, `*.pt`, `*.json`

## Flags

- `--features` or `--expansion` (default 32): set feature count directly or compute `d_transcoder = expansion * d_model`
- `--activation`: `relu` (default) or `jump_relu`
- `--lr` (default 4e-4), `--l1` (default 1.4e-4), `--warmup-steps` (default 5000)
- `--batch-size` (default 4096), `--dtype`: `float32|bfloat16|float16`
- `--b-dec-init` (default `mean`): initialize decoder bias to mean(y) or zeros
- `--save-safetensors`: write `W_enc_*.safetensors`, `W_dec_*.safetensors` compatible with circuit-tracer
- `--wandb`, `--wandb-project`, `--wandb-entity`: enable and configure W&B logging
- `--hf-repo-id`, `--hf-branch`, `--hf-subdir`: upload artifacts to the Hugging Face Hub

## Notes

- You are responsible for generating real `x`/`y` activations from your model (e.g., with transformer_lens hooks at `hook_resid_mid` and `hook_mlp_out`).
- `vendor/circuit_tracer` is importable via `sitecustomize.py`; no manual sys.path changes are needed.
