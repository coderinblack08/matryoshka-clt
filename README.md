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
- Train a CLT with nested prefixes (example, hyperparams inspired by sae_training transcoders):
  - `uv run matryoshka-clt --data ./pairs.npz --n-layers 4 --d-model 256 --expansion 32 --prefixes 128 256 512 --steps 200 --batch-size 4096 --lr 4e-4 --l1 1.4e-4 --warmup-steps 5000 --out ./out --save-safetensors`
- Load the trained CLT in circuit-tracer (example):
  - `python - << 'PY'
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
clt = load_clt('./out')
print('Loaded CLT with', clt.n_layers, 'layers and', clt.d_transcoder, 'features')
PY`

## Flags

- `--features` or `--expansion` (default 32): set feature count directly or compute `d_transcoder = expansion * d_model`
- `--activation`: `relu` (default) or `jump_relu`
- `--lr` (default 4e-4), `--l1` (default 1.4e-4), `--warmup-steps` (default 5000)
- `--batch-size` (default 4096), `--dtype`: `float32|bfloat16|float16`
- `--b-dec-init` (default `mean`): initialize decoder bias to mean(y) or zeros
- `--save-safetensors`: write `W_enc_*.safetensors`, `W_dec_*.safetensors` compatible with circuit-tracer

## Notes

- You are responsible for generating real `x`/`y` activations from your model (e.g., with transformer_lens hooks at `hook_resid_mid` and `hook_mlp_out`).
- `vendor/circuit_tracer` is importable via `sitecustomize.py`; no manual sys.path changes are needed.
