# An implementation of TRM

This is an implementation of the [Tiny Recursive Model (TRM)](https://arxiv.org/pdf/2510.04871v1)

Reference [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

# Scope

This repo focuses on training and solving Sudoku 4x4 with TRM. No external datasets are required — puzzles are generated on the fly.

# Setup

On macOS, install `uv` via Homebrew or the official installer:

- Homebrew: `brew install uv`
- Script: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Alternatively you can use a virtual environment with `python -m venv .venv && source .venv/bin/activate` and `pip install -e .`.

# Training

Quick GPU sanity run (macOS MPS):

```bash
uv run python src/nn/train.py \
  experiment=trm_sudoku4x4 \
  trainer=gpu trainer.accelerator=mps \
  timekeeping.max_epochs=5
```

Stable longer run (good results):

```bash
uv run python src/nn/train.py \
  experiment=trm_sudoku4x4 \
  trainer=gpu trainer.accelerator=mps trainer.precision=32-true \
  timekeeping.max_epochs=60 timekeeping.batch_size=128 \
  model_tuning.hidden_size=128 model_tuning.num_layers=2 \
  model_tuning.N_supervision=2 \
  model_tuning.learning_rate=3e-4 model_tuning.learning_rate_emb=3e-3
```

# Evaluation

Sudoku is evaluated via the validation loop and simple scripts. Example to validate a checkpoint:

```bash
uv run python - <<'PY'
from lightning import Trainer
import torch
from src.nn.data.sudoku4x4_datamodule import SudokuDataModule
from src.nn.models.trm_module import TRMModule

ckpt = 'train/runs/<timestamp>/checkpoints/last.ckpt'

dm = SudokuDataModule(
    batch_size=128, num_workers=0,
    grid_size=4, max_grid_size=6,
    generate_on_fly=True,
    num_train_puzzles=2000, num_val_puzzles=800,
)
dm.setup('fit')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = TRMModule.load_from_checkpoint(ckpt, map_location=device)

print(Trainer(accelerator=device, devices=1).validate(model, dm.val_dataloader()))
PY
```

# Sample Predictions

Here is a short script to print a few held-out Sudoku4x4 puzzles with predictions:

```bash
uv run python - <<'PY'
import torch, numpy as np
from src.nn.data.sudoku4x4_datamodule import SudokuDataModule
from src.nn.models.trm_module import TRMModule

def decode_token(t):
    if t <= 2: return 0
    return int(t-2)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
ckpt = 'train/runs/<timestamp>/checkpoints/last.ckpt'  # replace with your checkpoint
model = TRMModule.load_from_checkpoint(ckpt, map_location=device).to(device).eval()

N=3; GRID=4; MAX=6
dm = SudokuDataModule(batch_size=N, num_workers=0, grid_size=4, max_grid_size=6,
                      generate_on_fly=True, num_train_puzzles=2000, num_val_puzzles=N)
dm.setup('fit')
batch = next(iter(dm.val_dataloader()))
for k,v in list(batch.items()):
    if isinstance(v, torch.Tensor): batch[k]=v.to(device)

with torch.no_grad():
    carry = model.initial_carry(batch)
    while True:
        carry, outputs = model.forward(carry, batch)
        if carry.halted.all(): break
    preds = outputs['logits'].argmax(dim=-1)

labels = batch['output']
mask = labels != -100
pix_acc = (preds==labels)[mask].float().mean().item()
print(f'Batch pixel accuracy: {pix_acc:.3f}\n')

for i in range(N):
    pred = preds[i].cpu().numpy().reshape(MAX,MAX)
    lab  = labels[i].cpu().numpy().reshape(MAX,MAX)
    inp  = batch['input'][i].cpu().numpy().reshape(MAX,MAX)
    pred4 = np.vectorize(decode_token)(pred)[:GRID,:GRID]
    lab4  = np.vectorize(decode_token)(lab)[:GRID,:GRID]
    inp4  = np.vectorize(decode_token)(inp)[:GRID,:GRID]
    exact = int((pred4==lab4).all())
    print(f'Sample {i+1} (exact: {exact})')
    print('Input:')
    print(inp4)
    print('Target:')
    print(lab4)
    print('Pred:')
    print(pred4)
    print('-'*28)
PY
```

Example result from our 60‑epoch stable run (128×2, N_supervision=2): all samples exact and batch pixel accuracy 1.000.

Examples (from a held‑out batch)

Input:
```
[[0 0 0 0]
 [0 4 0 1]
 [0 0 0 4]
 [0 3 1 0]]
```
Target / Pred:
```
[[1 2 4 3]
 [3 4 2 1]
 [2 1 3 4]
 [4 3 1 2]]
```

Input:
```
[[0 0 4 2]
 [2 0 0 1]
 [3 0 0 4]
 [0 0 0 0]]
```
Target / Pred:
```
[[1 3 4 2]
 [2 4 3 1]
 [3 1 2 4]
 [4 2 1 3]]
```

Input:
```
[[0 4 1 3]
 [0 0 0 0]
 [4 0 3 1]
 [0 1 0 0]]
```
Target / Pred:
```
[[2 4 1 3]
 [1 3 2 4]
 [4 2 3 1]
 [3 1 4 2]]
```

# Note to contributors

If you would like to make contributions to this codebase, here are things you can do:

- Help reproduce the results of the original paper
- Implement missing features (carry, puzzle embeddings)
