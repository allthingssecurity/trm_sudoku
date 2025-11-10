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

`uv run python src/nn/train.py experiment=trm_sudoku4x4 trainer=gpu trainer.accelerator=mps timekeeping.max_epochs=5`

Stable longer run (good results):

`uv run python src/nn/train.py experiment=trm_sudoku4x4 trainer=gpu trainer.accelerator=mps trainer.precision=32-true timekeeping.max_epochs=60 timekeeping.batch_size=128 model_tuning.hidden_size=128 model_tuning.num_layers=2 model_tuning.N_supervision=2 model_tuning.learning_rate=3e-4 model_tuning.learning_rate_emb=3e-3`

# Evaluation

Sudoku is evaluated via the validation loop and simple scripts. Example to validate a checkpoint:

`uv run python - <<'PY'
from lightning import Trainer
import torch
from src.nn.data.sudoku4x4_datamodule import SudokuDataModule
from src.nn.models.trm_module import TRMModule
ckpt='train/runs/<timestamp>/checkpoints/last.ckpt'
dm=SudokuDataModule(batch_size=128,num_workers=0,grid_size=4,max_grid_size=6,generate_on_fly=True,num_train_puzzles=2000,num_val_puzzles=800)
dm.setup('fit')
device='mps' if torch.backends.mps.is_available() else 'cpu'
model=TRMModule.load_from_checkpoint(ckpt,map_location=device)
print(Trainer(accelerator=device,devices=1).validate(model,dm.val_dataloader()))
PY`

# Sample Predictions

You can print a few solved puzzles from a held-out batch using a small script in the discussion above.

# Note to contributors

If you would like to make contributions to this codebase, here are things you can do:

- Help reproduce the results of the original paper
- Implement missing features (carry, puzzle embeddings)
