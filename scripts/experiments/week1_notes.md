# Week 1 Notes

## Repository setup

- Forked the original repository `jariskueken/tsfm-playground`.
- Working fork: `as2203-ufr/tsfm-playground`.
- Working branch: `ar-experiments`.
- Verified that the local Git remote points to the fork, not the original repository.

## Environment setup

- The project uses `uv`.
- Created and synced the virtual environment.
- Verified that `scripts/training.py` and `scripts/evaluate.py` run, but they currently do not contain a full training/evaluation pipeline.

## Patching mechanism

Patching-related code was found mainly in:

- `src/playground/layers/patching.py`
- `src/playground/models/uv.py`
- `src/playground/models/transformer.py`
- `notebooks/model.ipynb`

The current implementation uses patches. According to the project proposal, this needs to be changed later so that scalar tokenization is used:

- current idea: one patch = several timesteps
- required idea: one token = one timestep

## AR(1) data generation

Added:

- `scripts/generate_ar_data.py`
- `scripts/check_ar_data.py`

The data generation script creates:

- fixed-coefficient AR(1) datasets with phi = 0.7
- sampled-coefficient AR(1) datasets with phi sampled from [-0.9, 0.9]

Each sample has:

- `X`: 64 context values
- `y`: next value
- `phi`: AR coefficient
- `p`: AR order

Generated local files:

- `data/processed/ar1_fixed_train.npz`
- `data/processed/ar1_fixed_val.npz`
- `data/processed/ar1_fixed_test.npz`
- `data/processed/ar1_sampled_train.npz`
- `data/processed/ar1_sampled_val.npz`
- `data/processed/ar1_sampled_test.npz`

These generated data files are ignored by Git using `.gitignore`.

## Dataset check

The check script verifies:

- X shape
- y shape
- phi shape
- AR order p
- context length
- phi min/max
- basic consistency between X, y, and phi

The dataset check passed.
## DataLoader and first training run

Added:

- `scripts/check_ar_loader.py`
- updated `scripts/training.py`

The AR(1) fixed dataset is now connected to a PyTorch `Dataset` and `DataLoader`.

The DataLoader check produced the expected batch shapes:

- `X`: `[64, 64]`
- `y`: `[64]`
- `phi`: `[64, 1]`

A first scalar-token transformer training run was implemented in `scripts/training.py`.

Current training setup:

- dataset: fixed AR(1), `phi = 0.7`
- context length: 64
- task: predict the next value
- loss: mean squared error
- model input: one scalar timestep per token

The first training run completed and printed training/validation loss for 20 epochs.

Current result:

- analytic AR(1) baseline validation MSE: approximately `0.9665`
- final transformer validation MSE: approximately `0.9845`

This means the training pipeline works, but the model does not yet outperform the analytic AR(1) baseline. The next step is to improve the training setup or model configuration.

## Next steps

- Improve the AR(1) training run so the model gets closer to or better than the analytic baseline.
- Clarify whether scalar tokenization should be implemented by setting `patch_size = 1` and `patch_stride = 1`, or by fully removing the patching module.
- Add attention extraction for the AR(1) model.
- Add perturbation-based lag sensitivity for AR(1).
- Later extend data generation and training to AR(2), AR(3), and AR(5).