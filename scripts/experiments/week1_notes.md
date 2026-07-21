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

## Next steps

- Connect the AR(1) dataset to a PyTorch Dataset/DataLoader.
- Create or complete the training pipeline in `scripts/training.py`.
- Run first AR(1) fixed-coefficient training.
- Modify the provided model to remove patching or use scalar tokenization.
- Later extend data generation to AR(2), AR(3), and AR(5).