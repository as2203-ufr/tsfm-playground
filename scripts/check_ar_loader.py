from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ARDataset(Dataset):
    """PyTorch Dataset for AR(.npz) files."""

    def __init__(self, dataset_path: Path) -> None:
        data = np.load(dataset_path)

        self.x_data = torch.tensor(data["X"], dtype=torch.float32)
        self.y_data = torch.tensor(data["y"], dtype=torch.float32)
        self.phi_values = torch.tensor(data["phi"], dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_data[index], self.y_data[index], self.phi_values[index]


def main() -> None:
    dataset_path = Path("data/processed/ar1_fixed_train.npz")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run `uv run python scripts/generate_ar_data.py` first."
        )

    dataset = ARDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    x_batch, y_batch, phi_batch = next(iter(dataloader))

    print("AR(1) DataLoader check")
    print(f"Number of samples: {len(dataset)}")
    print(f"Batch X shape: {x_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    print(f"Batch phi shape: {phi_batch.shape}")

    print("\nFirst batch example:")
    print(f"First X sample shape: {x_batch[0].shape}")
    print(f"First y value: {y_batch[0]}")
    print(f"First phi value: {phi_batch[0]}")


if __name__ == "__main__":
    main()