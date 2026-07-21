from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def check_dataset(dataset_path: Path) -> None:
    """Load and print basic information about one AR dataset."""
    data = np.load(dataset_path)

    x_data = data["X"]
    y_data = data["y"]
    phi_values = data["phi"]
    p = int(data["p"][0])
    context_length = int(data["context_length"][0])
    noise_std = float(data["noise_std"][0])

    print(f"\nChecking dataset: {dataset_path}")
    print(f"  X shape: {x_data.shape}")
    print(f"  y shape: {y_data.shape}")
    print(f"  phi shape: {phi_values.shape}")
    print(f"  p: {p}")
    print(f"  context length: {context_length}")
    print(f"  noise std: {noise_std}")
    print(f"  phi min: {phi_values.min():.3f}")
    print(f"  phi max: {phi_values.max():.3f}")
    print(f"  first 5 phi values: {phi_values[:5].flatten()}")

    if x_data.shape[1] != context_length:
        raise ValueError("Context length does not match X shape.")

    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("Number of X samples and y samples do not match.")

    if x_data.shape[0] != phi_values.shape[0]:
        raise ValueError("Number of X samples and phi values do not match.")

    print("  Dataset check passed.")


def plot_first_sample(dataset_path: Path, output_path: Path) -> None:
    """Plot the first context-target pair from a dataset."""
    data = np.load(dataset_path)

    x_data = data["X"]
    y_data = data["y"]
    context_length = int(data["context_length"][0])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(range(context_length), x_data[0], label="context")
    plt.scatter(context_length, y_data[0], label="target")
    plt.title("First AR(1) sample: context + target")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved sample plot to {output_path}")


def main() -> None:
    """Check generated AR(1) datasets."""
    dataset_paths = [
        Path("data/processed/ar1_fixed_train.npz"),
        Path("data/processed/ar1_fixed_val.npz"),
        Path("data/processed/ar1_fixed_test.npz"),
        Path("data/processed/ar1_sampled_train.npz"),
        Path("data/processed/ar1_sampled_val.npz"),
        Path("data/processed/ar1_sampled_test.npz"),
    ]

    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}. "
                "Run `uv run python scripts/generate_ar_data.py` first."
            )

        check_dataset(dataset_path)

    plot_first_sample(
        dataset_path=Path("data/processed/ar1_fixed_train.npz"),
        output_path=Path("data/examples/ar1_first_sample.png"),
    )


if __name__ == "__main__":
    main()