from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_ar1_series(
    phi: float = 0.7,
    length: int = 128,
    noise_std: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one AR(1) time series."""
    rng = np.random.default_rng(seed)

    x = np.zeros(length, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=noise_std, size=length).astype(np.float32)

    for t in range(1, length):
        x[t] = phi * x[t - 1] + noise[t]

    return x


def create_ar1_fixed_dataset(
    num_samples: int,
    context_length: int = 64,
    phi: float = 0.7,
    noise_std: float = 1.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create AR(1) data where every sample uses the same fixed phi."""
    rng = np.random.default_rng(seed)
    series_length = context_length + 1

    x_data = np.zeros((num_samples, context_length), dtype=np.float32)
    y_data = np.zeros(num_samples, dtype=np.float32)
    phi_values = np.full((num_samples, 1), phi, dtype=np.float32)

    for i in range(num_samples):
        series_seed = int(rng.integers(0, 1_000_000))
        series = generate_ar1_series(
            phi=phi,
            length=series_length,
            noise_std=noise_std,
            seed=series_seed,
        )

        x_data[i] = series[:context_length]
        y_data[i] = series[context_length]

    return x_data, y_data, phi_values


def create_ar1_sampled_dataset(
    num_samples: int,
    context_length: int = 64,
    phi_low: float = -0.9,
    phi_high: float = 0.9,
    noise_std: float = 1.0,
    seed: int = 456,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create AR(1) data where each sample has a different phi value."""
    rng = np.random.default_rng(seed)
    series_length = context_length + 1

    x_data = np.zeros((num_samples, context_length), dtype=np.float32)
    y_data = np.zeros(num_samples, dtype=np.float32)
    phi_values = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        phi = float(rng.uniform(phi_low, phi_high))
        series_seed = int(rng.integers(0, 1_000_000))
        series = generate_ar1_series(
            phi=phi,
            length=series_length,
            noise_std=noise_std,
            seed=series_seed,
        )

        x_data[i] = series[:context_length]
        y_data[i] = series[context_length]
        phi_values[i, 0] = phi

    return x_data, y_data, phi_values


def save_dataset(
    save_path: Path,
    x_data: np.ndarray,
    y_data: np.ndarray,
    phi_values: np.ndarray,
    p: int,
    context_length: int,
    noise_std: float,
    extra_metadata: dict[str, float] | None = None,
) -> None:
    """Save dataset as .npz file."""
    metadata = {
        "X": x_data,
        "y": y_data,
        "phi": phi_values,
        "p": np.array([p], dtype=np.int64),
        "context_length": np.array([context_length], dtype=np.int64),
        "noise_std": np.array([noise_std], dtype=np.float32),
    }

    if extra_metadata is not None:
        for key, value in extra_metadata.items():
            metadata[key] = np.array([value], dtype=np.float32)

    np.savez(save_path, **metadata)


def plot_example_series(output_dir: Path) -> None:
    """Save one example AR(1) plot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    series = generate_ar1_series(phi=0.7, length=128, noise_std=1.0, seed=42)

    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.title("Example AR(1) time series with phi = 0.7")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_dir / "ar1_example.png", dpi=300)
    plt.close()


def generate_all_ar1_datasets() -> None:
    """Generate fixed-phi and sampled-phi AR(1) datasets."""
    processed_dir = Path("data/processed")
    examples_dir = Path("data/examples")

    processed_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    context_length = 64
    noise_std = 1.0

    splits = {
        "train": 10_000,
        "val": 1_000,
        "test": 1_000,
    }

    fixed_phi = 0.7

    print("Generating fixed-phi AR(1) datasets...")

    for split_name, num_samples in splits.items():
        x_data, y_data, phi_values = create_ar1_fixed_dataset(
            num_samples=num_samples,
            context_length=context_length,
            phi=fixed_phi,
            noise_std=noise_std,
            seed=123 + len(split_name),
        )

        save_path = processed_dir / f"ar1_fixed_{split_name}.npz"

        save_dataset(
            save_path=save_path,
            x_data=x_data,
            y_data=y_data,
            phi_values=phi_values,
            p=1,
            context_length=context_length,
            noise_std=noise_std,
            extra_metadata={"fixed_phi": fixed_phi},
        )

        print(f"Saved {save_path}")
        print(f"  X shape: {x_data.shape}")
        print(f"  y shape: {y_data.shape}")
        print(f"  phi shape: {phi_values.shape}")

    phi_low = -0.9
    phi_high = 0.9

    print("\nGenerating sampled-phi AR(1) datasets...")

    for split_name, num_samples in splits.items():
        x_data, y_data, phi_values = create_ar1_sampled_dataset(
            num_samples=num_samples,
            context_length=context_length,
            phi_low=phi_low,
            phi_high=phi_high,
            noise_std=noise_std,
            seed=456 + len(split_name),
        )

        save_path = processed_dir / f"ar1_sampled_{split_name}.npz"

        save_dataset(
            save_path=save_path,
            x_data=x_data,
            y_data=y_data,
            phi_values=phi_values,
            p=1,
            context_length=context_length,
            noise_std=noise_std,
            extra_metadata={
                "phi_low": phi_low,
                "phi_high": phi_high,
            },
        )

        print(f"Saved {save_path}")
        print(f"  X shape: {x_data.shape}")
        print(f"  y shape: {y_data.shape}")
        print(f"  phi shape: {phi_values.shape}")
        print(f"  phi min: {phi_values.min():.3f}")
        print(f"  phi max: {phi_values.max():.3f}")

    plot_example_series(examples_dir)
    print("\nSaved example plot to data/examples/ar1_example.png")


if __name__ == "__main__":
    generate_all_ar1_datasets()