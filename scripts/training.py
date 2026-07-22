from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ARDataset(Dataset):
    """PyTorch Dataset for AR(.npz) files."""

    def __init__(self, dataset_path: Path) -> None:
        data = np.load(dataset_path)

        self.x_data = torch.tensor(data["X"], dtype=torch.float32)
        self.y_data = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[index], self.y_data[index]


class ScalarTransformerForecaster(nn.Module):
    """Small transformer with scalar tokenization: one timestep = one token."""

    def __init__(
        self,
        context_length: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, context_length, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.forecast_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next value from context.

        Input:
            x: [batch_size, context_length]

        Output:
            prediction: [batch_size]
        """
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = x + self.position_embedding[:, : x.shape[1], :]

        encoded = self.encoder(x)

        final_token = encoded[:, -1, :]
        prediction = self.forecast_head(final_token).squeeze(-1)

        return prediction


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train model for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def fixed_ar1_baseline_mse(
    dataloader: DataLoader,
    phi: float = 0.7,
) -> float:
    """Analytic AR(1) baseline for fixed phi."""
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        prediction = phi * x_batch[:, -1]
        loss = torch.mean((prediction - y_batch) ** 2)

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_loss_plot(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> None:
    """Save training/validation loss plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("AR(1) fixed training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    """Train first AR(1) fixed-coefficient model."""
    train_path = Path("data/processed/ar1_fixed_train.npz")
    val_path = Path("data/processed/ar1_fixed_val.npz")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "AR(1) data not found. Run "
            "`uv run python scripts/generate_ar_data.py` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ARDataset(train_path)
    val_dataset = ARDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
    )

    model = ScalarTransformerForecaster(context_length=64).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    baseline_mse = fixed_ar1_baseline_mse(val_loader, phi=0.7)
    print(f"Analytic AR(1) baseline validation MSE: {baseline_mse:.6f}")

    num_epochs = 20
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train MSE: {train_loss:.6f} | "
            f"val MSE: {val_loss:.6f}"
        )

    save_loss_plot(
        train_losses=train_losses,
        val_losses=val_losses,
        output_path=Path("results/figures/ar1_fixed_training_loss.png"),
    )

    print("\nSaved loss plot to results/figures/ar1_fixed_training_loss.png")
    print("AR(1) fixed training run completed.")


if __name__ == "__main__":
    main()