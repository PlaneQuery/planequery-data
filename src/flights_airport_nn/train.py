from datetime import date
import os
import torch
from torch.utils.data import DataLoader

from flights_airport_nn.dataset import (
    AirportCandidateDataset,
    collate_airport_candidates,
)
from flights_airport_nn.utils import device, model_folder_path
from flights_airport_nn.features.features import (
    ALL_FEATURE_CONFIG,
    FeatureConfig,
)
from flights_airport_nn.model import AirportGRU


def run_train(
    train_dates: list[date] | tuple[date, ...],
    feature_config: FeatureConfig = ALL_FEATURE_CONFIG,
    epochs: int = 2,
):
    if not train_dates:
        raise ValueError("train_dates must contain at least one date")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    print(device)
    model = AirportGRU(input_size=feature_config.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataset = AirportCandidateDataset(train_dates, feature_config)
    loader = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=collate_airport_candidates,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for X_batch, mask_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            scores = model(X_batch, mask_batch)
            loss = loss_fn(scores, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("training dates produced no airport candidates")

        print(f"Epoch {epoch + 1:02d} Loss: {total_loss / batch_count:.4f}")

    path = model_folder_path() / "model.pt"
    os.makedirs(path.parent, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": [
                feature.value
                for feature in feature_config.enabled
            ],
            "hidden_size": model.hidden_size,
        },
        path,
    )
    return model


if __name__ == "__main__":
    run_train([date(2026, 2, 1)], ALL_FEATURE_CONFIG)
