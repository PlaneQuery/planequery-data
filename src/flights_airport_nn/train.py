from datetime import date
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from flights_airport_nn.utils import device, model_folder_path
from flights_airport_nn.features.features import (
    ALL_FEATURE_CONFIG,
    FeatureConfig,
    create_features_for_day,
)
from flights_airport_nn.model import AirportGRU


def train(
    train_dt: date,
    feature_config: FeatureConfig = ALL_FEATURE_CONFIG,
):
    print(device)

    features = create_features_for_day(train_dt)

    X, mask, y, candidate_rows = features.numpy_features(feature_config)

    X = torch.from_numpy(X)
    mask = torch.from_numpy(mask)
    y = torch.from_numpy(y)
    dataset = TensorDataset(X, mask, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = AirportGRU(input_size=feature_config.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    epochs = 2

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, mask_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            scores = model(X_batch, mask_batch)
            loss = loss_fn(scores, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1:02d} Loss: {total_loss / len(loader):.4f}")

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


def run_train(
    train_dates: list[date] | tuple[date, ...],
    feature_config: FeatureConfig = ALL_FEATURE_CONFIG,
):
    model = train(train_dates[0], feature_config)
    return model


if __name__ == "__main__":
    run_train([date(2026, 2, 1)], ALL_FEATURE_CONFIG)
