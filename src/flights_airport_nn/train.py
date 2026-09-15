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


def run_train( # TODO: have this take in features not dates 
    train_dates: list[date] | tuple[date, ...],
    feature_config: FeatureConfig = ALL_FEATURE_CONFIG,
):
    if not train_dates:
        raise ValueError("train_dates must contain at least one date")

    print(device)
    model = AirportGRU(input_size=feature_config.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    epochs = 2

    for train_dt in train_dates:
        features = create_features_for_day(train_dt)
        X, mask, y, _ = features.numpy_features(feature_config)
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(mask),
            torch.from_numpy(y),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

            print(
                f"{train_dt} Epoch {epoch + 1:02d} "
                f"Loss: {total_loss / len(loader):.4f}"
            )

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
