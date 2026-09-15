from datetime import date
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flights_airport_nn.utils import device, model_folder_path
from flights_airport_nn.features.features import create_features_for_day
from flights_airport_nn.model import AirportGRU

def train() -> None:
    print(device)

    features = create_features_for_day(date(2026, 3, 1))

    X, mask, y, candidate_rows = features.numpy_features()

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = AirportGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    epochs = 2

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            scores = model(X_batch)
            loss = loss_fn(scores, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1:02d} Loss: {total_loss / len(loader):.4f}")

    path = model_folder_path() / "model.pt"
    os.makedirs(path.parent, exist_ok=True)
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    train()
