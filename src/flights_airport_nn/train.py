from datetime import date

import torch
from torch.utils.data import DataLoader, TensorDataset

from flights_airport_nn.features.features import create_features_for_day
from flights_airport_nn.model import AirportGRU
from flights_airport_nn.utils import device


print(device)


# --------------------------------------------------
# Data
# --------------------------------------------------

# Shape:
# X = (num_examples, sequence_length, num_features)
# y = (num_examples,)
#
# Each example is:
#   ADS-B trajectory relative to ONE candidate airport
#
# y = 1 -> correct airport
# y = 0 -> incorrect airport

X,mask, y = create_features_for_day(date(2026,3,1))
X = torch.from_numpy(X)
y = torch.from_numpy(y)
dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)


# --------------------------------------------------
# Model
# --------------------------------------------------


model = AirportGRU(
    input_size=X.shape[2],
    hidden_size=128,
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

loss_fn = torch.nn.BCEWithLogitsLoss()


# --------------------------------------------------
# Training
# --------------------------------------------------

epochs = 20

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

    print(
        f"Epoch {epoch + 1:02d} "
        f"Loss: {total_loss / len(loader):.4f}"
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

torch.save(model.state_dict(), "airport_gru.pt")
