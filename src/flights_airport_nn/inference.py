from pathlib import Path

import numpy as np
import torch

from flights_airport_nn.device import device
from flights_airport_nn.model import AirportGRU


def load_model(
    model_path: str | Path,
    input_size: int,
    hidden_size: int = 128,
) -> AirportGRU:
    """Load a trained airport GRU on the best available device."""
    model = AirportGRU(input_size=input_size, hidden_size=hidden_size).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict(model: AirportGRU, features: np.ndarray) -> np.ndarray:
    """Return the probability that each airport candidate is correct."""
    inputs = torch.as_tensor(features, dtype=torch.float32, device=device)

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(inputs))

    return probabilities.cpu().numpy()
