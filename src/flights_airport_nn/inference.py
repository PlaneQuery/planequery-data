from datetime import date
from pathlib import Path

import numpy as np
import torch

from flights_airport_nn.features.features import FlightFeatures, create_features_for_day
from flights_airport_nn.utils import device
from flights_airport_nn.model import AirportGRU


def load_model(
    model_path: str | Path,
) -> AirportGRU:
    """Load a trained airport GRU on the best available device."""
    model = AirportGRU().to(device)
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

def inferernce():
    # we batch input it. But maybe to start we go one by one?
    features = create_features_for_day(date(2026, 3, 1))
    flight_features: list[FlightFeatures] = []
    max_length = features.max_messages_length()
    model = load_model(
        model_path="/Volumes/T2-SSD/planequery/data/models/flights_airport_nn/runs/2026-09-13_17-39_9a999b3_0days_0epochs/model.pt",
    )
    for pf in features.plane_features:
        for ff in pf.flight_features:
            flight_features.append(ff)
    
    for ff in flight_features:
        max_score = -np.inf
        for c in ff.candidates:
            X, mask, y = c.features_to_numpy(max_length)
            output = predict(model, X) # yes, this is scaler.
            if output > max_score:
                max_score = output
                ff.predicted_airport = 
            print(output.shape)
'''
'''


def run_inferernce_day(target_dt: date):
    pass

if __name__ == "__main__":
    inferernce()