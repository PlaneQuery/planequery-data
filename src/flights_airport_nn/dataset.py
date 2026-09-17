"""Versioned daily caches and complete endpoint candidate groups."""

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from data_engineering.utils import OUTPUT_DIR
from flights_airport_nn.features.features import (
    ALL_FEATURE_CONFIG,
    CANDIDATE_RADIUS_KM,
    FEATURE_PERIOD,
    FEATURE_VERSION,
    MAX_CANDIDATES,
    MAX_POINTS,
    create_features_for_day,
)

CACHE_DIR = OUTPUT_DIR / "data/processed/flights_airport_nn"


def prepare_dates(dates, workers=1):
    """Prepare independent days in spawned processes with bounded concurrency."""
    if workers < 1:
        raise ValueError("workers must be positive")
    dates = list(dict.fromkeys(dates))
    if workers == 1:
        return [prepare_day(day) for day in dates]
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as pool:
        return list(pool.map(prepare_day, dates))


def cache_path(day):
    return CACHE_DIR / f"v{FEATURE_VERSION}" / f"{day.isoformat()}.npz"


def source_signature(day):
    from data_engineering.adsb.read_adsb import _parquet_day_dir
    from data_engineering.flights.flight_type import flights_algorithm_output_path
    from flights.ground_truth.bts_and_sfdps_to_gold import (
        ground_truth_flights_output_path,
    )

    paths = [
        flights_algorithm_output_path(day, no_airports_model=True),
        ground_truth_flights_output_path(day, "adsblol"),
        OUTPUT_DIR / "data/raw/ourairports/airports.csv",
    ]
    paths += sorted(_parquet_day_dir(day).glob("icao_bucket=*/data.parquet"))
    # Include feature implementation and retrieval parameters in cache identity.
    from flights_airport_nn.features import features

    paths += [
        Path(features.__file__),
        Path(__file__).parent / "features/ground_truth_join_segmentation_algorithm.py",
    ]
    stats = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(
        json.dumps(
            [
                stats,
                MAX_POINTS,
                MAX_CANDIDATES,
                CANDIDATE_RADIUS_KM,
                FEATURE_PERIOD.total_seconds(),
            ]
        ).encode()
    ).hexdigest()


def prepare_day(day, rebuild=False):
    path = cache_path(day)
    signature = source_signature(day)
    if path.exists() and not rebuild:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["signature"]) == signature:
                return path
    features = create_features_for_day(day)
    endpoints = features.endpoints
    n = len(endpoints)
    x = np.zeros(
        (n, MAX_CANDIDATES, MAX_POINTS, ALL_FEATURE_CONFIG.input_size), np.float16
    )
    lengths = np.zeros(n, np.int16)
    counts = np.zeros(n, np.int16)
    target = np.full(n, -1, np.int64)
    airports = np.full((n, MAX_CANDIDATES), "", dtype="U16")
    truth = np.full(n, "", dtype="U16")
    nearest = np.full(n, -1, np.int64)
    for i, endpoint in enumerate(endpoints):
        counts[i] = len(endpoint.candidates)
        # Preserve unsupported catalog labels as retrieval failures in evaluation.
        truth[i] = (
            getattr(
                features.flights_by_id[endpoint.flight_id],
                endpoint.endpoint + "_airport_ident",
            )
            or ""
        )
        for j, candidate in enumerate(endpoint.candidates):
            lengths[i] = len(candidate.features)
            x[i, j, : lengths[i]] = candidate.features
            airports[i, j] = candidate.predicted_airport.ident
            if candidate.predicted_airport == endpoint.true_airport:
                target[i] = j
        if counts[i]:
            nearest[i] = int(np.argmin(x[i, : counts[i], 0, 4]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            x=x,
            lengths=lengths,
            counts=counts,
            target=target,
            airports=airports,
            truth=truth,
            nearest=nearest,
            endpoint=np.array([f.endpoint for f in endpoints]),
            flight_id=np.array([f.flight_id for f in endpoints]),
            signature=np.array(signature),
        )
    temp.replace(path)
    print(
        f"{day}: {n} endpoints, candidate recall {(target >= 0).mean():.4%}, cache {path}",
        flush=True,
    )
    return path


class AirportCandidateDataset(IterableDataset):
    """A sample is an entire endpoint, so the loss compares competing airports."""

    def __init__(
        self,
        dates,
        feature_config=ALL_FEATURE_CONFIG,
        training=True,
        seed=42,
        excluded_airports=(),
    ):
        self.dates = tuple(dates)
        self.feature_config = feature_config
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.excluded_airports = frozenset(excluded_airports)
        self.cache_dir = CACHE_DIR

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        dates = list(self.dates)
        if self.training:
            rng.shuffle(dates)
        worker = get_worker_info()
        if worker is not None:
            dates = dates[worker.id :: worker.num_workers]
        for day in dates:
            path = self.cache_dir / f"v{FEATURE_VERSION}" / f"{day.isoformat()}.npz"
            with np.load(path, allow_pickle=False) as z:
                x, lengths, counts, target = (
                    z[k] for k in ("x", "lengths", "counts", "target")
                )
                truth = z["truth"]
                order = np.arange(len(x))
                if self.training:
                    rng.shuffle(order)
                for i in order:
                    if counts[i] == 0 or (self.training and target[i] < 0):
                        continue
                    if truth[i] in self.excluded_airports:
                        continue
                    values = x[i, : counts[i], : lengths[i], :].astype(np.float32)
                    values = values[:, :, self.feature_config.column_indices]
                    yield torch.from_numpy(values), int(target[i])


def collate_airport_candidates(samples):
    max_candidates = max(x.shape[0] for x, _ in samples)
    max_points = max(x.shape[1] for x, _ in samples)
    x = torch.zeros(len(samples), max_candidates, max_points, samples[0][0].shape[-1])
    mask = torch.zeros(x.shape[:-1], dtype=torch.bool)
    target = torch.tensor([y for _, y in samples], dtype=torch.long)
    for i, (values, _) in enumerate(samples):
        c, t, _ = values.shape
        x[i, :c, :t] = values
        mask[i, :c, :t] = True
    return x, mask, target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    for day in args.dates:
        prepare_day(day, args.rebuild)
