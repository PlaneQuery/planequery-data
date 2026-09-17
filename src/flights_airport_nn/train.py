"""Reproducible training; checkpoint selection uses validation endpoints only."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flights_airport_nn.benchmark import evaluate_dates, fit_temperature, summarize
from flights_airport_nn.dataset import (
    AirportCandidateDataset,
    cache_path,
    collate_airport_candidates,
    prepare_day,
)
from flights_airport_nn.features.features import (
    ALL_FEATURE_CONFIG,
    FEATURE_VERSION,
    FeatureName,
)
from flights_airport_nn.model import AirportGRU
from flights_airport_nn.utils import device, model_folder_path


def run_train(
    train_dates,
    feature_config=ALL_FEATURE_CONFIG,
    epochs=20,
    *,
    validation_dates=None,
    output_dir=None,
    hidden_size=64,
    dropout=0.15,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    patience=4,
    seed=42,
    prepare=True,
    excluded_airports=(),
):
    if not train_dates or epochs < 1:
        raise ValueError("Training requires dates and positive epochs")
    if not validation_dates:
        raise ValueError(
            "Separate validation_dates are required for checkpoint selection"
        )
    if set(train_dates) & set(validation_dates):
        raise ValueError("Training and validation dates must be disjoint")
    if batch_size < 1 or patience < 1 or hidden_size < 1 or learning_rate <= 0:
        raise ValueError(
            "Batch size, patience, hidden size and learning rate must be positive"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_dir = Path(output_dir or model_folder_path(len(train_dates), epochs=epochs))
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "model.pt").exists():
        raise FileExistsError(
            f"Existing model: {output_dir / 'model.pt'}; choose a new run directory"
        )
    if prepare:
        for day in (*train_dates, *validation_dates):
            prepare_day(day)
    distance_index = (
        feature_config.enabled.index(FeatureName.DISTANCE)
        if FeatureName.DISTANCE in feature_config.enabled
        else None
    )
    model = AirportGRU(
        feature_config.input_size, hidden_size, dropout, distance_index=distance_index
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    dataset = AirportCandidateDataset(
        train_dates, feature_config, seed=seed, excluded_airports=excluded_airports
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_airport_candidates,
        num_workers=1,
        multiprocessing_context="spawn",
        pin_memory=device.type == "cuda",
    )
    config = {
        "train_dates": [d.isoformat() for d in train_dates],
        "validation_dates": [d.isoformat() for d in validation_dates],
        "epochs": epochs,
        "seed": seed,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "feature_names": [f.value for f in feature_config.enabled],
        "feature_version": FEATURE_VERSION,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "excluded_airports": sorted(excluded_airports),
    }
    config["cache_signatures"] = {}
    for day in (*train_dates, *validation_dates):
        with np.load(cache_path(day), allow_pickle=False) as cache:
            config["cache_signatures"][day.isoformat()] = (
                str(cache["signature"]) if "signature" in cache else None
            )
    code_files = sorted(Path(__file__).parent.rglob("*.py"))
    config["code_sha256"] = hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in code_files)
    ).hexdigest()
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history = []
    best = float("inf")
    stale = 0
    for epoch in range(epochs):
        model.train()
        dataset.epoch = epoch
        total_loss = 0.0
        count = 0
        for x, mask, target in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Apply a shared coverage cut to competing airports. Labels never affect augmentation.
            if (
                random.random() < 0.15
                and x.shape[2] > 8
                and mask.sum(-1)[mask.any(-1)].min() > 8
            ):
                cut = random.randint(1, 6)
                x = x[:, :, cut:]
                mask = mask[:, :, cut:]
                if FeatureName.ELAPSED in feature_config.enabled:
                    k = feature_config.enabled.index(FeatureName.ELAPSED)
                    removed_time = x[:, :, :1, k].clone()
                    x[:, :, :, k] -= removed_time
                    if FeatureName.COVERAGE_GAP in feature_config.enabled:
                        g = feature_config.enabled.index(FeatureName.COVERAGE_GAP)
                        x[:, :, :, g] += removed_time
                for feature in (
                    FeatureName.TIME_DELTA_S,
                    FeatureName.RADIAL_SPEED_KM_S,
                ):
                    if feature in feature_config.enabled:
                        x[:, :, 0, feature_config.enabled.index(feature)] = 0
            if FeatureName.ALTITUDE in feature_config.enabled:
                k = feature_config.enabled.index(FeatureName.ALTITUDE)
                x[:, :, :, k] += torch.randn(x.shape[0], 1, 1, device=device) * 0.0075
            optimizer.zero_grad(set_to_none=True)
            logits = model.rank(x, mask)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            count += len(target)
        if not count:
            raise ValueError("No training endpoints have a retrieved label")
        frame, val_logits, val_targets = evaluate_dates(
            model, validation_dates, feature_config, excluded_airports=excluded_airports
        )
        if len(val_targets) == 0:
            raise ValueError("No validation endpoints have a retrieved label")
        val_loss = float(
            torch.nn.functional.cross_entropy(
                torch.tensor(val_logits), torch.tensor(val_targets)
            )
        )
        metrics = summarize(frame)
        scheduler.step(val_loss)
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / count,
            "validation_loss": val_loss,
            "validation_accuracy": metrics["overall"]["accuracy"],
            "nearest_accuracy": metrics["overall"]["nearest_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        meaningful_improvement = val_loss < best - 1e-5
        if val_loss < best:
            best = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "feature_names": config["feature_names"],
                "feature_version": FEATURE_VERSION,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "distance_index": distance_index,
                "epoch": epoch + 1,
                "validation_loss": best,
                "train_dates": config["train_dates"],
                "validation_dates": config["validation_dates"],
                "temperature": 1.0,
                "excluded_airports": sorted(excluded_airports),
            }
            torch.save(checkpoint, output_dir / "model.pt")
        if meaningful_improvement:
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    checkpoint = torch.load(
        output_dir / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    _, logits, targets = evaluate_dates(
        model, validation_dates, feature_config, excluded_airports=excluded_airports
    )
    model.temperature = fit_temperature(logits, targets)
    checkpoint["temperature"] = model.temperature
    torch.save(checkpoint, output_dir / "model.pt")
    validation, _, _ = evaluate_dates(
        model,
        validation_dates,
        feature_config,
        temperature=model.temperature,
        excluded_airports=excluded_airports,
    )
    validation.write_parquet(output_dir / "validation_predictions.parquet")
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(summarize(validation), indent=2) + "\n"
    )
    model.eval()
    model.output_dir = output_dir
    model.training_dates = tuple(train_dates)
    model.validation_dates = tuple(validation_dates)
    return model


if __name__ == "__main__":
    from flights_airport_nn.experiment import main

    main()
