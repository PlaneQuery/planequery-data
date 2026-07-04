import polars as pl
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb


def model_training(X, y, groups, airport_idents, endpoint: str = "airport"):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X[train_idx],
        y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        eval_metric="binary_logloss",
    )

    val_scores = model.predict_proba(X[val_idx])[:, 1]

    val_results = pl.DataFrame({
        "flight_id": groups[val_idx],
        "airport_ident": airport_idents[val_idx],
        "label": y[val_idx],
        "score": val_scores,
    })

    top1 = (
        val_results
        .sort(["flight_id", "score"], descending=[False, True])
        .group_by("flight_id")
        .head(1)
    )

    print(f"Top-1 {endpoint} airport accuracy:", top1["label"].mean())

    return model
