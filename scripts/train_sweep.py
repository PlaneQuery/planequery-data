from pathlib import Path
from flights.airport_nn.train import TRAIN_DAYS, VALIDATION_DAYS, train

for days in [1, 2, 5, 10, 15, 20]:
    output = Path("outputs/airport-nn-learning-curve") / f"days-{days:02d}"
    print(f"\n=== Training on {days} days → {output} ===", flush=True)
    train(
        output=output,
        train_days=TRAIN_DAYS[:days],
        validation_days=VALIDATION_DAYS,
        epochs=10,
        seed=42,
    )