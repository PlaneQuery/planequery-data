import argparse
from pathlib import Path

from flights.airport_nn.data import CACHE
from flights.airport_nn.train import TRAIN_DAYS, VALIDATION_DAYS, train


def main():
    parser = argparse.ArgumentParser(description="Train an airport-model learning curve.")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/airport-nn-learning-curve")
    )
    args = parser.parse_args()
    print(f"Using training cache: {args.cache}", flush=True)
    for days in [1, 2, 5, 10, 15, 20]:
        output = args.output / f"days-{days:02d}"
        print(f"\n=== Training on {days} days → {output} ===", flush=True)
        train(
            cache=args.cache,
            output=output,
            train_days=TRAIN_DAYS[:days],
            validation_days=VALIDATION_DAYS,
            epochs=10,
            seed=42,
        )


if __name__ == "__main__":
    main()
