"""Compare the selected learning-curve checkpoint with the default checkpoint."""

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import torch

from flights.airport_nn.data import fingerprint
from flights.airport_nn.evaluate import files_for, score, write_report
from flights.airport_nn.model import load_model
from flights.airport_nn.source import DEFAULT_MODEL_PATH


def main():
    output = Path(__file__).resolve().parent
    root = Path('/mnt/storage2/planequery-data')
    cache = root / 'data/processed/flights_airport_nn/benchmark-2026-03-01-20261021'
    files = files_for(cache, [date(2026, 3, 1)])
    models = {
        'default': DEFAULT_MODEL_PATH,
        'days-20': root / 'data/models/flights_airport_nn/learning-curve-10-20/days-20/model.pt',
    }
    torch.set_num_threads(4)
    reports, frames, losses, provenance = {}, {}, {}, {}
    for name, path in models.items():
        print(f'Scoring {name} on {len(files)} cached parts', flush=True)
        model = load_model(path)
        data, loss = score(model, files)
        reports[name] = write_report(data, output / name)
        frames[name] = data.filter(pl.col('eligible'))
        losses[name] = loss
        provenance[name] = {
            'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'weighted_cross_entropy': loss,
        }
        print(json.dumps({'model': name, 'loss': loss, 'metrics': reports[name]}), flush=True)
        del model

    keys = ['date', 'icao', 'flight_id', 'endpoint', 'truth_id', 'airport_type']
    assert frames['default'].select(keys).equals(frames['days-20'].select(keys))
    paired = frames['default'].select(
        *keys, pl.col('nn_correct').alias('default_correct'),
        pl.col('nn_id').alias('default_id'),
    ).with_columns(
        frames['days-20']['nn_correct'].alias('days20_correct'),
        frames['days-20']['nn_id'].alias('days20_id'),
    )
    paired.filter(pl.col('default_id') != pl.col('days20_id')).write_parquet(
        output / 'prediction-disagreements.parquet'
    )
    rows = []
    for group in reports['default']:
        a, b = reports['default'][group], reports['days-20'][group]
        n = a['endpoints']
        assert n == b['endpoints']
        rows.append({
            'group': group, 'endpoints': n,
            'default_accuracy': a['nn_accuracy'], 'days20_accuracy': b['nn_accuracy'],
            'accuracy_delta_percentage_points': 100 * (b['nn_accuracy'] - a['nn_accuracy']),
            'default_errors': round(n * (1 - a['nn_accuracy'])),
            'days20_errors': round(n * (1 - b['nn_accuracy'])),
        })
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'date': '2026-03-01', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'cache': str(cache), 'models': provenance, 'comparison': rows,
        'days20_only_correct': paired.filter(~pl.col('default_correct') & pl.col('days20_correct')).height,
        'default_only_correct': paired.filter(pl.col('default_correct') & ~pl.col('days20_correct')).height,
        'cache_inputs': [fingerprint(p) for f in files for p in (f, f.with_suffix('.parquet'))],
        'scope': 'Matched ADSB.lol endpoints with eligible BTS/SFDPS labels; candidate misses retained in accuracy and excluded from weighted cross entropy.',
        'benchmark_caveat': 'This date was examined in earlier experiments and is not an untouched test.',
    }
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cache_inputs'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
