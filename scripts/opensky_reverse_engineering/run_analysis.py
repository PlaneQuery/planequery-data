"""Run the complete cached evidence analysis. Extract all 72 hours first."""

from datetime import date
from pathlib import Path

from airports.history import AirportHistory
from airports.history.schema import HistoryPaths

from extract import OUT
from analyze_endpoints import endpoint_audit
from airport_geometry import calibrate
from airport_candidates import run as airport_candidates
from segments import run as segmentation
from gap_rules import run as gaps
from callsigns import run as callsigns
from filters import run as filters
from build_report import build
from replica import export
from daily_replay import run as daily_replay
from evaluate_replica import run as evaluate_replica
import json


def main():
    files=list((OUT/'sample_hours').glob('*.parquet'))
    if len(files)!=72:
        raise ValueError(f'Expected 72 cached hours; found {len(files)}. Run extract.py first.')
    airport_path=OUT/'ourairports_2024-06-01.parquet'
    if not airport_path.exists():
        root=Path('/mnt/storage2/planequery-data/data/raw/ourairports')
        history=AirportHistory(HistoryPaths(root/'history',root/'repository'))
        history.as_of(date(2024,6,1)).write_parquet(airport_path)
    endpoint_audit()
    segmentation(OUT,focused=True)
    daily_replay()
    gaps()
    callsigns()
    filters()
    calibrate()
    airport_candidates()
    if (OUT/'ourairports_2022-01-01.parquet').exists():
        airport_candidates(catalog='ourairports_2022-01-01.parquet')
    metrics=json.loads((OUT/'segmentation_metrics.json').read_text())
    export(max(metrics,key=lambda x:x['exact_pairs'])['name'])
    evaluate_replica()
    build()


if __name__=='__main__':
    main()
