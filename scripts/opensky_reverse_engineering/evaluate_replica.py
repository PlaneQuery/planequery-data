"""Compare the independently exported sample to references, without changing it."""

import json

import polars as pl

from extract import OUT
from segments import DAY, evaluate


def run(out=OUT,day=DAY):
    aircraft=json.loads((out/'manifest.json').read_text())['sample_aircraft']
    reference=pl.read_parquet(out/'reference_flights.parquet').filter(pl.col('icao24').is_in(aircraft))
    predicted=pl.read_parquet(out/'reconstructed_sample_flights.parquet')
    keys=['icao24','firstseen','lastseen']
    assert predicted.select(keys).n_unique()==predicted.height
    assert predicted['day'].n_unique()==1
    assert (predicted['lastseen']-predicted['firstseen']).min()>600
    _,best_pairs=evaluate(predicted,reference,day=day)
    residuals=best_pairs.with_columns(
        (pl.col('firstseen_pred')-pl.col('firstseen')).alias('start_delta_seconds'),
        (pl.col('lastseen_pred')-pl.col('lastseen')).alias('end_delta_seconds')).filter(
            (pl.col('start_delta_seconds')!=0)|(pl.col('end_delta_seconds')!=0)|pl.col('firstseen_pred').is_null())
    residuals.write_parquet(out/'best_model_residuals.parquet')
    residuals.select('icao24','firstseen','lastseen','start_delta_seconds','end_delta_seconds').sort('firstseen').write_csv(out/'best_model_residuals.csv')
    paired=reference.join(predicted,on=keys,suffix='_pred')
    fields=['callsign','estdepartureairport','estarrivalairport',
            'estdepartureairporthorizdistance','estdepartureairportvertdistance',
            'estarrivalairporthorizdistance','estarrivalairportvertdistance',
            'departureairportcandidatescount','arrivalairportcandidatescount']
    matches={c:int(paired.select(pl.col(c).eq_missing(pl.col(c+'_pred')).sum()).item()) for c in fields}
    primary=['callsign','estdepartureairport','estarrivalairport']
    report={'reference_rows':reference.height,'reference_known_starts':reference['firstseen'].count(),
            'predicted_rows':predicted.height,'exact_boundary_pairs':paired.height,
            'field_matches_conditional_on_exact_boundaries':matches,
            'boundary_callsign_both_airports_matches':paired.filter(pl.all_horizontal(
                [pl.col(c).eq_missing(pl.col(c+'_pred')) for c in primary])).height,
            'all_nine_nonarray_fields_matches':paired.filter(pl.all_horizontal(
                [pl.col(c).eq_missing(pl.col(c+'_pred')) for c in fields])).height,
            'note':('Exploratory agreement on the same aircraft sample used to select hypotheses; not held-out validation.'
                    if day==DAY else 'Temporal validation of the fixed 2024 rules and catalog; sampled from aircraft present in the target flight table.')}
    (out/'replica_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()
