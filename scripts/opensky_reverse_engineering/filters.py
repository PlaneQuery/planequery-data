"""Audit the sparse-bin condition without using it to select endpoints."""

import json

import polars as pl

from extract import OUT


def run(out=OUT):
    aircraft=json.loads((out/'manifest.json').read_text())['sample_aircraft']
    f=pl.read_parquet(out/'reference_flights.parquet').filter(pl.col('icao24').is_in(aircraft))
    bins=(pl.scan_parquet(sorted((out/'sample_hours').glob('*.parquet')))
        .filter(pl.col('lat').is_finite()&pl.col('lon').is_finite()&pl.col('baroaltitude').is_finite()
                &((pl.col('time')-pl.col('lastposupdate'))<1))
        .with_columns((pl.col('lastposupdate')/30).floor().cast(pl.Int64).alias('bin'))
        .group_by('icao24','bin').len().collect(engine='streaming'))
    endpoints=pl.concat([f.select('flight_id','icao24',pl.col(c).alias('endpoint')).with_columns(pl.lit(k).alias('kind'))
        for c,k in [('firstseen','departure'),('lastseen','arrival')]]).drop_nulls('endpoint')
    endpoints=endpoints.with_columns((pl.col('endpoint')//30).alias('bin')).join(bins,on=['icao24','bin'],how='left')
    results=endpoints.group_by('kind').agg(pl.len().alias('endpoints'),pl.col('len').min().alias('min_bin_count'),
        (pl.col('len')<=3).sum().alias('in_sparse_bins'),(pl.col('len')==4).sum().alias('in_four_point_bins')).to_dicts()
    (out/'state_filter_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    endpoints.write_parquet(out/'endpoint_density_counts.parquet')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    run()
