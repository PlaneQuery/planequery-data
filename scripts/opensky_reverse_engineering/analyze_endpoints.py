"""Audit full-day flight metadata and match its endpoints to state vectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from extract import OUT

RADIUS = 6371000.8


def select_endpoint_states(probes: pl.DataFrame):
    """Resolve the usual +1 snapshot and exact-integer timestamp exceptions.

    Multiple updates can share an integer second; choosing the earliest fresh
    valid probe is explicit, and the raw probes remain available for review.
    """
    return probes.filter(
        (pl.col('lastposupdate').floor()==pl.col('endpoint_time'))
        & (pl.col('time')-pl.col('lastposupdate')<1)
        & pl.col('lat').is_finite() & pl.col('lon').is_finite()
        & pl.col('baroaltitude').is_finite()
    ).sort('flight_id','kind','time').unique(['flight_id','kind'],keep='first',maintain_order=True)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*RADIUS*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def quantiles(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return dict(zip(['min', 'p01', 'p10', 'p50', 'p90', 'p99', 'max'],
                    np.quantile(a, [0, .01, .1, .5, .9, .99, 1]).tolist())) if len(a) else {}


def metadata_audit(f: pl.DataFrame):
    duration = f['lastseen']-f['firstseen']
    ordered=f.drop_nulls('firstseen').sort('icao24','firstseen')
    adjacent=ordered.with_columns((pl.col('firstseen')-pl.col('lastseen').shift().over('icao24')).alias('gap'))
    result = {'flights': f.height, 'aircraft': f['icao24'].n_unique(),
              'null_firstseen': f['firstseen'].null_count(),
              'duration_seconds': quantiles(duration),
              'duration_below_900': int((duration < 900).sum()),
              'duration_at_most_600': int((duration <= 600).sum()),
              'consecutive_flights_min_gap':adjacent['gap'].min(),
              'consecutive_flights_gap_600':adjacent.filter(pl.col('gap')==600).height,
              'consecutive_flights_overlap':adjacent.filter(pl.col('gap')<0).height,
              'all_end_on_partition_day': bool(((f['lastseen']//86400)*86400 == f['day']).all()),
              'firstseen_before_day': int((f['firstseen'] < f['day']).sum()),
              'firstseen_null_lastseen_range': quantiles(f.filter(pl.col('firstseen').is_null())['lastseen'] - 1717200000)}
    for kind in ['departure', 'arrival']:
        prefix = 'est'+kind+'airport'
        rows = f.select(prefix, prefix+'horizdistance', prefix+'vertdistance', 'other'+kind+'airportcandidates', kind+'airportcandidatescount')
        result[kind] = {'null_airport': rows[prefix].null_count(),
                        'horizontal_m': quantiles(rows[prefix+'horizdistance']),
                        'vertical_m': quantiles(rows[prefix+'vertdistance'])}
        multi, nearest, euclid, count_mismatch, populated_ids, all_ids = 0, 0, 0, 0, 0, 0
        for airport, h, v, others, count in rows.iter_rows():
            if others is None:
                continue
            count_mismatch += len(others) != count
            populated_ids += sum(x[0] is not None for x in others)
            all_ids += len(others)
            if others:
                multi += 1
                nearest += all(h <= x[1] for x in others)
                euclid += all(h*h + v*v <= x[1]**2 + x[2]**2 for x in others)
        result[kind].update({'multiple_candidate_records': multi, 'chosen_minimum_horizontal': nearest,
                             'chosen_minimum_3d': euclid, 'candidate_count_length_mismatch': count_mismatch,
                             'other_candidate_codes_present': populated_ids, 'other_candidates_total': all_ids})
    return result


def endpoint_audit(out: Path = OUT):
    f = pl.read_parquet(out / 'reference_flights.parquet')
    report = {'metadata': metadata_audit(f)}
    files = sorted((out/'endpoint_hours').glob('*.parquet'))
    report['extracted_hours'] = len(files)
    if not files:
        (out/'endpoint_audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return
    e = pl.read_parquet(files).with_columns(
        (pl.col('time')-pl.col('lastposupdate')).alias('position_age'),
        (pl.col('lastposupdate').floor()==pl.col('endpoint_time')).alias('lpu_matches'),
    )
    report['probe_rows'] = e.height
    report['endpoint_probe_summary'] = e.group_by('kind', 'offset').agg(
        pl.len().alias('rows'), pl.col('flight_id').n_unique().alias('flights'),
        pl.col('onground').mean().alias('ground_fraction'),
        (pl.col('position_age')<1).mean().alias('fresh_fraction'),
        pl.col('lpu_matches').mean().alias('floor_lastposupdate_matches'),
        pl.col('baroaltitude').is_null().mean().alias('null_altitude_fraction'),
        pl.col('position_age').median().alias('median_position_age'),
    ).sort('kind','offset').to_dicts()
    # Preserve all probes for alternate timestamp/position hypotheses.
    e.write_parquet(out/'endpoint_probes.parquet')
    selected=select_endpoint_states(e)
    selected.write_parquet(out/'selected_endpoint_states.parquet')
    report['selected_endpoint_coverage']=selected.group_by('kind').agg(pl.len().alias('endpoints')).to_dicts()
    report['endpoint_lpu_coverage'] = e.filter(pl.col('lpu_matches') & ~pl.col('onground')).group_by('kind').agg(
        pl.col('flight_id').n_unique()).to_dicts()
    # Start with exact snapshot timestamp; later analyses compare +1 snapshots.
    airports = pl.read_parquet(out/'ourairports_2024-06-01.parquet')
    for offset in [0, 1]:
        results = []
        for kind in ['departure','arrival']:
            prefix = 'est'+kind+'airport'
            x = e.filter((pl.col('kind')==kind)&(pl.col('offset')==offset)).join(
                f.select('flight_id', prefix, prefix+'horizdistance', prefix+'vertdistance'), on='flight_id')
            x = x.join(airports.select('ident','latitude_deg','longitude_deg','elevation_ft','type'), left_on=prefix, right_on='ident', how='left')
            d = haversine(x['lat'].to_numpy(),x['lon'].to_numpy(),x['latitude_deg'].to_numpy(),x['longitude_deg'].to_numpy())
            v = np.abs(x['baroaltitude'].to_numpy()-x['elevation_ft'].to_numpy()*.3048)
            x = x.with_columns(pl.Series('computed_horizontal', d),pl.Series('computed_vertical', v),
                pl.Series('horizontal_error', d-x[prefix+'horizdistance'].to_numpy()),
                pl.Series('vertical_error', v-x[prefix+'vertdistance'].to_numpy()))
            x.write_parquet(out/f'{kind}_airport_offset{offset}.parquet')
            results.append({'kind':kind,'rows':x.height,'matched_airports':int(x['latitude_deg'].is_not_null().sum()),
                            'assigned_airport_above_2500m_baro':x.filter(pl.col(prefix).is_not_null()&(pl.col('baroaltitude')>2500)).height,
                            'horizontal_error':quantiles(x['horizontal_error']), 'vertical_error':quantiles(x['vertical_error']),
                            'horizontal_within_2m':int((x['horizontal_error'].abs()<2).sum()),
                            'vertical_within_2m':int((x['vertical_error'].abs()<2).sum())})
        report[f'airport_snapshot_offset_{offset}'] = results
    (out/'endpoint_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    endpoint_audit()
