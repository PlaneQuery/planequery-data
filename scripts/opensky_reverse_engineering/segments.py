"""Independent flight-segmentation hypotheses and reproducible comparisons.

Input is state vectors only. Reference flight boundaries are consumed solely
by evaluate(), after predictions have been computed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import median_filter

from analyze_endpoints import haversine
from extract import OUT

DAY = 1717200000


@dataclass(frozen=True)
class Rules:
    name: str
    freshness: float = 1
    clock: str = 'snapshot'
    sparse: bool = False
    median: bool = False
    gap: float = 600
    split: str = 'gap'
    min_duration: float = 600
    min_points: int = 0
    ground: bool = False
    trim_start: int = 0
    trim_end: int = 0
    trim_daily: bool = False
    density_first: bool = False
    median_scope: str = 'all'
    trim_before_filter: bool = False


def clean(states: pl.DataFrame, rules: Rules):
    columns = ['time','lastposupdate','lat','lon','baroaltitude','velocity','heading','vertrate','onground','callsign']
    x = {c: states[c].to_numpy() for c in columns}
    mask = np.isfinite(x['lat']) & np.isfinite(x['lon']) & np.isfinite(x['baroaltitude']) & np.isfinite(x['lastposupdate'])
    if not rules.ground:
        mask &= x['onground'] == False  # noqa: E712
    if np.isfinite(rules.freshness):
        mask &= (x['time']-x['lastposupdate']) < rules.freshness
    x = {c: a[mask] for c,a in x.items()}
    if not len(x['time']):
        return x
    clock = x['time'] if rules.clock == 'snapshot' else x['lastposupdate']
    order = np.argsort(clock, kind='stable')
    x = {c: a[order] for c,a in x.items()}
    x['clock'] = clock[order]
    keep = np.ones(len(order), dtype=bool)
    if rules.sparse:
        _, inverse, counts = np.unique(np.floor(x['clock']/30), return_inverse=True, return_counts=True)
        keep &= counts[inverse] > 3
        if rules.density_first:
            x = {c:a[keep] for c,a in x.items()}
            keep = np.ones(len(x['clock']),dtype=bool)
    if rules.median and len(keep) >= 7:
        for c, threshold in [('baroaltitude', 50), ('lat', .5), ('lon', .5)]:
            med = median_filter(x[c], size=7, mode='nearest')
            # R runmed(..., endrule='constant') fills the first/last 3 values.
            med[:3] = np.median(x[c][:7])
            med[-3:] = np.median(x[c][-7:])
            keep &= np.abs(x[c]-med) <= threshold
    if rules.trim_before_filter and (rules.trim_start or rules.trim_end):
        cuts=np.r_[0,np.flatnonzero(np.diff(np.floor(x['clock']/86400)))+1,len(keep)] if rules.trim_daily else [0,len(keep)]
        for start,end in zip(cuts[:-1],cuts[1:]):
            keep[start:min(start+rules.trim_start,end)]=False
            if rules.trim_end:
                keep[max(start,end-rules.trim_end):end]=False
    x = {c: a[keep] for c,a in x.items()}
    if not rules.trim_before_filter and (rules.trim_start or rules.trim_end):
        keep = np.ones(len(x['clock']),dtype=bool)
        cuts = np.r_[0,np.flatnonzero(np.diff(np.floor(x['clock']/86400)))+1,len(keep)] if rules.trim_daily else [0,len(keep)]
        for start,end in zip(cuts[:-1],cuts[1:]):
            keep[start:min(start+rules.trim_start,end)] = False
            if rules.trim_end:
                keep[max(start,end-rules.trim_end):end] = False
        x = {c:a[keep] for c,a in x.items()}
    return x


def gap_features(x):
    dt = np.diff(x['clock'])
    distance = haversine(x['lat'][:-1], x['lon'][:-1], x['lat'][1:], x['lon'][1:])
    speed = np.divide(distance, dt, out=np.full(len(dt), np.nan), where=dt>0)
    return dt, distance, speed


def reconstruct(aircraft: str, states: pl.DataFrame, rules: Rules):
    x = clean(states, rules)
    if len(x['time']) < 2:
        return []
    dt, distance, speed = gap_features(x)
    split = dt > rules.gap
    if rules.split == 'median_quarter':
        if rules.median_scope == 'day':
            days=np.floor(x['clock'][1:]/86400)
            median=np.zeros(len(speed))
            for day in np.unique(days):
                median[days==day]=np.nanmedian(speed[days==day])
            split &= speed < median/4
        else:
            split &= speed < np.nanmedian(speed)/4
    elif rules.split == 'last_quarter':
        split &= speed < x['velocity'][:-1]/4
    elif rules.split == 'last_half':
        split &= speed < x['velocity'][:-1]/2
    elif rules.split == 'reachable':
        # A speed budget with the paper's roughly 0.5 degree spatial allowance.
        split &= distance < x['velocity'][:-1]*dt-55597.5
    elif rules.split == 'extrapolate':
        # Local constant velocity extrapolation (explicit hypothesis).
        lat = x['lat'][:-1] + np.cos(np.radians(x['heading'][:-1]))*x['velocity'][:-1]*dt/111195
        lon = x['lon'][:-1] + np.sin(np.radians(x['heading'][:-1]))*x['velocity'][:-1]*dt/(111195*np.cos(np.radians(x['lat'][:-1])))
        split &= (np.abs(lat-x['lat'][1:])>.5) | (np.abs(lon-x['lon'][1:])>.5)
    cuts = np.r_[0, np.flatnonzero(split)+1, len(x['clock'])]
    output = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        first, last = int(x['clock'][start]), int(x['clock'][end-1])
        if last-first <= rules.min_duration or end-start < rules.min_points:
            continue
        callsigns = x['callsign'][start:end]
        callsigns = callsigns[callsigns != None]  # noqa: E711
        if len(callsigns):
            values, counts = np.unique(callsigns, return_counts=True)
            mode = str(values[np.argmax(counts)])
            first_callsign, last_callsign = str(callsigns[0]), str(callsigns[-1])
        else:
            mode, first_callsign, last_callsign = None, None, None
        output.append({'icao24':aircraft, 'firstseen':first, 'lastseen':last, 'points':end-start,
                       'callsign_mode':mode, 'callsign_first':first_callsign, 'callsign_last':last_callsign,
                       'first_lat':x['lat'][start], 'first_lon':x['lon'][start],
                       'last_lat':x['lat'][end-1], 'last_lon':x['lon'][end-1],
                       'first_baroaltitude':x['baroaltitude'][start],
                       'last_baroaltitude':x['baroaltitude'][end-1]})
    return output


def evaluate(predictions: pl.DataFrame, reference: pl.DataFrame):
    p = predictions.filter(pl.col('lastseen').is_between(DAY, DAY+86399))
    r = reference.drop_nulls('firstseen')
    result = {'reference':r.height, 'predicted':p.height}
    result['exact_pairs'] = p.join(r, on=['icao24','firstseen','lastseen'], how='semi').height
    result['exact_starts'] = r.join(p, on=['icao24','firstseen'], how='semi').height
    result['exact_ends'] = r.join(p, on=['icao24','lastseen'], how='semi').height
    pairs = r.join(p, on='icao24', how='left', suffix='_pred').with_columns(
        (pl.col('firstseen')-pl.col('firstseen_pred')).abs().alias('start_error'),
        (pl.col('lastseen')-pl.col('lastseen_pred')).abs().alias('end_error'),
    )
    for tolerance in [1, 5, 30, 60, 120]:
        good = pairs.filter((pl.col('start_error')<=tolerance)&(pl.col('end_error')<=tolerance))
        matched = good['flight_id'].n_unique()
        result[f'pairs_within_{tolerance}s'] = matched
    best = pairs.with_columns(pl.max_horizontal('start_error','end_error').alias('max_error')).sort('max_error',nulls_last=True).unique('flight_id', keep='first')
    return result, best


def default_hypotheses():
    rules = []
    for fresh in [1, 2, 10, float('inf')]:
        for clock in ['snapshot','position']:
            for filters in ['none','sparse','median','both']:
                rules.append(Rules(f'gap_f{fresh}_{clock}_{filters}', freshness=fresh, clock=clock,
                                   sparse=filters in ['sparse','both'], median=filters in ['median','both']))
    for split in ['median_quarter','last_quarter','last_half','reachable','extrapolate']:
        for filters in ['none','both']:
            for clock in ['snapshot','position']:
                rules.append(Rules(f'{split}_{clock}_{filters}', clock=clock,
                                   sparse=filters=='both', median=filters=='both', split=split))
    for ground in [False,True]:
        for start,end in [(0,0),(3,3),(3,6),(3,0),(0,6)]:
            for split in ['gap','median_quarter','last_half']:
                rules.append(Rules(f'trim_{start}_{end}_ground{ground}_{split}', clock='position', sparse=True,
                                   median=True, ground=ground, trim_start=start, trim_end=end, split=split))
    return rules


def focused_hypotheses():
    rules=[]
    for density_first in [False,True]:
        for trim_daily in [False,True]:
            for end in [0,3,6]:
                rules.append(Rules(f'focused_density{density_first}_daily{trim_daily}_end{end}', clock='position',
                    sparse=True,median=True,ground=True,split='median_quarter',trim_start=3,trim_end=end,
                    density_first=density_first,trim_daily=trim_daily,median_scope='day' if trim_daily else 'all'))
    rules.append(Rules('baseline_gap',clock='position',sparse=True,median=True,ground=True))
    rules.append(Rules('baseline_R_inspired',clock='position',sparse=True,median=True,split='median_quarter'))
    rules.append(Rules('baseline_paper_literal',clock='position',split='extrapolate',min_duration=900))
    return rules


def run(out: Path, limit_aircraft: int | None = None, partial: bool = False, focused: bool = False):
    manifest = json.loads((out/'manifest.json').read_text())
    aircraft = manifest['sample_aircraft'][:limit_aircraft]
    reference = pl.read_parquet(out/'reference_flights.parquet').filter(pl.col('icao24').is_in(aircraft))
    files = sorted((out/'sample_hours').glob('*.parquet'))
    if partial:
        files = [p for p in files if p.name.startswith('2024-06-01')]
    if len(files) != 72 and not partial:
        raise ValueError(f'Need all 72 hours to avoid artificial boundaries, found {len(files)}')
    if partial:
        lower = DAY + int(files[0].stem[-2:])*3600 + 600
        upper = DAY + (int(files[-1].stem[-2:])+1)*3600 - 600
        reference = reference.filter((pl.col('firstseen')>=lower)&(pl.col('lastseen')<upper))
    rules = focused_hypotheses() if focused else default_hypotheses()
    records = {rule.name:[] for rule in rules}
    # Bounded groups avoid materializing the complete 3-day sample in RAM.
    for index in range(0,len(aircraft),32):
        acs = aircraft[index:index+32]
        data = pl.scan_parquet(files).filter(pl.col('icao24').is_in(acs)).collect().sort('icao24','time')
        for key, group in data.partition_by('icao24', as_dict=True).items():
            for rule in rules:
                records[rule.name].extend(reconstruct(key[0], group, rule))
        print(f'Analyzed {min(index+32,len(aircraft))}/{len(aircraft)} aircraft',flush=True)
    metrics=[]
    (out/'hypotheses').mkdir(exist_ok=True)
    for rule in rules:
        predictions=pl.DataFrame(records[rule.name])
        if partial:
            predictions = predictions.filter((pl.col('firstseen')>=lower)&(pl.col('lastseen')<upper))
        summary, pairs=evaluate(predictions,reference)
        summary['name']=rule.name
        summary['rules']=asdict(rule)
        metrics.append(summary)
        predictions.write_parquet(out/'hypotheses'/f'{rule.name}.parquet')
        pairs.write_parquet(out/'hypotheses'/f'{rule.name}_pairs.parquet')
    metrics.sort(key=lambda x:x['pairs_within_5s'],reverse=True)
    (out/'segmentation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'segmentation_run.json').write_text(json.dumps({'partial':partial,'focused':focused,'sample_aircraft':len(aircraft),
        'input_hours':[p.stem for p in files],'lower_reference_bound':lower if partial else DAY,
        'upper_reference_bound':upper if partial else DAY+86400},indent=2)+'\n')
    print(json.dumps(metrics[:10],indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--limit-aircraft',type=int)
    parser.add_argument('--partial',action='store_true',help='Explore currently extracted June 1 hours; censor both ends by 600 s')
    parser.add_argument('--focused',action='store_true')
    args=parser.parse_args()
    run(args.out,args.limit_aircraft,args.partial,args.focused)
