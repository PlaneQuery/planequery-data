"""Build a standalone evidence report and scientific figures from cached results."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from extract import OUT

ROOT=Path(__file__).resolve().parents[2]


def table(rows,columns):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(label)}</th>' for key,label in columns)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(row.get(key,"")))}</td>' for key,label in columns)+'</tr>' for row in rows)+'</tbody></table>'


def build(out=OUT):
    audit=json.loads((out/'endpoint_audit.json').read_text())
    metadata=audit['metadata']
    metrics=json.loads((out/'segmentation_metrics.json').read_text())
    run=json.loads((out/'segmentation_run.json').read_text())
    gap_metrics=json.loads((out/'gap_metrics.json').read_text())
    candidates=[dict(x,catalog='2024-06-01') for x in json.loads((out/'airport_candidate_metrics.json').read_text())]
    older=out/'airport_candidate_metrics_ourairports_2022-01-01.json'
    if older.exists():
        candidates += [dict(x,catalog='2022-01-01') for x in json.loads(older.read_text())]
    geometry=json.loads((out/'airport_geometry.json').read_text())
    best=max(metrics,key=lambda x:x['exact_pairs'])
    replica=json.loads((out/'replica_metrics.json').read_text()) if (out/'replica_metrics.json').exists() else None
    replica_text=(f"The exported replica contains {replica['predicted_rows']:,} flights. {replica['boundary_callsign_both_airports_matches']:,} match both timestamps, callsign, departure and arrival airport together; {replica['all_nine_nonarray_fields_matches']:,} additionally match all four integer distances and both candidate counts. Alternative-candidate ordering is deliberately excluded from this comparison because it has not been recovered." if replica else '')
    callsign=json.loads((out/'callsign_metrics.json').read_text()) if (out/'callsign_metrics.json').exists() else None
    callsign_text=(f"Callsign selection was checked inside the supplied reference intervals for {callsign['flights']:,} sample flights: first non-null {callsign['first_matches']:,}, last non-null {callsign['last_matches']:,}, modal value {callsign['mode_matches']:,} matches. {callsign['multiple_callsigns']} intervals contain multiple distinct callsigns. The approximate exporter uses the mode and preserves padding." if callsign else "Callsign selection has not yet been evaluated in this run.")
    g=pl.read_parquet(out/'labeled_gaps.parquet')
    long=g.filter(pl.col('gap')>600)
    error=long.filter(pl.col('split')!=(pl.col('gap_speed')<pl.col('day_median_speed')/4)).height
    gap_rule=next(x for x in gap_metrics if x['name']=='gap_600_day_median_speed_x0.25')
    f=pl.read_parquet(out/'reference_flights.parquet')
    geo=pl.read_parquet(out/'airport_geometry_holdout.parquet')
    reports=ROOT/'reports'
    reports.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    durations=(f['lastseen']-f['firstseen']).drop_nulls().to_numpy()
    axes[0,0].hist(durations[durations<1800]/60,bins=np.arange(0,31,.25),color='#287891')
    axes[0,0].axvline(10,color='#bd422e',label='Observed lower cutoff: 10 min')
    axes[0,0].axvline(15,color='#aa8833',ls='--',label='Paper: 15 min')
    axes[0,0].set(xlabel='Reported duration (minutes)',ylabel='Flights',title='A. Full flight partition: short durations')
    axes[0,0].legend(fontsize=8)
    for kind,color,limit in [('departure','#287891',15),('arrival','#bd422e',30)]:
        d=f['est'+kind+'airporthorizdistance'].drop_nulls().to_numpy()/1000
        axes[0,1].hist(d,bins=np.arange(0,31,.25),histtype='step',density=True,label=kind,color=color)
        axes[0,1].axvline(limit,color=color,ls='--')
    axes[0,1].set(xlabel='Stored horizontal airport distance (km)',ylabel='Density',title='B. Distinct airport search limits')
    axes[0,1].legend(fontsize=8)
    for split,color,label in [(False,'#287891','Same reference flight'),(True,'#bd422e','Different reference flights')]:
        x=g.filter(pl.col('split')==split)
        axes[1,0].scatter(x['gap'],x['gap_speed']/x['day_median_speed'],s=8,alpha=.45,color=color,label=label)
    axes[1,0].axvline(600,color='black',lw=1);axes[1,0].axhline(.25,color='black',lw=1)
    axes[1,0].set(xscale='log',yscale='log',xlabel='Position gap (seconds)',ylabel='Gap speed / aircraft-day median speed',title='C. Split when gap > 600 s and ratio < 0.25')
    axes[1,0].legend(fontsize=8)
    axes[1,1].hist(geo['horizontal_error_calibrated'].to_numpy(),bins=np.linspace(-2,3,100),color='#287891')
    axes[1,1].axvline(0,color='black',lw=1);axes[1,1].axvline(1,color='black',lw=1)
    axes[1,1].set(xlabel='Calculated distance minus stored integer (m)',ylabel='Held-out observations',title='D. Fixed airport coordinates and integer truncation')
    png=reports/'opensky_reverse_engineering_2024-06-01.png'
    fig.savefig(png,dpi=170)
    plt.close(fig)
    graphic=base64.b64encode(png.read_bytes()).decode()
    comparison=[
        {'item':'Minimum stored flight duration','paper':'15 minutes','observed':f'{metadata["duration_seconds"]["min"]:.0f} seconds; {metadata["duration_below_900"]:,} records below 15 minutes'},
        {'item':'Gap split','paper':'10 minutes with velocity/extrapolation check','observed':f'Δt > 600 s AND distance/Δt < median observed speed / 4; {len(long)-error:,}/{len(long):,} labeled long-gap decisions'},
        {'item':'Departure airport gates','paper':'10 km; described as reverse of arrival','observed':'Horizontal < 15,000 m; absolute vertical < 2,000 m'},
        {'item':'Arrival airport gates','paper':'10 km and 2,500 m','observed':'Horizontal < 30,000 m; absolute vertical < 2,500 m'},
        {'item':'Airport ranking','paper':'Closest to extrapolated ground trajectory','observed':'Chosen horizontal distance is minimal in every supplied candidate list'},
        {'item':'Airport geometry','paper':'Ground extrapolation','observed':'Direct endpoint-to-fixed-airport haversine distance explains held-out fields to metre precision'},
        {'item':'Timestamps','paper':'First/last contact','observed':'Integer floor of position-update timestamp; distinguish it from snapshot time'},
    ]
    model_rows=[]
    for x in sorted(metrics,key=lambda x:x['exact_pairs'],reverse=True):
        model_rows.append({'model':x['name'],'predicted':x['predicted'],'reference':x['reference'],'exact':x['exact_pairs'],
            'recall':f'{100*x["exact_pairs"]/x["reference"]:.2f}%','within30':x['pairs_within_30s']})
    airport_rows=[{'catalog':x['catalog'],'kind':x['kind'],'observations':x['observations'],
        'matches':x['airport_matches_including_null'],
        'agreement':f"{100*x['airport_matches_including_null']/x['observations']:.2f}%",
        'counts':x['candidate_counts_match']} for x in candidates if x['policy']=='fixed_wing_gps']
    api_rows=[r for r in audit['endpoint_probe_summary'] if r['offset']==1]
    residual_path=out/'best_model_residuals.csv'
    residuals=pl.read_csv(residual_path).to_dicts() if residual_path.exists() else []
    residual_table=table(residuals,[('icao24','Aircraft'),('firstseen','Reference start (epoch s)'),
        ('lastseen','Reference end (epoch s)'),('start_delta_seconds','Predicted − reference start (s)'),
        ('end_delta_seconds','Predicted − reference end (s)')]) if residuals else ''
    partial='<p class="caution">This run is censored to the currently extracted June 1 hours, not a full-day validation.</p>' if run['partial'] else ''
    body=f'''<h1>Reverse engineering OpenSky flights — 2024-06-01</h1>
<p class="lead">The observed table is much closer to the supplied legacy R algorithm than to a literal implementation of the 2021 description. The core split rule, timestamp convention, duration cutoff and airport distance rules are supported directly by the data. The supplied artifacts are an inferred reconstruction, not a recovered copy of OpenSky’s production source.</p>
<p>Study inputs: <b>{metadata['flights']:,} flights</b> from <b>{metadata['aircraft']:,} aircraft</b>; 72 hourly raw files for May 31–June 2, containing 2,108,598,395 state snapshots (52.50 GB compressed). Full trajectories are evaluated for a deterministic SHA-256 sample of 512 aircraft drawn from the target flight table. This positive-aircraft sample does not measure false positives among aircraft absent from that table. Endpoint probes cover every target record for which a timestamp is supplied. This report currently includes {audit['extracted_hours']} extracted hours and {len(run['input_hours'])} hours in the segmentation run.</p>
{partial}<h2>What the data establish</h2>
{table(comparison,[('item','Property'),('paper','2021 description'),('observed','Observed 2024 evidence')])}
<p>The paper comparison uses the supplied excerpt and <a href="https://essd.copernicus.org/articles/13/357/2021/">Strohmeier et al. (2021), section 3.2</a>. Differences may reflect a different dataset/export or later implementation changes; these observations alone do not date the changes.</p>
<img alt="Duration, airport distance, gap classification and airport geometry evidence" src="data:image/png;base64,{graphic}">
<p>Every one of the 115,557 non-null starts and 115,699 ends maps to a fresh, valid supplied state with the same floored position-update timestamp. Usually this is the following snapshot; exact-integer updates can use the snapshot at the endpoint itself. In the aircraft sample, all 2,863 supplied endpoints fall in bins containing at least four fresh valid positions; 261 occur in bins containing exactly four. This is direct support for the legacy sparse-bin filter.</p>
<p>The supplied R files are exploratory scripts with January 2018 examples; the companion file plots results from a separate Spark implementation. They do not contain the complete production flight-table writer. Their median-speed split and quality filters are useful evidence, while the R airport routine uses three-dimensional nearest neighbors and arrival inheritance. Those details should not be copied unchanged into a 2024 reconstruction.</p>
<h2>Executable reconstruction and remaining uncertainty</h2>
<ol><li>Group snapshots by aircraft and sort chronologically. Require finite latitude, longitude, barometric altitude and position-update time; keep newly updated positions with <code>time - lastposupdate &lt; 1</code>. Use <code>lastposupdate</code> as the high-resolution clock. The sampled implementation that retains valid positions despite the aggregate <code>onground</code> flag performs better than excluding all ground-flagged rows.</li>
<li>Process daily batches, prepending an unfinished track from the preceding day. Remove sparse data in UTC/epoch-aligned 30-second bins with at most three fresh samples. Apply seven-point centered median checks, rejecting altitude deviations above 50 m and coordinate deviations above 0.5 degrees. The best tested window convention removes the first three and last six samples of the dense batch, before rejecting median outliers. This boundary behavior is inferred from agreement, rather than present verbatim in the R script.</li>
<li>Calculate <code>v[i] = haversine(position[i-1], position[i]) / (t[i]-t[i-1])</code>. Split only when <code>t[i]-t[i-1] &gt; 600</code> and <code>v[i] &lt; median(v)/4</code>. The median is derived from consecutive observed positions, not simply the reported velocity field. Median aggregation scope matters for borderline cases.</li>
<li>Use retained segment endpoints for <code>firstseen</code> and <code>lastseen</code>, converted to integer seconds. Reject reported durations of 600 seconds or less in the current replica. All non-null durations in the supplied reference exceed 600 seconds. Reported consecutive-flight gaps of exactly 600 seconds can arise from fractional gaps above 600.</li>
<li>Finalize segments followed by a detected landing gap. For the last segment of a batch, the tested legacy closure rule requires more than 600 seconds between its final position and midnight, and altitude within 2,500 m of the nearest airport elevation. Otherwise retain it for another daily batch, including short pending segments that have not yet reached the minimum duration. The hypothesis table compares carrying filtered positions with retaining input positions at the unfinished tail. Closure depends on the airport catalog, so it remains less certain than the split rule.</li>
<li>Find eligible airports using the asymmetric horizontal and vertical limits above, then choose the minimum horizontal distance. Vertical distance is the absolute difference between barometric altitude and airport elevation in metres, not a universal 2,500 m ceiling above sea level. Candidate counts exclude the selected airport. Integer distance fields are consistent with truncation.</li>
<li>Use the most frequent non-null callsign among retained positions, preserving its whitespace. Select records by UTC day of <code>lastseen</code>. The exact historical airport catalog, null-start handling, callsign ties and remaining batch-edge behavior are not uniquely determined by the supplied data.</li></ol>
<p><b>Best tested model:</b> <code>{html.escape(best['name'])}</code>. It exactly matches both timestamps for <b>{best['exact_pairs']:,}/{best['reference']:,}</b> reference records ({100*best['exact_pairs']/best['reference']:.2f}%) and matches both within 30 seconds for <b>{best['pairs_within_30s']:,}/{best['reference']:,}</b> ({100*best['pairs_within_30s']/best['reference']:.2f}%). It emits {best['predicted']:,} records in the comparison window. These are exploratory agreement metrics on the same aircraft sample used to select hypotheses, not held-out segmentation validation or detection accuracy against independent ground truth. A tolerance match requires both endpoints to match the same reference flight. Three days of input do not reproduce any unfinished state inherited before May 31.</p>
<p>The separate gap diagnostic labels {gap_rule['n']:,} gaps above 30 seconds using reference interval membership with a 30-second endpoint tolerance. A June 1 aircraft median classifies {gap_rule['correct']:,} correctly, including {len(long)-error:,}/{len(long):,} gaps over 600 seconds. This conditional diagnostic excludes gaps that cannot be assigned to reference flights; its accuracy must not be confused with full reconstruction accuracy. The daily replay instead calculates its median over the current batch and pending track.</p>
<p>The best model retains the pending segment's input positions from its retained start onward, including the trailing samples needed for the next batch's median window. It reruns the daily filters even when a later day has no new positions. Permanently discarding the pending tail reduces exact timestamp agreement. Remaining timestamp differences are listed below; the sample's one reference record with a null start is outside this comparison.</p>
{residual_table}
{table(model_rows,[('model','Hypothesis'),('predicted','Predicted'),('reference','Reference'),('exact','Exact endpoint pairs'),('recall','Exact recall'),('within30','Both within 30 s')])}
<p>{callsign_text}</p><p>{replica_text}</p><h2>Airport geometry and source limitations</h2>
<p>Airport coordinates were calibrated using flight IDs whose remainder modulo 3 is nonzero, and evaluated on the remaining flight IDs. This diagnostic uses reference labels for calibration; it is not an independent airport database. Across {geometry['airports']:,} airports and {geometry['holdout_observations']:,} held-out observations, {geometry['horizontal_within_2m']:,} ({100*geometry['horizontal_within_2m']/geometry['holdout_observations']:.2f}%) direct horizontal distances agree within 2 m. The concentration of residuals between 0 and 1 m supports integer truncation. Using unmodified OurAirports coordinates leaves larger errors where metadata have changed.</p>
<p>The local <a href="https://github.com/davidmegginson/ourairports-data">OurAirports history</a> supplies an as-of-2024-06-01 catalog; a pinned 2022 snapshot is also tested. Neither is asserted to be OpenSky’s private historical catalog. Current catalog type and code filters can change candidate counts even when the distance rule is correct. With small, medium and large airports having GPS codes and elevations, the following assignment results use the <em>reference endpoints</em>; they isolate airport rules from segmentation error and include null-to-null matches:</p>
{table(airport_rows,[('catalog','OurAirports snapshot'),('kind','Endpoint'),('observations','Reference endpoints'),('matches','Matching airport'),('agreement','Agreement'),('counts','Matching alternate count')])}
<p>Metre-scale agreement supports direct endpoint distances, without requiring trajectory extrapolation. It does not uniquely establish the exact Earth radius or distinguish surface arcs from horizontal chords at every observed distance.</p>
<p>The provided file omits <code>track</code>. All 741,362 alternate-candidate airport-code slots are null, although their distances remain available. There are {metadata['null_firstseen']} null <code>firstseen</code> values. Therefore alternate-candidate identities, those start times and some production processing details cannot be recovered exactly from these artifacts alone.</p>
<h2>Reproduce and inspect</h2>
<p>Raw data are read-only. Research code is under <code>scripts/opensky_reverse_engineering/</code>. Extracted evidence and detailed per-flight comparisons are under <code>{html.escape(str(out))}</code>. Extraction resumes completed hourly files. Run from the repository root:</p>
<pre>POLARS_MAX_THREADS=2 .venv/bin/python scripts/opensky_reverse_engineering/extract.py
POLARS_MAX_THREADS=2 .venv/bin/python scripts/opensky_reverse_engineering/run_analysis.py
.venv/bin/python -m pytest -q tests/opensky_reverse_engineering/test_rules.py</pre>
<p>Key outputs: <code>reconstructed_sample_flights.parquet</code> (approximate full flight rows with catalog provenance), <code>endpoint_audit.json</code>, <code>segmentation_run.json</code>, <code>segmentation_metrics.json</code>, <code>labeled_gaps.parquet</code>, <code>gap_metrics.json</code>, <code>airport_geometry.json</code>, <code>airport_candidate_metrics*.json</code>, and <code>hypotheses/*_pairs.parquet</code>. The latter retain mismatches for inspection, rather than hiding them in aggregate scores. The old R source remains at <code>/mnt/storage2/opensky-algorithm/opensky-algorithm-from-mstromeier/</code>; its embedded examples refer to January 2018.</p>'''
    css='''body{font-family:system-ui,sans-serif;max-width:1120px;margin:42px auto;padding:0 25px;color:#192d37;line-height:1.55}h1{font-size:2.1rem;line-height:1.2}h2{margin-top:2rem}p,li{max-width:1000px}.lead{font-size:1.15rem}table{border-collapse:collapse;width:100%;font-size:.86rem}th,td{border:1px solid #d3dfe3;padding:8px;text-align:left}th{background:#eef4f6}code,pre{font-size:.87em}pre{background:#eef4f6;overflow:auto;padding:14px}img{width:100%;margin:25px 0}.caution{border-left:4px solid #bd422e;padding-left:15px}a{color:#12647d}'''
    dest=reports/'opensky_reverse_engineering_2024-06-01.html'
    dest.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenSky reverse engineering 2024-06-01</title><style>'+css+'</style><body>'+body+'</body></html>')
    print(dest)


if __name__=='__main__':
    build()
