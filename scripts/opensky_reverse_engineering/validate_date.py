"""Validate the fixed 2024 reconstruction on another UTC flight partition.

No hypothesis search or airport-coordinate fitting is performed. Extract the
three-day cache with extract.py --date DATE --sample-size 2048 first.
"""

from __future__ import annotations

import argparse
import ast
from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

import polars as pl
import pyarrow.parquet as pq

from analyze_endpoints import endpoint_audit
from airport_candidates import run as airport_candidates
from daily_replay import infer
from evaluate_replica import run as evaluate_replica
from extract import OUT, RAW
from replica import AirportIndex, export
from segments import evaluate

NAME='daily_replay_emptyTrue_trimBeforeTrue_rawCarry'
CATALOG='ourairports_2024-06-01.parquet'
ROOT=Path(__file__).resolve().parents[2]


def prepare(out, target_date, sample_size):
    """Freeze the preselected model and catalog before looking at agreement."""
    out.mkdir(parents=True,exist_ok=True)
    source=OUT/CATALOG
    destination=out/CATALOG
    if not destination.exists():
        shutil.copy2(source,destination)
    catalog_hash=hashlib.sha256(destination.read_bytes()).hexdigest()
    if catalog_hash!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Validation must use the unchanged 2024 airport catalog')
    path=out/'validation_protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text())
        if (protocol['target_date']!=target_date.isoformat() or protocol['sample_size']!=sample_size
                or protocol['airport_catalog_sha256']!=catalog_hash or protocol['hypothesis']!=NAME):
            raise ValueError('Existing validation protocol differs')
        return protocol
    protocol={'target_date':target_date.isoformat(),'sample_size':sample_size,
        'population':'SHA256 sample of aircraft present in target flights_data4 partition',
        'selection_seed':f'opensky-{target_date:%Y%m%d}:',
        'validation':'Temporal holdout; no rule or catalog tuning on this date',
        'original_model_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'hypothesis':NAME,'reprocess_empty':True,'trim_before_filter':True,'carry_raw':True,
        'freshness_seconds':1,'sparse_bin_seconds':30,'minimum_bin_points':4,'median_window_points':7,
        'altitude_tolerance_m':50,'coordinate_tolerance_degrees':.5,'trim_start_points':3,'trim_end_points':6,
        'gap_seconds':600,'median_speed_fraction':.25,'minimum_duration_exclusive_seconds':600,
        'context_days_before':1,'context_days_after':1,'airport_catalog':CATALOG,
        'airport_catalog_sha256':catalog_hash,'departure_horizontal_limit_m':15000,
        'departure_vertical_limit_m':2000,'arrival_horizontal_limit_m':30000,'arrival_vertical_limit_m':2500}
    path.write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def verify_fixed_rules(out,protocol):
    """Allow date plumbing, but require prediction rules to match the frozen commit."""
    checks={}
    targets={'daily_replay.py':{'infer':{'target_day':'DAY'},'record':{}},
             'segments.py':{'Rules':{},'clean':{},'gap_features':{},'evaluate':{'day':'DAY'}},
             'analyze_endpoints.py':{'haversine':{}},
             'replica.py':{'AirportIndex':{},'export':{'day':'DAY'}}}
    class NormalizeDate(ast.NodeTransformer):
        def __init__(self,mapping):
            self.mapping=mapping
        def visit_Name(self,node):
            if node.id in self.mapping:
                node.id=self.mapping[node.id]
            return node
        def visit_FunctionDef(self,node):
            offset=len(node.args.args)-len(node.args.defaults)
            defaults={arg.arg:node.args.defaults[i-offset] for i,arg in enumerate(node.args.args) if i>=offset}
            node.args.args=[a for a in node.args.args if a.arg not in self.mapping]
            node.args.defaults=[defaults[a.arg] for a in node.args.args if a.arg in defaults]
            return self.generic_visit(node)
    for filename,functions in targets.items():
        relative='scripts/opensky_reverse_engineering/'+filename
        original=subprocess.check_output(['git','show',protocol['original_model_commit']+':'+relative],cwd=ROOT,text=True)
        current=(ROOT/relative).read_text()
        for name,mapping in functions.items():
            nodes=[]
            for code in [original,current]:
                node=next(x for x in ast.parse(code).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
                nodes.append(ast.dump(NormalizeDate(mapping).visit(node),include_attributes=False))
            checks[filename+':'+name]=nodes[0]==nodes[1]
    (out/'validation_rule_equivalence.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()):
        raise ValueError(f'Prediction rules changed beyond date plumbing: {checks}')


def infer_sample(out,target_date,protocol):
    day=int(datetime.combine(target_date,datetime.min.time(),timezone.utc).timestamp())
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['target_date']!=target_date.isoformat() or manifest['sample_size']!=protocol['sample_size']:
        raise ValueError('Extraction does not match the frozen protocol')
    aircraft=sorted(manifest['sample_aircraft'])
    files=sorted((out/'sample_hours').glob('*.parquet'))
    expected={(target_date+timedelta(days=d)).strftime('%Y-%m-%d')+f'-{h:02d}' for d in [-1,0,1] for h in range(24)}
    if {p.stem for p in files}!=expected:
        raise ValueError(f'Need exactly the 72 context hours, found {len(files)}')
    airports=AirportIndex(pl.read_parquet(out/CATALOG))
    records=[]
    observed=set()
    for start in range(0,len(aircraft),32):
        batch=aircraft[start:start+32]
        data=pl.scan_parquet(files).filter(pl.col('icao24').is_between(pl.lit(batch[0]),pl.lit(batch[-1]))
            &pl.col('icao24').is_in(batch)).collect().sort('icao24','time')
        for (ac,),group in data.partition_by('icao24',as_dict=True).items():
            observed.add(ac)
            records.extend(infer(ac,group,airports,target_day=day,reprocess_empty=True,
                                 trim_before_filter=True,carry_raw=True))
        print(f'Fixed model: {min(start+32,len(aircraft))}/{len(aircraft)} aircraft',flush=True)
    predictions=pl.DataFrame(records)
    reference=pl.read_parquet(out/'reference_flights.parquet').filter(pl.col('icao24').is_in(aircraft))
    summary,pairs=evaluate(predictions,reference,day=day)
    summary.update(name=NAME,reference_including_null_starts=reference.height,
        sample_aircraft=len(aircraft),sample_aircraft_with_states=len(observed),
        starts_before_context=reference.filter(pl.col('firstseen')<day-86400).height,
        rules={'implementation':'daily_replay.infer','target_day':day,
               'reprocess_empty':True,'trim_before_filter':True,'carry_raw':True})
    (out/'hypotheses').mkdir(exist_ok=True)
    predictions.write_parquet(out/'hypotheses'/f'{NAME}.parquet')
    pairs.write_parquet(out/'hypotheses'/f'{NAME}_pairs.parquet')
    (out/'segmentation_metrics.json').write_text(json.dumps([summary],indent=2)+'\n')
    (out/'segmentation_run.json').write_text(json.dumps({'target_date':target_date.isoformat(),
        'partial':False,'sample_aircraft':len(aircraft),'input_hours':[p.stem for p in files],
        'lower_reference_bound':day,'upper_reference_bound':day+86400,
        'protocol':'validation_protocol.json','tuned_on_target_date':False},indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return day


def inventory(out,target_date):
    manifest=json.loads((out/'manifest.json').read_text())
    raw=Path(manifest.get('raw_root',RAW))
    files=sorted(p for offset in [-1,0,1] for p in (raw/'state_vectors_data4'/
        (target_date+timedelta(days=offset)).strftime('year=%Y/month=%m/day=%d')).glob('hour=*/data.parquet'))
    entries=[{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
              'rows':pq.ParquetFile(p).metadata.num_rows} for p in files]
    source={'state_files':entries,'total_bytes':sum(x['bytes'] for x in entries),
            'total_state_rows':sum(x['rows'] for x in entries),
            'reference_path':str(raw/'flights_data4'/target_date.strftime('year=%Y/month=%m/day=%d')/'data.parquet')}
    (out/'source_inventory.json').write_text(json.dumps(source,indent=2)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (out/'validation_code_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')


def table(rows,columns):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(label)}</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(row.get(key,"")))}</td>' for key,_ in columns)+'</tr>' for row in rows)+'</tbody></table>'


def report(out,target_date):
    protocol=json.loads((out/'validation_protocol.json').read_text())
    metrics=json.loads((out/'segmentation_metrics.json').read_text())[0]
    replica=json.loads((out/'replica_metrics.json').read_text())
    audit=json.loads((out/'endpoint_audit.json').read_text())
    metadata=audit['metadata']
    source=json.loads((out/'source_inventory.json').read_text())
    airport=json.loads((out/'airport_candidate_metrics.json').read_text())
    previous=max(json.loads((OUT/'segmentation_metrics.json').read_text()),key=lambda x:x['exact_pairs'])
    old_replica=json.loads((OUT/'replica_metrics.json').read_text())
    n=metrics['reference']
    def ratio(count,total):
        return f'{count:,}/{total:,} ({100*count/total:.2f}%)'
    comparisons=[]
    for key,label in [('exact_pairs','Both timestamps exact'),('exact_starts','Start exact'),('exact_ends','End exact'),
        ('pairs_within_1s','Both timestamps within 1 s'),('pairs_within_5s','Both timestamps within 5 s'),
        ('pairs_within_30s','Both timestamps within 30 s'),('pairs_within_60s','Both timestamps within 60 s'),
        ('pairs_within_120s','Both timestamps within 120 s')]:
        comparisons.append({'metric':label,'previous':ratio(previous[key],previous['reference']),
                            'current':ratio(metrics[key],n)})
    comparisons.append({'metric':'Exact timestamps, callsign and both airports',
        'previous':ratio(old_replica['boundary_callsign_both_airports_matches'],previous['reference']),
        'current':ratio(replica['boundary_callsign_both_airports_matches'],n)})
    fields=[{'field':key,'match':ratio(value,replica['exact_boundary_pairs'])}
            for key,value in replica['field_matches_conditional_on_exact_boundaries'].items()]
    endpoint_counts={x['kind']:x['endpoints'] for x in audit['selected_endpoint_coverage']}
    endpoint_rows=[{'kind':kind,'coverage':ratio(endpoint_counts.get(kind,0),metadata['flights']-(metadata['null_firstseen'] if kind=='departure' else 0)),
        'airport':ratio(next(x for x in airport if x['kind']==kind)['airport_matches_including_null'],
                       next(x for x in airport if x['kind']==kind)['observations'])}
        for kind in ['departure','arrival']]
    residuals=pl.read_csv(out/'best_model_residuals.csv').with_columns(
        pl.max_horizontal(pl.col('start_delta_seconds').abs(),pl.col('end_delta_seconds').abs()).alias('max_error'))
    residual_table=table(residuals.sort('max_error',descending=True,nulls_last=False).head(30).to_dicts(),
        [('icao24','Aircraft'),('firstseen','Reference start'),('lastseen','Reference end'),
         ('start_delta_seconds','Predicted − reference start (s)'),('end_delta_seconds','Predicted − reference end (s)')])
    body=f'''<h1>OpenSky fixed-model validation — {target_date}</h1>
<p class="lead">Exact start/end agreement: <b>{ratio(metrics['exact_pairs'],n)}</b>.
Both endpoints within 30 seconds: <b>{ratio(metrics['pairs_within_30s'],n)}</b>.
The rules and airport catalog were fixed from the 2024-06-01 reconstruction before this validation.</p>
<p>The full target partition contains {metadata['flights']:,} flights from {metadata['aircraft']:,} aircraft,
including {metadata['null_firstseen']:,} null starts. Trajectories were reconstructed for a deterministic sample of
{metrics['sample_aircraft']:,} aircraft, covering {metrics['reference_including_null_starts']:,} reference flights
({n:,} with known starts). The model emitted {metrics['predicted']:,} target-day flights.
All {metrics['sample_aircraft_with_states']:,} sampled aircraft with available states were processed.</p>
<p>Input context: {len(source['state_files'])} hours, {source['total_state_rows']:,} snapshots,
{source['total_bytes']/1e9:.2f} GB compressed, from {target_date-timedelta(days=1)} through {target_date+timedelta(days=1)}.
The sample is selected by SHA-256 ordering of aircraft IDs with seed <code>{protocol['selection_seed']}</code>.
No parameters were fitted, no alternative segmentation hypotheses were selected, and no airport coordinates were calibrated on this date.
An abstract-syntax comparison verifies that the inference and export functions match the original commit, allowing only date parameters.</p>
<h2>Accuracy compared with the development sample</h2>
{table(comparisons,[('metric','Metric'),('previous','2024-06-01: development sample'),('current',f'{target_date}: temporal validation')])}
<p>Each tolerance match requires both endpoints to agree with the same reference flight. Reference records with null starts are excluded from these denominators.
The 2024 and validation samples contain different aircraft populations and sizes. This is agreement with OpenSky, not accuracy against independently verified flights.
Sampling aircraft already present in the flight table does not measure false positives among aircraft absent from that table.</p>
<p>Of the {metrics['predicted']:,} predictions, {replica['predictions_matching_known_start_reference_within_30s']:,} match a known-start reference flight within 30 seconds at both ends;
{replica['other_predictions_matching_null_start_reference_end']:,} others share the exact end of a reference flight whose start is null.
The remaining {replica['remaining_unmatched_predictions']:,} predictions have neither match. These cases are saved in <code>unmatched_predictions.csv</code>.
An unmatched prediction can be an extra flight, a split/merged reference flight or a larger timestamp discrepancy.</p>
<h2>Airport fields and callsign</h2>
<p>The following comparisons are conditional on the {replica['exact_boundary_pairs']:,} exact boundary matches.
The public <code>{CATALOG}</code> catalog is unchanged; airport differences can reflect missing, renamed or moved airports.
Null-to-null comparisons count as matches.</p>
{table(fields,[('field','Field'),('match','Exact matches')])}
<p>{ratio(replica['all_nine_nonarray_fields_matches'],n)} reference flights match the timestamps and all nine fields above together.
Alternative-candidate ordering and identities remain unresolved and are excluded from that measure.</p>
<h2>Full-table endpoint checks</h2>
{table(endpoint_rows,[('kind','Endpoint'),('coverage','Fresh valid source position found'),('airport','Airport agreement using reference endpoints')])}
<p>The conditional airport check recognizes ident/GPS/ICAO aliases, while the exported replica emits GPS codes.
The full table has {metadata['duration_at_most_600']:,} known durations at or below 600 seconds and
{metadata['duration_below_900']:,} below 900 seconds. Every record ends on its partition day: {metadata['all_end_on_partition_day']}.</p>
<h2>Residuals and limits</h2>
<p>{residuals.height:,} known-start reference flights lack an exact boundary-pair match. The largest 30 residuals are shown below;
all are saved in <code>best_model_residuals.csv</code> and Parquet. Blank deltas mean no prediction for that aircraft.
The comparison selects the closest predicted pair for diagnostics; large deltas can indicate missing or merged flights rather than simple timestamp shifts.</p>
{residual_table}
<p>{metrics['starts_before_context']} sampled known starts precede the extracted context.
One preceding day cannot recreate older pending state; one subsequent day may leave a high-altitude track unfinished.
Null-start behavior and the exact historical OpenSky airport database are not reconstructed.</p>
<h2>Reproduce</h2>
<pre>POLARS_MAX_THREADS=2 .venv/bin/python scripts/opensky_reverse_engineering/validate_date.py --date {target_date} --sample-size {metrics['sample_aircraft']} --prepare-only
POLARS_MAX_THREADS=2 .venv/bin/python scripts/opensky_reverse_engineering/extract.py --date {target_date} --sample-size {metrics['sample_aircraft']}
POLARS_MAX_THREADS=2 .venv/bin/python scripts/opensky_reverse_engineering/validate_date.py --date {target_date} --sample-size {metrics['sample_aircraft']}</pre>
<p>Evidence and exported flights: <code>{html.escape(str(out))}</code>.
The frozen settings are in <code>validation_protocol.json</code>; runtime code hashes are in <code>validation_code_hashes.json</code>.
Raw input files are read-only. The original model commit is <code>{protocol['original_model_commit']}</code>.</p>'''
    css='body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#192d37;line-height:1.5}h1{line-height:1.2}.lead{font-size:1.2rem}table{border-collapse:collapse;width:100%;font-size:.88rem;margin:20px 0}th,td{border:1px solid #ccd8dd;padding:8px;text-align:left}th,pre{background:#eef4f6}pre{padding:15px;overflow:auto}h2{margin-top:32px}code{font-size:.9em}'
    path=ROOT/'reports'/f'opensky_validation_{target_date}.html'
    path.parent.mkdir(exist_ok=True)
    path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenSky validation '+str(target_date)+'</title><style>'+css+'</style><body>'+body+'</body></html>')
    print(path,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',type=date.fromisoformat,required=True)
    parser.add_argument('--sample-size',type=int,default=2048)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    out=args.out or OUT.parent/args.date.isoformat()
    if args.date==date(2024,6,1):
        raise ValueError('Use run_analysis.py for the original development date')
    protocol=prepare(out,args.date,args.sample_size)
    verify_fixed_rules(out,protocol)
    if args.prepare_only:
        print(out/'validation_protocol.json')
        return
    if args.report_only:
        report(out,args.date)
        return
    day=infer_sample(out,args.date,protocol)
    export(NAME,out=out,day=day)
    evaluate_replica(out=out,day=day)
    endpoint_audit(out)
    airport_candidates(out,policies=['fixed_wing_gps'])
    inventory(out,args.date)
    report(out,args.date)


if __name__=='__main__':
    main()
