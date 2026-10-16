"""Test daily processing with a retained unfinished track from the previous day.

This hypothesis combines the observed 3/6 point window edges with the legacy
final-arrival test: more than 600 seconds old and within 2500 m of airport
elevation. No reference boundaries or labels enter reconstruction.
"""

import json

import numpy as np
import polars as pl

from airport_candidates import xyz
from extract import OUT
from replica import AirportIndex
from segments import DAY, Rules, clean, evaluate, gap_features


def record(ac,x,start,end):
    callsigns=x['callsign'][start:end]
    callsigns=callsigns[callsigns!=None]  # noqa: E711
    if len(callsigns):
        values,counts=np.unique(callsigns,return_counts=True)
        mode=str(values[np.argmax(counts)])
        first,last=str(callsigns[0]),str(callsigns[-1])
    else:
        mode=first=last=None
    return {'icao24':ac,'firstseen':int(x['clock'][start]),'lastseen':int(x['clock'][end-1]),'points':int(end-start),
        'callsign_mode':mode,'callsign_first':first,'callsign_last':last,
        'first_lat':float(x['lat'][start]),'first_lon':float(x['lon'][start]),
        'last_lat':float(x['lat'][end-1]),'last_lon':float(x['lon'][end-1]),
        'first_baroaltitude':float(x['baroaltitude'][start]),'last_baroaltitude':float(x['baroaltitude'][end-1])}


def infer(ac,states,airports,reprocess_empty=True,trim_before_filter=True,carry_raw=True,target_day=DAY):
    """Infer flights; defaults select the best measured configuration.

    These are empirically inferred batch semantics, not recovered production
    source. All state comes from the supplied trajectories and airport catalog.
    """
    rules=Rules('daily_replay',clock='position',sparse=True,median=True,ground=True,density_first=True,
                trim_start=3,trim_end=6,split='median_quarter',trim_before_filter=trim_before_filter)
    carry=None
    output=[]
    for day in [target_day-86400,target_day,target_day+86400]:
        new=states.filter(pl.col('time').is_between(day,day+86399)
            &pl.col('lat').is_finite()&pl.col('lon').is_finite()&pl.col('baroaltitude').is_finite()
            &((pl.col('time')-pl.col('lastposupdate'))<1)).select(
                'time','lastposupdate','lat','lon','baroaltitude','velocity','heading','vertrate','onground','callsign')
        if new.is_empty() and carry is None:
            continue
        if new.is_empty() and not reprocess_empty:
            combined=carry
            x=clean(carry,Rules('',clock='position',ground=True))
        else:
            combined=pl.concat([carry,new],how='vertical_relaxed') if carry is not None else new
            x=clean(combined,rules)
        if len(x['time'])<2:
            carry=None
            continue
        dt,distance,speed=gap_features(x)
        split=(dt>600)&(speed<np.nanmedian(speed)/4)
        cuts=np.r_[0,np.flatnonzero(split)+1,len(x['clock'])]
        nearest=airports.tree.query(xyz(np.array([x['lat'][-1]]),np.array([x['lon'][-1]])),k=1)[1][0]
        close_last=(day+86400-x['clock'][-1]>600) and abs(x['baroaltitude'][-1]-airports.elevation[nearest])<2500
        carry=None
        for number,(start,end) in enumerate(zip(cuts[:-1],cuts[1:])):
            r=record(ac,x,start,end)
            if number==len(cuts)-2 and not close_last:
                if carry_raw:
                    # Retain trailing samples as rolling-filter context for the
                    # next batch; do not permanently discard an unfinished tail.
                    carry=combined.filter(pl.col('lastposupdate')>=x['clock'][start])
                else:
                    carry=pl.DataFrame({c:(a[start:end].tolist() if a.dtype==object else a[start:end])
                        for c,a in x.items() if c!='clock'},schema_overrides={'callsign':pl.String,'onground':pl.Boolean})
            elif r['lastseen']-r['firstseen']>600:
                output.append(r)
    return output


def run(out=OUT):
    aircraft=json.loads((out/'manifest.json').read_text())['sample_aircraft']
    reference=pl.read_parquet(out/'reference_flights.parquet').filter(pl.col('icao24').is_in(aircraft))
    airports=AirportIndex(pl.read_parquet(out/'ourairports_2024-06-01.parquet'))
    outputs={(False,False,False):[],(True,False,False):[],(False,True,False):[],(True,True,False):[],
             (False,True,True):[],(True,True,True):[]}
    files=sorted((out/'sample_hours').glob('*.parquet'))
    if len(files)!=72:
        raise ValueError('Daily replay needs all 72 extracted hours')
    for index in range(0,len(aircraft),32):
        d=pl.scan_parquet(files).filter(pl.col('icao24').is_in(aircraft[index:index+32])).collect().sort('icao24','time')
        for (ac,),a in d.partition_by('icao24',as_dict=True).items():
            for empty,before,raw in outputs:
                outputs[empty,before,raw].extend(infer(ac,a,airports,reprocess_empty=empty,trim_before_filter=before,carry_raw=raw))
        print(f'Daily replay {min(index+32,len(aircraft))}/{len(aircraft)}',flush=True)
    metrics=json.loads((out/'segmentation_metrics.json').read_text())
    metrics=[x for x in metrics if not x['name'].startswith('daily_replay')]
    for (empty,before,raw),rows in outputs.items():
        name=f'daily_replay_empty{empty}_trimBefore{before}'+('_rawCarry' if raw else '')
        p=pl.DataFrame(rows)
        summary,pairs=evaluate(p,reference)
        summary.update(name=name,rules={'implementation':'daily_replay.infer','reprocess_empty':empty,'trim_before_filter':before,'carry_raw':raw})
        metrics.append(summary)
        p.write_parquet(out/'hypotheses'/f'{name}.parquet')
        pairs.write_parquet(out/'hypotheses'/f'{name}_pairs.parquet')
        print(json.dumps(summary,indent=2))
    metrics.sort(key=lambda x:x['exact_pairs'],reverse=True)
    (out/'segmentation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__=='__main__':
    run()
