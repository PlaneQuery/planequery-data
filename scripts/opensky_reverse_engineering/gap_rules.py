"""Compare split/merge decisions at observed gaps against reference flights."""

import json

import numpy as np
import polars as pl

from extract import OUT
from segments import DAY, Rules, clean, gap_features


def run(out=OUT):
    manifest=json.loads((out/'manifest.json').read_text())
    f=pl.read_parquet(out/'reference_flights.parquet').drop_nulls('firstseen')
    files=sorted(p for p in (out/'sample_hours').glob('*.parquet')
                 if p.name.startswith(('2024-05-31','2024-06-01')))
    records=[]
    endpoint_counts=[]
    for index in range(0,len(manifest['sample_aircraft']),32):
        acs=manifest['sample_aircraft'][index:index+32]
        data=pl.scan_parquet(files).filter(pl.col('icao24').is_in(acs)).collect().sort('icao24','time')
        for (ac,),a in data.partition_by('icao24',as_dict=True).items():
            x=clean(a,Rules('',clock='position',ground=True,sparse=True,median=True))
            if len(x['time'])<2:
                continue
            dt,distance,speed=gap_features(x)
            day_median=float(np.nanmedian(speed[x['clock'][1:]>=DAY]))
            r=f.filter(pl.col('icao24')==ac).sort('firstseen')
            if not r.height:
                continue
            starts,ends=r['firstseen'].to_numpy(),r['lastseen'].to_numpy()
            membership=np.searchsorted(starts,x['clock']+30,side='right')-1
            membership[(membership<0)|(x['clock']>ends[np.maximum(membership,0)]+30)]=-1
            for i in np.flatnonzero(dt>30):
                left,right=membership[i],membership[i+1]
                if left<0 or right<0:
                    continue
                records.append({'icao24':ac,'t0':float(x['clock'][i]),'t1':float(x['clock'][i+1]),
                                'gap':float(dt[i]),'distance':float(distance[i]),'gap_speed':float(speed[i]),
                                'median_speed':float(np.nanmedian(speed)),'day_median_speed':day_median,'last_velocity':float(x['velocity'][i]),
                                'next_velocity':float(x['velocity'][i+1]),'last_heading':float(x['heading'][i]),
                                'lat0':float(x['lat'][i]),'lon0':float(x['lon'][i]),'lat1':float(x['lat'][i+1]),'lon1':float(x['lon'][i+1]),
                                'split':bool(left!=right),'left_flight':int(r['flight_id'][int(left)]),'right_flight':int(r['flight_id'][int(right)])})
    g=pl.DataFrame(records)
    g.write_parquet(out/'labeled_gaps.parquet')
    y=g['split'].to_numpy();dt=g['gap'].to_numpy();ds=g['distance'].to_numpy();v=g['gap_speed'].to_numpy()
    metrics=[]
    def score(name,p):
        metrics.append({'name':name,'n':len(y),'splits':int(y.sum()),'merges':int((~y).sum()),
                        'correct':int((p==y).sum()),'false_split':int((p&~y).sum()),'false_merge':int((~p&y).sum())})
    for gap in [300,600,900,1200]:
        score(f'gap_{gap}',dt>gap)
        for fraction in [.125,.2,.25,1/3,.5,.75,1.]:
            for velocity in ['median_speed','day_median_speed','last_velocity','next_velocity']:
                score(f'gap_{gap}_{velocity}_x{fraction:g}',(dt>gap)&(v<g[velocity].to_numpy()*fraction))
    vel=g['last_velocity'].to_numpy();heading=np.radians(g['last_heading'].to_numpy())
    lat=g['lat0'].to_numpy()+np.cos(heading)*vel*dt/111195
    lon=g['lon0'].to_numpy()+np.sin(heading)*vel*dt/(111195*np.cos(np.radians(g['lat0'].to_numpy())))
    score('paper_literal_constant_velocity_0.5deg',(dt>600)&((np.abs(lat-g['lat1'].to_numpy())>.5)|(np.abs(lon-g['lon1'].to_numpy())>.5)))
    metrics.sort(key=lambda x:x['correct'],reverse=True)
    (out/'gap_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics[:12],indent=2))


if __name__=='__main__':
    run()
