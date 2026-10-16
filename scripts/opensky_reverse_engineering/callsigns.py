"""Compare callsign selection rules using the supplied reference intervals."""

import json

import numpy as np
import polars as pl

from extract import OUT
from segments import Rules, clean


def run(out=OUT):
    aircraft=json.loads((out/'manifest.json').read_text())['sample_aircraft']
    f=pl.read_parquet(out/'reference_flights.parquet').drop_nulls('firstseen')
    files=sorted((out/'sample_hours').glob('*.parquet'))
    records=[]
    for index in range(0,len(aircraft),32):
        acs=aircraft[index:index+32]
        d=pl.scan_parquet(files).filter(pl.col('icao24').is_in(acs)).collect().sort('icao24','time')
        for (ac,),a in d.partition_by('icao24',as_dict=True).items():
            x=clean(a,Rules('',clock='position',sparse=True,median=True,ground=True,density_first=True))
            if not len(x['time']):
                continue
            for row in f.filter(pl.col('icao24')==ac).to_dicts():
                lo,hi=np.searchsorted(x['clock'],[row['firstseen'],row['lastseen']+1])
                values=x['callsign'][lo:hi]
                values=values[values!=None]  # noqa: E711
                if len(values):
                    unique,counts=np.unique(values,return_counts=True)
                    mode=str(unique[np.argmax(counts)]);first=str(values[0]);last=str(values[-1]);n=len(unique)
                else:
                    mode=first=last=None;n=0
                records.append({'flight_id':row['flight_id'],'icao24':ac,'expected':row['callsign'],
                                'first':first,'last':last,'mode':mode,'distinct_callsigns':n})
    d=pl.DataFrame(records)
    d.write_parquet(out/'callsign_comparison.parquet')
    report={'flights':d.height,'multiple_callsigns':d.filter(pl.col('distinct_callsigns')>1).height}
    for c in ['first','last','mode']:
        report[c+'_matches']=d.select(pl.col(c).eq_missing(pl.col('expected')).sum()).item()
        changing=d.filter(pl.col('distinct_callsigns')>1)
        report[c+'_matches_when_changing']=changing.select(pl.col(c).eq_missing(pl.col('expected')).sum()).item()
    (out/'callsign_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()
