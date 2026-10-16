"""Export approximate flights from independent predictions and a public catalog.

This never reads reference boundaries or reference airport labels. The caller
chooses an already evaluated segmentation hypothesis. The catalog and ordering
of alternative airports are explicit approximations, not OpenSky source code.
"""

import argparse
import json
import math

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from airport_candidates import xyz
from analyze_endpoints import RADIUS
from extract import OUT
from segments import DAY, Rules, reconstruct


class AirportIndex:
    def __init__(self, airports):
        self.airports=airports.drop_nulls(['gps_code','latitude_deg','longitude_deg','elevation_ft']).filter(
            pl.col('type').is_in(['small_airport','medium_airport','large_airport']))
        self.tree=cKDTree(xyz(self.airports['latitude_deg'].to_numpy(),self.airports['longitude_deg'].to_numpy()))
        self.elevation=self.airports['elevation_ft'].to_numpy()*.3048
        self.codes=self.airports['gps_code'].to_list()

    def candidates(self, lat, lon, altitude, kind):
        hlimit,vlimit=(15000,2000) if kind=='departure' else (30000,2500)
        point=xyz(np.asarray([lat]),np.asarray([lon]))[0]
        ids=np.asarray(self.tree.query_ball_point(point,2*np.sin(hlimit/(2*RADIUS))),dtype=int)
        ids=ids[np.abs(self.elevation[ids]-altitude)<vlimit]
        chord=np.linalg.norm(self.tree.data[ids]-point,axis=1)
        horizontal=2*RADIUS*np.arcsin(np.clip(chord/2,0,1))
        result=[(self.codes[int(i)],float(h),float(abs(self.elevation[i]-altitude))) for i,h in zip(ids,horizontal) if h<hlimit]
        return sorted(result,key=lambda x:(x[1],x[0]))


def export(hypothesis, out=OUT, day=DAY):
    predictions=pl.read_parquet(out/'hypotheses'/f'{hypothesis}.parquet').filter(pl.col('lastseen').is_between(day,day+86399))
    if 'first_baroaltitude' not in predictions.columns:
        # Refresh an older research cache with endpoint altitude fields.
        metrics=json.loads((out/'segmentation_metrics.json').read_text())
        rules=Rules(**next(x['rules'] for x in metrics if x['name']==hypothesis))
        aircraft=json.loads((out/'manifest.json').read_text())['sample_aircraft']
        records=[]
        for start in range(0,len(aircraft),32):
            data=pl.scan_parquet(sorted((out/'sample_hours').glob('*.parquet'))).filter(
                pl.col('icao24').is_in(aircraft[start:start+32])).collect().sort('icao24','time')
            for (ac,),group in data.partition_by('icao24',as_dict=True).items():
                records.extend(reconstruct(ac,group,rules))
        refreshed=pl.DataFrame(records).filter(pl.col('lastseen').is_between(day,day+86399))
        keys=['icao24','firstseen','lastseen']
        if not refreshed.select(keys).sort(keys).equals(predictions.select(keys).sort(keys)):
            raise ValueError('Refreshed model boundaries changed; rerun the analysis')
        predictions=refreshed
    index=AirportIndex(pl.read_parquet(out/'ourairports_2024-06-01.parquet'))
    rows=[]
    for p in predictions.iter_rows(named=True):
        row={c:p[c] for c in ['icao24','firstseen','lastseen']}
        row['callsign']=p['callsign_mode']
        row['day']=(p['lastseen']//86400)*86400
        for kind,end in [('departure','first'),('arrival','last')]:
            candidates=index.candidates(p[end+'_lat'],p[end+'_lon'],p[end+'_baroaltitude'],kind)
            prefix='est'+kind+'airport'
            row[prefix]=candidates[0][0] if candidates else None
            row[prefix+'horizdistance']=math.floor(candidates[0][1]) if candidates else None
            row[prefix+'vertdistance']=math.floor(candidates[0][2]) if candidates else None
            row[kind+'airportcandidatescount']=max(0,len(candidates)-1)
            # Match the provided parquet's all-null first slot. Candidate order
            # is deterministic here; the unavailable production order is unknown.
            row['other'+kind+'airportcandidates']=[[None,math.floor(h),math.floor(v)] for _,h,v in candidates[1:]] if candidates else None
        rows.append(row)
    result=pl.DataFrame(rows)
    result.write_parquet(out/'reconstructed_sample_flights.parquet')
    (out/'reconstructed_sample_provenance.json').write_text(json.dumps({'hypothesis':hypothesis,
        'catalog':'ourairports_2024-06-01.parquet','catalog_filter':'fixed-wing types with GPS code and elevation',
        'callsign':'mode of retained positions','alternate_order':'ascending horizontal distance, then GPS code',
        'exact_replica':False,'rows':result.height},indent=2)+'\n')
    print(f'Exported {result.height} approximate sample flights')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hypothesis',help='Name from segmentation_metrics.json')
    export(parser.parse_args().hypothesis)
