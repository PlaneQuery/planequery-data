"""Test the inferred airport gates against the available historical catalog."""

import json
import argparse

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from analyze_endpoints import RADIUS, select_endpoint_states
from extract import OUT


def xyz(lat, lon):
    lat,lon=np.radians(lat),np.radians(lon)
    return np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))


def run(out=OUT, catalog='ourairports_2024-06-01.parquet'):
    airports=pl.read_parquet(out/catalog).drop_nulls(['latitude_deg','longitude_deg','elevation_ft'])
    f=pl.read_parquet(out/'reference_flights.parquet')
    e=select_endpoint_states(pl.read_parquet(sorted((out/'endpoint_hours').glob('*.parquet'))))
    metrics=[]
    for policy in ['all','gps_code','fixed_wing','fixed_wing_gps']:
        a=airports
        if 'gps' in policy:
            a=a.filter(pl.col('gps_code').is_not_null())
        if 'fixed_wing' in policy:
            a=a.filter(pl.col('type').is_in(['small_airport','medium_airport','large_airport']))
        tree=cKDTree(xyz(a['latitude_deg'].to_numpy(),a['longitude_deg'].to_numpy()))
        elevation=a['elevation_ft'].to_numpy()*.3048
        aliases=a.select('ident','gps_code','icao_code').to_numpy()
        for kind,hlimit,vlimit in [('departure',15000,2000),('arrival',30000,2500)]:
            prefix='est'+kind+'airport'
            x=e.filter(pl.col('kind')==kind).join(f.select('flight_id',prefix,kind+'airportcandidatescount'),on='flight_id').drop_nulls(['lat','lon','baroaltitude'])
            predicted=[];counts=[]
            for batch in x.iter_slices(2000):
                positions=xyz(batch['lat'].to_numpy(),batch['lon'].to_numpy())
                neighbors=tree.query_ball_point(positions,2*np.sin(hlimit/(2*RADIUS)))
                for row,(ids,alt) in enumerate(zip(neighbors,batch['baroaltitude'].to_numpy())):
                    ids=np.asarray(ids,dtype=int)
                    ids=ids[np.abs(elevation[ids]-alt)<vlimit]
                    counts.append(max(0,len(ids)-1))
                    if not len(ids):
                        predicted.append(None)
                    else:
                        chord=np.linalg.norm(tree.data[ids]-positions[row],axis=1)
                        predicted.append(int(ids[np.argmin(chord)]))
            correct=nonnull_correct=null_correct=0
            expected=x[prefix].to_list()
            for idx,code in zip(predicted,expected):
                hit=code is None if idx is None else code is not None and code in aliases[idx]
                correct+=hit
                nonnull_correct+=hit and code is not None
                null_correct+=hit and code is None
            result={'policy':policy,'catalog_airports':a.height,'kind':kind,'observations':x.height,
                    'airport_matches_including_null':correct,'nonnull_matches':nonnull_correct,
                    'nonnull_reference':x.height-x[prefix].null_count(),'null_matches':null_correct,
                    'null_reference':x[prefix].null_count(),
                    'candidate_counts_match':int((np.asarray(counts)==x[kind+'airportcandidatescount'].to_numpy()).sum())}
            metrics.append(result)
    filename='airport_candidate_metrics.json' if catalog=='ourairports_2024-06-01.parquet' else f'airport_candidate_metrics_{catalog.removesuffix(".parquet")}.json'
    (out/filename).write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',default='ourairports_2024-06-01.parquet')
    args=parser.parse_args()
    run(catalog=args.catalog)
