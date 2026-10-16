"""Test endpoint distance formulas, separating airport metadata differences.

Airport coordinate calibration uses only training flights; residuals on the
held-out flights test whether a fixed airport point explains stored distances.
This is a diagnostic, not an independent replacement airport database.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.optimize import least_squares

from analyze_endpoints import haversine, quantiles, select_endpoint_states
from extract import OUT


def calibrate(out=OUT):
    flights = pl.read_parquet(out/'reference_flights.parquet')
    probes = select_endpoint_states(pl.read_parquet(sorted((out/'endpoint_hours').glob('*.parquet'))))
    airports = pl.read_parquet(out/'ourairports_2024-06-01.parquet')
    observations = []
    for kind in ['departure','arrival']:
        prefix='est'+kind+'airport'
        f=flights.select('flight_id',pl.col(prefix).alias('airport'),
                         pl.col(prefix+'horizdistance').alias('horizontal'),
                         pl.col(prefix+'vertdistance').alias('vertical'))
        x=probes.filter(pl.col('kind')==kind).join(f,on='flight_id').drop_nulls(['airport','lat','lon','baroaltitude'])
        observations.append(x.select('flight_id','kind','airport','lat','lon','baroaltitude','horizontal','vertical'))
    data=pl.concat(observations).join(airports.select(pl.col('ident').alias('airport'),'latitude_deg','longitude_deg','elevation_ft'),on='airport',how='left')
    results=[]
    tests=[]
    for (airport,),x in data.drop_nulls('latitude_deg').partition_by('airport',as_dict=True).items():
        # An aircraft's arrival and departure from the same flight stay together.
        train=x.filter(pl.col('flight_id')%3!=0)
        test=x.filter(pl.col('flight_id')%3==0)
        if train.height<6 or test.height<2:
            continue
        lat0,lon0,elevation=x.select('latitude_deg','longitude_deg','elevation_ft').row(0)
        if elevation is None:
            continue
        lat,lon,h=train['lat'].to_numpy(),train['lon'].to_numpy(),train['horizontal'].to_numpy()
        def residual(delta):
            return haversine(lat,lon,np.full(len(lat),lat0+delta[0]/111195),np.full(len(lat),lon0+delta[1]/(111195*np.cos(np.radians(lat0)))))-h-.5
        fit=least_squares(residual,[0.,0.],loss='soft_l1',f_scale=2,max_nfev=150)
        alat,alon=lat0+fit.x[0]/111195,lon0+fit.x[1]/(111195*np.cos(np.radians(lat0)))
        alt=train['baroaltitude'].to_numpy()
        vd=train['vertical'].to_numpy()+.5
        inferred_elevation=np.median(alt-np.sign(alt-elevation*.3048)*vd)
        result={'airport':airport,'train':train.height,'test':test.height,'latitude':float(alat),'longitude':float(alon),
                'elevation_m':float(inferred_elevation),'coordinate_shift_m':float(np.linalg.norm(fit.x)),
                'train_median_abs_error':float(np.median(np.abs(residual(fit.x))))}
        d=haversine(test['lat'].to_numpy(),test['lon'].to_numpy(),np.full(test.height,alat),np.full(test.height,alon))
        v=np.abs(test['baroaltitude'].to_numpy()-inferred_elevation)
        result['test_median_abs_error']=float(np.median(np.abs(d-test['horizontal'].to_numpy()-.5)))
        results.append(result)
        tests.append(test.with_columns(pl.Series('horizontal_error_calibrated',d-test['horizontal'].to_numpy()),
                                       pl.Series('vertical_error_calibrated',v-test['vertical'].to_numpy())))
    if not results:
        print('Insufficient airport observations')
        return
    pl.DataFrame(results).write_parquet(out/'calibrated_airports.parquet')
    test=pl.concat(tests)
    test.write_parquet(out/'airport_geometry_holdout.parquet')
    report={'airports':len(results),'holdout_observations':test.height,
            'horizontal_errors_m':quantiles(test['horizontal_error_calibrated']),
            'vertical_errors_m':quantiles(test['vertical_error_calibrated']),
            'horizontal_within_1m':int(((test['horizontal_error_calibrated']>=-.5)&(test['horizontal_error_calibrated']<1.5)).sum()),
            'horizontal_within_2m':int((test['horizontal_error_calibrated'].abs()<2).sum())}
    (out/'airport_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    calibrate()
