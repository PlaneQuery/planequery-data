"""Small synthetic checks for the research implementation's inferred rules."""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'opensky_reverse_engineering'))
from segments import DAY, Rules, clean, reconstruct, evaluate  # noqa: E402
from analyze_endpoints import select_endpoint_states  # noqa: E402
from replica import AirportIndex  # noqa: E402
from analyze_endpoints import RADIUS  # noqa: E402
from daily_replay import infer  # noqa: E402
from extract import extract  # noqa: E402


def states(seconds, longitude=None):
    seconds=np.asarray(seconds)
    n=len(seconds)
    if longitude is None:
        longitude=seconds*100/111195
    return pl.DataFrame({'time':DAY+seconds+1,'lastposupdate':DAY+seconds+.25,
        'lat':np.zeros(n),'lon':longitude,'baroaltitude':np.full(n,1000.),
        'velocity':np.full(n,100.),'heading':np.full(n,90.),'vertrate':np.zeros(n),
        'onground':np.zeros(n,dtype=bool),'callsign':['TEST123 ']*n})


def test_long_gap_merges_when_displacement_matches_typical_speed():
    t=np.r_[np.arange(21),np.arange(1021,1042)]
    rules=Rules('test',clock='position',split='median_quarter',min_duration=0)
    assert len(reconstruct('abcdef',states(t),rules))==1
    landed=states(t,np.r_[np.arange(21),np.arange(20,41)]*100/111195)
    assert len(reconstruct('abcdef',landed,rules))==2


def test_strict_minimum_duration_and_position_timestamp():
    rules=Rules('test',clock='position')
    assert reconstruct('abcdef',states(np.arange(601)),rules)==[]
    result=reconstruct('abcdef',states(np.arange(602)),rules)
    assert (result[0]['firstseen'],result[0]['lastseen'])==(DAY,DAY+601)


def test_exact_600_second_gap_does_not_split():
    t=np.r_[np.arange(21),np.arange(620,641)]
    landed=states(t,np.r_[np.arange(21),np.arange(20,41)]*100/111195)
    assert len(reconstruct('abcdef',landed,Rules('test',clock='position',split='median_quarter',min_duration=0)))==1


def test_sparse_filter_uses_epoch_aligned_position_time_bins():
    x=clean(states([0,1,2,30,31,32,33]),Rules('test',clock='position',sparse=True))
    assert list(x['clock'].astype(int)-DAY)==[30,31,32,33]


def test_median_removes_isolated_vertical_outlier():
    s=states(np.arange(21)).with_columns(
        pl.when(pl.col('time')==DAY+11).then(5000.).otherwise(pl.col('baroaltitude')).alias('baroaltitude'))
    x=clean(s,Rules('test',clock='position',median=True))
    assert DAY+10 not in x['clock'].astype(int)
    assert len(x['clock'])==20


def test_stale_positions_and_ground_flag_option():
    s=states([0,1,2]).with_columns(pl.Series('lastposupdate',[DAY+.25,DAY-.25,DAY+2.25]),
                                 pl.Series('onground',[False,False,True]))
    x=clean(s,Rules('test',clock='position',ground=True))
    assert list(x['clock'].astype(int)-DAY)==[0,2]


def test_endpoint_integer_second_exception_uses_valid_earlier_snapshot():
    probes=pl.DataFrame({'flight_id':[1,1],'kind':['arrival','arrival'],
        'time':[DAY,DAY+1],'endpoint_time':[DAY,DAY],'lastposupdate':[float(DAY),DAY+.5],
        'lat':[1.,1.1],'lon':[2.,2.1],'baroaltitude':[10.,None]})
    selected=select_endpoint_states(probes)
    assert selected.height==1
    assert selected['time'].item()==DAY


def test_airports_rank_horizontal_distance_after_vertical_gate():
    catalog=pl.DataFrame({'gps_code':['NEAR','LEVEL','HIGH'],'latitude_deg':[0.,0.,0.],
        'longitude_deg':np.degrees(np.array([1000.,1100.,0.])/RADIUS),
        'elevation_ft':[0.,1900/.3048,10000/.3048],'type':['small_airport']*3})
    candidates=AirportIndex(catalog).candidates(0.,0.,1900.,'departure')
    assert [x[0] for x in candidates]==['NEAR','LEVEL']


def test_airport_search_radius_is_asymmetric():
    catalog=pl.DataFrame({'gps_code':['DISTANT'],'latitude_deg':[0.],
        'longitude_deg':[float(np.degrees(20000/RADIUS))],
        'elevation_ft':[0.],'type':['small_airport']})
    index=AirportIndex(catalog)
    assert index.candidates(0.,0.,100.,'departure')==[]
    assert index.candidates(0.,0.,100.,'arrival')[0][0]=='DISTANT'


def test_window_edges_are_removed_before_outlier_rejection():
    s=states(np.arange(30)).with_columns(
        pl.when(pl.col('time')==DAY+26).then(9000.).otherwise(pl.col('baroaltitude')).alias('baroaltitude'))
    before=clean(s,Rules('test',clock='position',median=True,trim_start=3,trim_end=6,trim_before_filter=True))
    after=clean(s,Rules('test',clock='position',median=True,trim_start=3,trim_end=6))
    assert int(before['clock'][-1])==DAY+23
    assert int(after['clock'][-1])==DAY+22


def test_pending_track_retains_tail_for_next_daily_batch():
    catalog=pl.DataFrame({'gps_code':['TEST'],'latitude_deg':[0.],'longitude_deg':[0.],
                         'elevation_ft':[0.],'type':['small_airport']})
    # An aircraft disappears at cruising altitude, then returns close to its
    # previous location the next day. The old segment is finalized by the gap.
    t=np.r_[np.arange(-3000,-1999),np.arange(2000,3001)]
    lon=np.r_[np.arange(1001),np.arange(1000,2001)]*100/111195
    s=states(t,lon).with_columns(pl.lit(10000.).alias('baroaltitude'))
    flights=infer('abcdef',s,AirportIndex(catalog),carry_raw=True)
    assert len(flights)==1
    assert flights[0]['firstseen']==DAY-2994
    assert flights[0]['lastseen']==DAY-2000


def test_evaluation_retains_reference_aircraft_with_no_prediction():
    predictions=pl.DataFrame({'icao24':['aaaaaa'],'firstseen':[DAY],'lastseen':[DAY+1000]})
    reference=pl.DataFrame({'flight_id':[0,1],'icao24':['aaaaaa','bbbbbb'],
                            'firstseen':[DAY,DAY],'lastseen':[DAY+1000,DAY+2000]})
    metrics,pairs=evaluate(predictions,reference)
    assert metrics['exact_pairs']==1
    assert metrics['reference']==2
    assert pairs.height==2
    assert pairs.filter(pl.col('icao24')=='bbbbbb')['firstseen_pred'].item() is None


def test_replay_is_invariant_to_changing_utc_date():
    catalog=pl.DataFrame({'gps_code':['TEST'],'latitude_deg':[0.],'longitude_deg':[0.],
                         'elevation_ft':[0.],'type':['small_airport']})
    t=np.r_[np.arange(-1000,0),np.arange(0,1001)]
    s=states(t)
    index=AirportIndex(catalog)
    original=infer('abcdef',s,index)
    target=int(datetime(2026,3,1,tzinfo=timezone.utc).timestamp())
    shift=target-DAY
    moved=s.with_columns(pl.col('time')+shift,pl.col('lastposupdate')+shift)
    shifted=infer('abcdef',moved,index,target_day=target)
    assert len(original)==len(shifted)==1
    expected=dict(original[0],firstseen=original[0]['firstseen']+shift,lastseen=original[0]['lastseen']+shift)
    assert shifted[0]==expected
    pred=pl.DataFrame(shifted)
    reference=pred.with_row_index('flight_id')
    metrics,_=evaluate(pred,reference,day=target)
    assert metrics['exact_pairs']==1
    assert metrics['predicted']==1


def test_extraction_selects_requested_date_across_year_boundary(tmp_path):
    target=date(2026,1,1)
    epoch=int(datetime(2026,1,1,tzinfo=timezone.utc).timestamp())
    flight_path=tmp_path/'raw/flights_data4/year=2026/month=01/day=01/data.parquet'
    flight_path.parent.mkdir(parents=True)
    pl.DataFrame({'icao24':['abcdef'],'firstseen':[epoch+10],'lastseen':[epoch+700]}).write_parquet(flight_path)
    for offset in [-1,0,1]:
        day=target+timedelta(days=offset)
        for hour in range(24):
            path=tmp_path/'raw/state_vectors_data4'/day.strftime('year=%Y/month=%m/day=%d')/f'hour={hour:02d}/data.parquet'
            path.parent.mkdir(parents=True)
            time=epoch+offset*86400+hour*3600+10
            s=states([0]).with_columns(pl.lit(time).alias('time'),pl.lit(float(time)-.25).alias('lastposupdate'),
                pl.lit('abcdef').alias('icao24'),pl.lit(1000.).alias('geoaltitude'),pl.lit(float(time)-.25).alias('lastcontact'))
            s.write_parquet(path)
    out=tmp_path/'output'
    extract(tmp_path/'raw',out,sample_size=1,only_hours=['2026-01-01-00'],target_date=target)
    cached=pl.read_parquet(out/'sample_hours/2026-01-01-00.parquet')
    assert cached.height==1
    assert cached['time'].item()==epoch+10
    assert len(list((out/'sample_hours').glob('*.parquet')))==1
