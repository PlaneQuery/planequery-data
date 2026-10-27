from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from visualization.adsb_track import flight_events, plot_track


def test_sparse_track_preserves_missing_values_and_breaks_gaps():
    start = datetime(2026, 3, 1)
    d = pl.DataFrame({
        'icao': ['abc123'] * 4,
        'time': [start + timedelta(seconds=s) for s in [0, 5, 120, 125]],
        'baro_altitude_ft': [None, -50, 200, 300],
        'ground_speed_kt': [0., 80., 100., 110.],
        'on_ground': [True, False, None, False],
        'lat': [43.6, 43.61, 43.5, 43.49],
        'lon': [-84.2, -84.19, -83.9, -83.89],
    })
    fig = plot_track(d, icao='ABC123', day=date(2026, 3, 1), show_map=True)
    altitude = fig.axes[0].collections[0].get_offsets()
    assert np.ma.is_masked(altitude[0, 1])  # Ground altitude must not become zero.
    assert altitude[1, 1] == -50
    assert len(fig.axes[0].lines) == 0  # Measurements are points only.
    assert len(fig.axes[1].lines) == 0
    strips = {c.get_gid(): c for c in fig.axes[1].collections if c.get_gid()}
    assert len(strips['on_ground=False'].get_segments()) == 0  # Gap is not filled.
    assert 'on_ground=unknown' not in strips
    assert len(fig.axes[0].patches) == 0  # No shading over missing observations.
    map_axis = next(ax for ax in fig.axes if ax.get_title() == 'Observed positions')
    assert np.isnan(map_axis.lines[0].get_xdata()[2])
    assert np.isnan(map_axis.lines[0].get_ydata()[2])
    assert len(map_axis.collections[0].get_offsets()) == 4  # Every actual position retained.
    plt.close(fig)


def test_annotations_keep_supplied_times_and_unknown_airport():
    start = datetime(2026, 3, 1)
    flights = pl.DataFrame({'icao': ['abc123'], 'takeoff_time': [start],
                           'takeoff_airport_ident': [None], 'landing_time': [None]})
    assert flight_events(flights, 'ABC123', start, start + timedelta(hours=1)) == [
        (start, 'takeoff', 'unknown airport')]


def test_nearby_airports_prioritize_segment_endpoints_over_midtrack():
    start = datetime(2026, 3, 1)
    observations = pl.DataFrame({
        'icao': ['abc123'] * 3,
        'time': [start + timedelta(seconds=s) for s in [0, 5, 10]],
        'baro_altitude_ft': [100., 200., 100.],
        'ground_speed_kt': [80., 90., 80.],
        'on_ground': [False] * 3,
        'lat': [30., 30., 30.], 'lon': [-97., -96.9, -96.8],
    })
    airports = pl.DataFrame({
        'ident': ['END', 'MID'], 'type': ['small_airport'] * 2,
        'latitude_deg': [30.001, 30.], 'longitude_deg': [-97., -96.9],
    })
    fig = plot_track(observations, icao='abc123', day=date(2026, 3, 1),
                     show_map=True, airports=airports, nearby_airport_limit=1)
    labels = [text.get_text() for ax in fig.axes for text in ax.texts]
    assert 'END' in labels
    assert 'MID' not in labels
    plt.close(fig)
    fig = plot_track(observations, icao='abc123', day=date(2026, 3, 1),
                     show_map=True, map_below=True, airports=airports,
                     nearby_airport_limit=1, include_airports=('MID',))
    labels = [text.get_text() for ax in fig.axes for text in ax.texts]
    assert 'MID' in labels
    assert 'END' not in labels
    assert fig.get_figheight() == 8.8
    plt.close(fig)
