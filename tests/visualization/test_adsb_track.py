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
    })
    fig = plot_track(d, icao='ABC123', day=date(2026, 3, 1))
    altitude = fig.axes[0].lines[0].get_ydata()
    assert np.isnan(altitude[0])  # Ground altitude must not become zero.
    assert altitude[1] == -50
    assert np.isnan(altitude[2])  # No gap bridge.
    strips = {c.get_gid(): c for c in fig.axes[1].collections if c.get_gid()}
    assert len(strips['on_ground=False'].get_segments()) == 0  # Gap is not filled.
    assert 'on_ground=unknown' not in strips
    assert len(fig.axes[0].patches) == 0  # No shading over missing observations.
    plt.close(fig)


def test_annotations_keep_supplied_times_and_unknown_airport():
    start = datetime(2026, 3, 1)
    flights = pl.DataFrame({'icao': ['abc123'], 'takeoff_time': [start],
                           'takeoff_airport_ident': [None], 'landing_time': [None]})
    assert flight_events(flights, 'ABC123', start, start + timedelta(hours=1)) == [
        (start, 'takeoff', 'unknown airport')]
