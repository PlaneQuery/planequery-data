"""Reproduce the a96262_20260401_adsbx_map layout using native OpenSky values."""
from datetime import date, datetime
from pathlib import Path
import sys
import json
import hashlib
import os
import argparse

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/opensky-a65c29-native-2026-03-01/mpl-cache'))
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from visualization.adsb_track import plot_track
from visualization.basemap import project

DATA=ROOT/'outputs/opensky-a65c29-native-2026-03-01'
OUT=ROOT/'paper/figures'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--horizontal',action='store_true',help='Place features left and map right; preserve the vertical outputs.')
args=parser.parse_args()
gap_minutes=30 if args.horizontal else 12
STEM='a65c29_20260301_opensky_native_map'+('_horizontal' if args.horizontal else '')
raw=pl.read_parquet(DATA/'state_vectors_data4.parquet')
# Compatibility adapter for the existing plotter, NOT a unit conversion or
# canonical normalization. Values are native metres and metres per second.
# Only the snapshot time is formatted as a datetime; raw epoch values are kept
# in the accompanying native data extract.
observations=raw.select(
    pl.col('icao24').alias('icao'),pl.from_epoch('time',time_unit='s'),
    pl.col('baroaltitude').alias('baro_altitude_ft'),
    pl.col('velocity').alias('ground_speed_kt'),
    pl.col('onground').alias('on_ground'),'lat','lon')
stamp=lambda s:datetime.fromisoformat('2026-03-01T'+s)
flights=pl.DataFrame([
    {'icao':'a65c29','takeoff_time':stamp('10:19:00'),'landing_time':stamp('14:02:00'),'takeoff_airport_ident':'KJFK','landing_airport_ident':'TJSJ'},
    {'icao':'a65c29','takeoff_time':stamp('15:14:00'),'landing_time':stamp('18:42:00'),'takeoff_airport_ident':'TJSJ','landing_airport_ident':'KJFK'}])
# Coordinates from the date-specific reference comparisons used in this audit.
airports=pl.DataFrame({'ident':['KJFK','TJSJ'],'longitude_deg':[-73.779317,-66.001801],'latitude_deg':[40.639447,18.439399]})
fig=plot_track(observations,icao='a65c29',day=date(2026,3,1),flights=flights,
               start=stamp('10:10:00'),end=stamp('18:50:00'),
               title='A65C29 · OpenSky',event_source='BTS flight boundaries',
               show_map=True,basemap=True,map_zoom=4,map_below=not args.horizontal,
               airports=airports,nearby_airport_limit=0,compress_gaps=True,
               compressed_gap_minutes=gap_minutes)
primary,secondary,map_ax,color_ax=fig.axes
if args.horizontal:
    for collection in map_ax.collections:
        if collection.get_gid() == 'observed-map-positions':
            collection.set_edgecolor('none')
            collection.set_sizes([10])
primary.set_ylabel('baroaltitude (m)',color='#0072B2')
secondary.set_ylabel('velocity (m/s)',color='#333333')
primary.set_xlabel('time (UTC), 2026-03-01 · // long gaps compressed')
map_ax.set_xlabel('lon (°)');map_ax.set_ylabel('lat (°)')
color_ax.set_xlabel('time (UTC)')
for text in fig.legends[0].get_texts():
    text.set_text(text.get_text().replace('on_ground','onground'))
fig.text(.305 if args.horizontal else .5,.095 if args.horizontal else .505,
         'Published OpenSky: one merged flight, 10:19:07–18:42:25 UTC.',
         ha='center',fontsize=7,color='.35',fontfamily='serif')
# Keep all native positions, including the isolated erroneous-looking jump,
# but never imply a physical flight path by connecting it to New York.
map_ax.lines[0].remove()
outlier=raw.filter(pl.col('lat')>43)
if outlier.height:
    o=outlier.row(0,named=True);xx,yy=project(o['lon'],o['lat'])
    when=datetime.fromtimestamp(o['time'],__import__('datetime').timezone.utc)
    map_ax.annotate(f'Isolated raw position\n{when:%H:%M:%S} UTC',xy=(xx,yy),
                    xytext=(8,8),textcoords='offset points',fontsize=7,fontfamily='serif',
                    bbox=dict(facecolor='white',edgecolor='.65',pad=2),
                    arrowprops=dict(arrowstyle='-',lw=.6,color='.35'),zorder=8)
# The reference's broken time axis can crowd ticks when there are several gaps.
fig.canvas.draw()
renderer=fig.canvas.get_renderer();kept=[];right=-float('inf')
for tick,label in zip(primary.get_xticks(),primary.get_xticklabels()):
    bounds=label.get_window_extent(renderer)
    if bounds.x0>=right+7:
        kept.append(tick);right=bounds.x1
primary.xaxis.set_major_locator(FixedLocator(kept))
for ext in ['pdf','svg','png']:
    p=OUT/f'{STEM}.{ext}';fig.savefig(p,dpi=300,facecolor='white');print(p,flush=True)
plt.close(fig)
manifest={'icao24':'a65c29','date':'2026-03-01','native_source':str(DATA/'state_vectors_data4.parquet'),
          'layout':'horizontal' if args.horizontal else 'vertical',
          'native_source_sha256':hashlib.sha256((DATA/'state_vectors_data4.parquet').read_bytes()).hexdigest(),
          'native_fields_and_units':{'baroaltitude':'m','velocity':'m/s','onground':'boolean','time':'Unix seconds; UTC labels','lat':'degrees','lon':'degrees'},
          'normalization':'No unit conversion, freshness filter, deduplication, or ground-state transformation. Native snapshot values retained.',
          'internal_adapter':'Existing plotter receives aliases for compatibility; displayed field names and units are native.',
          'display_interval':['2026-03-01T10:10:00Z','2026-03-01T18:50:00Z'],
          'event_source':'BTS v16 cached comparison; SFDPS v14 confirms two flights',
          'compression':f'Snapshot-free gaps >30 minutes occupy {gap_minutes} minutes of axis width, marked //; actual UTC retained.',
          'basemap':'OpenStreetMap tiles z4; current geography; cached locally',
          'map_rendering':'Native points only: no connecting lines, preserving the isolated raw position at 18:22:44 without implying a physical excursion.',
          'reference_style':'/Users/jonahgoode/Documents/PlaneQuery/paper-adsb-flight-derivation/figures/a96262_20260401_adsbx_map.pdf'}
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2))
latex=r'''\begin{figure*}[t]
  \centering
  \includegraphics[width=0.72\textwidth]{figures/a65c29_20260301_opensky_native_map.pdf}
  \caption{Native OpenSky state vectors for A65C29 on 1 March 2026. The upper panel shows \texttt{baroaltitude} (metres), \texttt{velocity} (metres per second), and the native \texttt{onground} flag. Dotted lines mark BTS boundaries for KJFK--TJSJ and TJSJ--KJFK. OpenSky instead reports one merged 10:19:07--18:42:25 flight. Snapshot-free gaps exceeding 30 minutes are compressed and marked //; timestamps remain UTC. The map colors observed positions by native snapshot \texttt{time}; missing coverage is not interpolated. Basemap: OpenStreetMap contributors.}
  \label{fig:a65c29-opensky-native-map}
\end{figure*}
'''
if args.horizontal:
    latex=latex.replace('are compressed and marked //','are displayed with 30 minutes of axis width and marked //')
    latex=latex.replace('width=0.72\\textwidth','width=\\textwidth').replace(
        'figures/a65c29_20260301_opensky_native_map.pdf',f'figures/{STEM}.pdf').replace(
        'The upper panel','The left panel').replace(
        'fig:a65c29-opensky-native-map','fig:a65c29-opensky-native-map-horizontal')
(OUT/f'{STEM}.tex').write_text(latex)
