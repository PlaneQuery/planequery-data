"""Paper figures for the native OpenSky A65C29 JFK--San Juan round trip.

Run from the repository root. Reads the saved native Trino extract, never the
canonical ADS-B cache. Exports vector PDF/SVG, 300 dpi PNG, and a LaTeX snippet.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import polars as pl
from shapely.geometry import shape, box
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs/opensky-a65c29-native-2026-03-01'
OUT = ROOT / 'paper/figures'
STEM = 'a65c29_20260301_opensky_native'
BLUE, ORANGE, GREEN, PINK = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
STYLE = {'font.family':'serif','font.size':8,'axes.labelsize':8,
         'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
         'legend.fontsize':7,'axes.linewidth':.6,'pdf.fonttype':42,
         'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
         'axes.spines.right':False}

def utc(s):
    return datetime.fromisoformat('2026-03-01T'+s).replace(tzinfo=timezone.utc)

START, END = utc('10:19:00'), utc('18:43:00')
GAP0, GAP1 = utc('14:01:58.300'), utc('15:16:07.609')
raw = pl.read_parquet(DATA/'state_vectors_data4.parquet')
df = raw.filter(pl.col('time').is_between(START.timestamp(), END.timestamp(),closed='left')).sort('time')
t = mdates.date2num([datetime.fromtimestamp(v,timezone.utc) for v in df['time']])
world = json.loads((DATA/'countries-50m.geojson').read_text())
groups = [(name,col,df.filter(pl.col('callsign').str.strip_chars()==name)) for name,col in [('JBU1603',BLUE),('JBU904',ORANGE)]]

def base_map(ax, extent):
    west,east,south,north=extent
    region=box(west,south,east,north)
    for f in world['features']:
        geom=shape(f['geometry'])
        if not geom.intersects(region):continue
        clipped=geom.intersection(region)
        polys=[clipped] if clipped.geom_type=='Polygon' else list(clipped.geoms) if clipped.geom_type=='MultiPolygon' else []
        for poly in polys:
            ax.add_patch(Polygon(np.asarray(poly.exterior.coords),facecolor='#eef0e9',edgecolor='#a4aaa4',lw=.4,zorder=0))
    ax.set(xlim=(west,east),ylim=(south,north),xlabel='lon (°)',ylabel='lat (°)')
    ax.set_aspect(1/np.cos(np.deg2rad((south+north)/2)))
    ax.set_facecolor('#f7fafc')
    ax.grid(color='.88',lw=.35,zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    for name,col,track in groups:
        p=track.drop_nulls(['lat','lon','lastposupdate']).unique('lastposupdate',maintain_order=True).sort('lastposupdate')
        coords=p.select('lon','lat').to_numpy()
        gaps=np.diff(p['lastposupdate'].to_numpy())
        segs=np.stack([coords[:-1],coords[1:]],axis=1)
        ax.add_collection(LineCollection(segs[gaps<=300],colors=col,linewidths=.65,zorder=2))
        ax.scatter(coords[:,0],coords[:,1],s=1.2,c=col,linewidths=0,zorder=3)
    for name,x,y in [('KJFK',-73.779317,40.639447),('TJSJ',-66.001801,18.439399)]:
        if west<x<east and south<y<north:
            ax.plot(x,y,marker='*',ms=7,color='#25422e',mec='white',mew=.4,zorder=5)
            offset=(6,5) if name=='KJFK' else ((-24,7) if north>20 else (6,-11))
            ax.annotate(name,(x,y),xytext=offset,textcoords='offset points',fontsize=8,
                        bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=6)

def timeline(ax):
    for source,y,segments in [('BTS',2,[('10:19:00','14:02:00',BLUE),('15:14:00','18:42:00',ORANGE)]),
                               ('SFDPS',1,[('10:19:00','14:03:00',BLUE),('15:14:00','18:45:00',ORANGE)]),
                               ('OpenSky',0,[('10:19:07','18:42:25','#666666')])]:
        for a,b,col in segments:ax.plot([utc(a),utc(b)],[y,y],color=col,lw=4,solid_capstyle='butt')
    ax.set(yticks=[0,1,2],yticklabels=['OpenSky','SFDPS','BTS'],ylim=(-.6,2.7))
    ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='both',length=0)

def scatter(ax,field,color,marker='.',size=1.1):
    y=np.array([np.nan if v is None else float(v) for v in df[field]])
    ax.plot(t,y,linestyle='none',marker=marker,ms=size,color=color,mew=.3,label=field)

def features(axes, bounds, detail=False):
    timeline(axes[0])
    scatter(axes[1],'geoaltitude','#888888',size=1.3)
    scatter(axes[1],'baroaltitude',BLUE,size=1.2)
    axes[1].set_ylabel('altitude (m)')
    axes[1].legend(handles=[Line2D([],[],color=BLUE,lw=1,label='baroaltitude'),Line2D([],[],color='#888888',lw=1,label='geoaltitude')],loc='upper left',frameon=False,ncol=2,handlelength=1.4)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
    scatter(axes[2],'velocity','#333333');axes[2].set_ylabel('velocity\n(m/s)')
    scatter(axes[3],'vertrate','#777777');axes[3].set_ylabel('vertrate\n(m/s)');axes[3].axhline(0,color='.75',lw=.45)
    scatter(axes[4],'heading','#605481');axes[4].set_ylabel('heading\n(°)');axes[4].set_yticks([0,180,360])
    ground=df['onground'].to_numpy()
    axes[5].plot(t[~ground],np.zeros((~ground).sum()),'|',color=BLUE,ms=3,mew=.6)
    axes[5].plot(t[ground],np.ones(ground.sum()),'|',color=ORANGE,ms=3,mew=.6)
    axes[5].set(yticks=[0,1],yticklabels=['false','true'],ylim=(-.25,1.25),ylabel='onground')
    axes[5].set_ylabel('onground',rotation=0,ha='right',va='center',labelpad=14,fontsize=7)
    axes[6].plot(t,df['time'].to_numpy()-df['lastposupdate'].to_numpy(),'.',color='#333333',ms=1,label='time − lastposupdate')
    axes[6].plot(t,df['time'].to_numpy()-df['lastcontact'].to_numpy(),'.',color=GREEN,ms=.8,label='time − lastcontact')
    axes[6].set_ylabel('age (s)\n[derived]');axes[6].set_yscale('symlog',linthresh=1);axes[6].set_yticks([0,10,100]);axes[6].set_yticklabels(['0','10','100'])
    axes[6].legend(loc='upper center',ncol=2,frameon=False,markerscale=3,handletextpad=.3)
    for i,ax in enumerate(axes):
        ax.set_xlim(bounds)
        ax.tick_params(axis='x',labelbottom=i==len(axes)-1,length=2.5,width=.6)
        if i:
            ax.grid(axis='y',color='.91',lw=.45)
            ax.axvspan(GAP0,GAP1,color='.9',zorder=-1)
            for v,col in [('14:02:00',PINK),('15:14:00',GREEN)]:ax.axvline(utc(v),color=col,ls=':',lw=.75,zorder=1)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30]) if detail else mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('time (UTC), 2026-03-01')
    if detail:
        visible=(t>=mdates.date2num(bounds[0])) & (t<=mdates.date2num(bounds[1]))
        ymax=max(np.nanmax(df[c].to_numpy()[visible]) for c in ['baroaltitude','geoaltitude'])
        axes[1].set_ylim(-100,ymax*1.12)
        axes[1].text(.48,.66,'No fresh position updates\n14:01:58–15:16:07\n(74.2 min)',transform=axes[1].transAxes,ha='center',va='center',fontsize=8,color='.35')
    else:
        axes[1].text(mdates.date2num(utc('14:39:00')),.56,'74.2 min\nposition gap',transform=axes[1].get_xaxis_transform(),ha='center',fontsize=7,color='.35')

def save(fig,stem):
    for ext in ['pdf','svg','png']:
        p=OUT/f'{stem}.{ext}';fig.savefig(p,dpi=300,facecolor='white');print(p)
    plt.close(fig)

OUT.mkdir(parents=True,exist_ok=True)
with plt.rc_context(STYLE):
    fig=plt.figure(figsize=(7.2,7.6))
    gs=fig.add_gridspec(8,2,left=.15,right=.97,bottom=.105,top=.92,
                        height_ratios=[2.0,.55,1.0,.55,.55,.55,.38,.62],hspace=.38,wspace=.35)
    m0=fig.add_subplot(gs[0,0]);m1=fig.add_subplot(gs[0,1])
    base_map(m0,(-79,-62,17,42));base_map(m1,(-66.40,-65.65,18.22,18.90))
    for map_ax in (m0,m1):
        pos=map_ax.get_position()
        map_ax.set_position([pos.x0,pos.y0+.028,pos.width,pos.height-.028])
    m0.set_title('(a) Observed route',loc='left');m1.set_title('(b) San Juan',loc='left')
    m0.set_xlabel('lon (°)',labelpad=0);m1.set_xlabel('lon (°)',labelpad=0)
    axes=[]
    for i in range(1,8):axes.append(fig.add_subplot(gs[i,:],sharex=axes[0] if axes else None))
    features(axes,(START,END))
    fig.suptitle('A65C29 · OpenSky native state vectors · 1 March 2026',fontsize=11,y=.985)
    fig.legend(handles=[Line2D([],[],color=BLUE,lw=1.2,label='JBU1603: KJFK → TJSJ'),Line2D([],[],color=ORANGE,lw=1.2,label='JBU904: TJSJ → KJFK')],loc='upper center',bbox_to_anchor=(.53,.965),ncol=2,frameon=False)
    fig.text(.13,.018,'Native snapshot values; gray band: no fresh position updates. Dotted lines: BTS TJSJ landing / takeoff.\nMap: Natural Earth 1:50m. SFDPS return arrival is 18:45 (outside the displayed interval).',fontsize=6.5,color='.35')
    save(fig,STEM)
    fig,axes=plt.subplots(7,1,figsize=(7.2,5.6),sharex=True,gridspec_kw={'height_ratios':[.55,1.2,.6,.6,.6,.4,.65]})
    fig.subplots_adjust(left=.15,right=.97,bottom=.105,top=.87,hspace=.25)
    features(axes,(utc('13:40:00'),utc('15:40:00')),detail=True)
    fig.suptitle('A65C29 · San Juan turnaround · native OpenSky fields',fontsize=11,y=.975)
    fig.text(.14,.915,'BTS: landing 14:02 / takeoff 15:14; SFDPS: landing 14:03 / takeoff 15:14.',fontsize=8)
    fig.text(.14,.018,'Raw onground is false throughout this window. Carried-forward snapshots remain visible; no interpolation.',fontsize=7,color='.35')
    save(fig,STEM+'_turnaround')

caption=(r'Native OpenSky state vectors for aircraft \texttt{a65c29} on 1 March 2026. '
 r'BTS reports JBU1603 (KJFK--TJSJ, 10:19--14:02 UTC) and JBU904 (TJSJ--KJFK, 15:14--18:42 UTC); '
 r'SFDPS confirms the two legs, with arrivals at 14:03 and 18:45. Published OpenSky flights merge them into '
 r'one 10:19:07--18:42:25 interval. Maps show observed positions without connecting updates separated by more than five minutes. '
 r'Feature panels retain native field values and units at snapshot \texttt{time}, including carried-forward states. '
 r'The gray interval spans 74.2 minutes between fresh position updates. Timestamp ages are derived quantities. '
 r'Raw \texttt{onground} is false throughout the turnaround; missing updates are not evidence that the aircraft stayed airborne.')
(OUT/f'{STEM}.tex').write_text('\\begin{figure*}[t]\n  \\centering\n  \\includegraphics[width=\\textwidth]{figures/'+STEM+'.pdf}\n  \\caption{'+caption+'}\n  \\label{fig:opensky-a65c29-native}\n\\end{figure*}\n')
manifest={'native_source':str(DATA/'state_vectors_data4.parquet'),'native_source_sha256':hashlib.sha256((DATA/'state_vectors_data4.parquet').read_bytes()).hexdigest(),'native_rows_plotted':df.height,'fields':df.columns,'snapshot_interval':[START.isoformat(),END.isoformat()],'native_units':{'baroaltitude':'m','geoaltitude':'m','velocity':'m/s','vertrate':'m/s','heading':'degrees'},'transformations':['UTC formatting of native time','map duplicate lastposupdate removal','derived time minus lastposupdate/lastcontact ages'],'no_canonical_fields_used':True,'gap_seconds':(GAP1-GAP0).total_seconds(),'query_provenance':json.loads((DATA/'query.json').read_text()),'basemap':'Natural Earth 1:50m admin_0_countries, public domain','raw_correction':'No onground=true native snapshots during the turnaround. Prior normalized artifact disagreed; its cause was not established.'}
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2))
