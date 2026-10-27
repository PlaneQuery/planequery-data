"""Small, cached OpenStreetMap viewport for paper figures (Web Mercator)."""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import requests

RADIUS = 6378137.0
HALF_WORLD = np.pi * RADIUS


def project(lon, lat):
    """Longitude/latitude degrees to EPSG:3857 meters."""
    return (RADIUS * np.deg2rad(lon),
            RADIUS * np.arcsinh(np.tan(np.deg2rad(np.clip(lat, -85.05, 85.05)))))


def add_basemap(ax, zoom=11, cache=Path('data/cache/map_tiles/osm')):
    """Fetch only tiles intersecting this viewport; reuse cached tiles thereafter.

    Uses an identified client and persistent cache, with visible attribution.
    This is a single figure viewport, not an offline-area download facility.
    """
    if not 0 <= zoom <= 18:
        raise ValueError('Map zoom must be between 0 and 18')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    count = 2**zoom
    tile_width = 2 * HALF_WORLD / count
    x0, x1 = np.floor((np.array([xmin, xmax]) + HALF_WORLD) / tile_width).astype(int)
    y0, y1 = np.floor((HALF_WORLD - np.array([ymax, ymin])) / tile_width).astype(int)
    if (x1-x0+1)*(y1-y0+1) > 64:
        raise ValueError('Map viewport needs over 64 tiles; reduce --map-zoom')
    mosaic = Image.new('RGB', ((x1-x0+1)*256, (y1-y0+1)*256))
    with requests.Session() as session:
        session.headers['User-Agent'] = 'PlaneQuery-PaperFigures/1.0'
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                path = cache / str(zoom) / str(x) / f'{y}.png'
                if not path.exists():
                    response = session.get(f'https://tile.openstreetmap.org/{zoom}/{x}/{y}.png', timeout=30)
                    response.raise_for_status()
                    # Decode before caching so HTML error pages cannot poison it.
                    tile = Image.open(BytesIO(response.content)).convert('RGB')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tile.save(path)
                else:
                    with Image.open(path) as cached:
                        tile = cached.convert('RGB')
                mosaic.paste(tile, ((x-x0)*256, (y-y0)*256))
    extent = [x0*tile_width-HALF_WORLD, (x1+1)*tile_width-HALF_WORLD,
              HALF_WORLD-(y1+1)*tile_width, HALF_WORLD-y0*tile_width]
    ax.imshow(mosaic, extent=extent, interpolation='lanczos', zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.text(.99, .01, '© OpenStreetMap contributors', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 2}, zorder=10)
