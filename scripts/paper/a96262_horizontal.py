"""Reflow the existing paper PDF without changing its data or vector artwork.

Run with the Codex runtime Python (requires pypdf). Coordinates below are
specific to the 518.4 x 720 point source figure.
"""
from copy import deepcopy
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / 'paper/figures'
SOURCE = FIGURES / 'a96262_20260401_adsbx_map.pdf'
STEM = 'a96262_20260401_adsbx_map_horizontal'
source = PdfReader(SOURCE).pages[0]
assert tuple(float(v) for v in source.mediabox) == (0, 0, 518.4, 720)
writer = PdfWriter()
page = writer.add_blank_page(width=1036.8, height=388)


def panel(top, bottom, x, target_top):
    """Place a full-width clipped source strip; arguments use top coordinates."""
    clipped = deepcopy(source)
    bounds = RectangleObject((0, 720 - bottom, 518.4, 720 - top))
    clipped.cropbox = bounds
    clipped.trimbox = bounds
    page.merge_transformed_page(
        clipped,
        Transformation().translate(tx=x, ty=388 - target_top - (720 - top)),
    )


panel(0, 25, 259.2, 0)       # Center the existing title across both panels.
panel(30, 347, 0, 30)        # Features, flight callouts, and ground-state legend.
panel(362, 710, 518.4, 30)   # Original map, airport labels, and time colorbar.
writer.add_metadata({'/Title': 'A96262 · ADS-B Exchange — horizontal',
                     '/Subject': 'Horizontal arrangement of the original paper figure; data and annotations unchanged.'})
with (FIGURES / f'{STEM}.pdf').open('wb') as output:
    writer.write(output)
(FIGURES / f'{STEM}.tex').write_text(
    '\\begin{figure*}[t]\n  \\centering\n'
    f'  \\includegraphics[width=\\textwidth]{{figures/{STEM}.pdf}}\n'
    '  \\caption{ADS-B Exchange observations for A96262 on 1 April 2026. '
    'Left: altitude, ground speed, ground state, and the original flight boundary annotations. '
    'Right: observed positions and airports. Long observation gaps are compressed as in the original figure.}\n'
    '  \\label{fig:a96262-adsbx-horizontal}\n\\end{figure*}\n'
)
print(FIGURES / f'{STEM}.pdf')
