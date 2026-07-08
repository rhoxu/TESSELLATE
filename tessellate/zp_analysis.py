"""
Evaluate PSF-calibration zeropoints across the TESS focal plane and over time.

The calibration writes one psf_calibration_zp.csv per cut
(<data_path>/Sector{S}/Cam{C}/Ccd{D}/Cut{K}of{n^2}/calibration/), each with
zp_ab, e_zp_ab, zp_scatter, n_stars.  This module collects them and plots:

  collect_zeropoints() : gather all per-cut zeropoints into one table.
  zp_focal_plane()     : ZP laid out in the physical TESS focal-plane geometry
                         (4 cameras stacked, each a 2x2 CCD block, correct
                         orientations) -- median per CCD or the full per-cut image.
  zp_cut_map()         : per-cut ZP map within one CCD (spatial across the cut grid).
  zp_vs_time()         : ZP vs sector (sectors are time-ordered) per cam/ccd.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Render with LaTeX + a serif font.
_TEX_RC = {'text.usetex': True, 'font.family': 'serif'}


def _tex(s):
    """Escape a plain string so it is safe inside a usetex label."""
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_')


# Pretty math labels for the calibration columns (AB etc. as subscripts).
_VALUE_LABELS = {
    'zp_ab': r'$\mathrm{zp}_{\mathrm{AB}}$',
    'e_zp_ab': r'$\sigma_{\mathrm{zp,\,AB}}$',
    'zp_scatter': r'zp scatter',
    'n_stars': r'$N_{\mathrm{stars}}$',
}


def _value_label(value):
    """Nicely typeset a value/column name; fall back to escaping underscores."""
    return _VALUE_LABELS.get(value, _tex(value))

_ZP_RE = re.compile(r'Sector(\d+)/Cam(\d+)/Ccd(\d+)/Cut(\d+)of(\d+)/'
                    r'calibration/psf_calibration_zp\.csv$')


def collect_zeropoints(data_path='/fred/oz335/TESSdata', sectors=None):
    """
    Gather every per-cut psf_calibration_zp.csv under data_path.

    sectors : optional int or iterable of sectors to restrict to.
    Returns a DataFrame with sector, cam, ccd, cut, n (grid), and the zeropoint
    columns (zp_ab, e_zp_ab, zp_scatter, n_stars).
    """
    if np.isscalar(sectors):
        sectors = [int(sectors)]
    files = glob.glob(os.path.join(data_path, '**', 'calibration',
                                   'psf_calibration_zp.csv'), recursive=True)
    rows = []
    for f in files:
        m = _ZP_RE.search(f.replace(os.sep, '/'))
        if not m:
            continue
        sec, cam, ccd, cut, n2 = map(int, m.groups())
        if sectors is not None and sec not in sectors:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d) == 0:
            continue
        r = d.iloc[0].to_dict()
        r.update(sector=sec, cam=cam, ccd=ccd, cut=cut, n=int(round(n2 ** 0.5)))
        rows.append(r)
    if not rows:
        raise FileNotFoundError(f'No psf_calibration_zp.csv under {data_path}.')
    df = pd.DataFrame(rows).sort_values(['sector', 'cam', 'ccd', 'cut'])
    return df.reset_index(drop=True)


def zp_summary(df, value='zp_ab'):
    """Per (sector, cam, ccd) median / robust-scatter / count of `value`."""
    g = df.groupby(['sector', 'cam', 'ccd'])[value]
    out = g.agg(median='median',
                mad=lambda x: 1.4826 * np.median(np.abs(x - np.median(x))),
                n='count').reset_index()
    return out


# Physical TESS focal-plane layout, taken from tessellate's own transient-search
# code (tesstransient.py, "as given by manual" -- the TESS Instrument Handbook).
#
#   ccdArray[camIndex] : 2x2 of CCD numbers, indexed [i=row][j=col] with j=0 left
#     (+x to the right).  The top-row CCDs (i=0) are rotated 180 deg (invert).
#     Camera positions 0,1 share one arrangement; positions 2,3 the (flipped) other.
#   camera order along the strip: [1,2,3,4] (south, dec<0) or [4,3,2,1] (north).
_CCD_ARRAY = [[[4, 3], [1, 2]], [[4, 3], [1, 2]],
              [[2, 1], [3, 4]], [[2, 1], [3, 4]]]
_INVERT_TOP_ROW = True   # i=0 CCDs are 180-deg rotated relative to the mosaic


def _cam_order(north):
    return [4, 3, 2, 1] if north else [1, 2, 3, 4]


def _ccd_subgrid(sub, value, n, per_cut, invert):
    """(block x block) array for one CCD, origin lower; rotated 180 if inverted."""
    block = n if per_cut else 1
    g = np.full((block, block), np.nan)
    if per_cut:
        for _, r in sub.iterrows():
            k = int(r['cut']) - 1            # cut 1..n^2, row-major (row 0 = low y)
            g[k // n, k % n] = r[value]
    elif len(sub):
        g[0, 0] = np.median(sub[value])
    if invert:
        g = np.rot90(g, 2)                   # 180 deg for the flipped CCDs
    return g


def zp_focal_plane(df, sector=None, value='zp_ab', per_cut=False, north=None,
                   savepath=None):
    """
    Focal-plane map of `value` in the physical TESS geometry: the four cameras
    stacked along the 96-deg strip (camera 1 at the bottom), each a 2x2 CCD block,
    with CCDs placed and oriented per the TESS Instrument Handbook layout (the
    top-row CCDs are 180-deg rotated).

    per_cut=False : one median value per CCD.
    per_cut=True  : fill each CCD with its n x n per-cut grid (intra-CCD structure),
                    rotating the inverted CCDs so the whole image is contiguous.
    north : hemisphere/camera order; if None, inferred from the sign of the median
        dec if a 'dec' column exists, else south ([1,2,3,4]).
    """
    d = df if sector is None else df[df['sector'] == sector]
    if north is None:
        north = ('dec' in d and len(d) and np.nanmedian(d['dec']) > 0)
    cam_order = _cam_order(north)

    n = int(d['n'].iloc[0]) if per_cut and len(d) else 1
    block = n if per_cut else 1
    grid = np.full((4 * 2 * block, 2 * block), np.nan)   # 4 cams x 2 CCD-rows, 2 CCD-cols

    for cidx, cam in enumerate(cam_order):
        for i in range(2):                    # in-camera CCD row (0 = top)
            for j in range(2):                # in-camera CCD col (0 = left)
                ccd = _CCD_ARRAY[cidx][i][j]
                sub = d[(d['cam'] == cam) & (d['ccd'] == ccd)]
                invert = _INVERT_TOP_ROW and (i == 0)
                g = _ccd_subgrid(sub, value, n, per_cut, invert)
                # camera cidx occupies vertical band [cidx*2, cidx*2+2) CCD-rows;
                # within it i=0 is the TOP row -> higher y (origin lower).
                r0 = (cidx * 2 + (1 - i)) * block
                c0 = j * block
                grid[r0:r0 + block, c0:c0 + block] = g

    with plt.rc_context(_TEX_RC):
        fig, ax = plt.subplots(figsize=(3.2, 1.4 * len(cam_order)))
        im = ax.imshow(grid, origin='lower', cmap='cividis', aspect='equal')

        for cidx, cam in enumerate(cam_order):
            yb = cidx * 2 * block
            if cidx:
                ax.axhline(yb - 0.5, color='w', lw=2)      # camera boundary
            ax.text(-0.6 * block, yb + block - 0.5, f'Cam {cam}',
                    rotation=90, ha='right', va='center', fontsize=9)
            if not per_cut:
                for i in range(2):
                    for j in range(2):
                        ccd = _CCD_ARRAY[cidx][i][j]
                        r = (cidx * 2 + (1 - i))
                        if np.isfinite(grid[r, j]):
                            label = f'CCD{ccd}\n{grid[r, j]:.3f}'
                            # usetex + path_effects strokes don't compose in
                            # this mpl version, so fake the outline with
                            # offset black copies behind the white text.
                            d = 0.018
                            for dx, dy in [(-d, -d), (-d, d), (d, -d), (d, d),
                                           (-d, 0), (d, 0), (0, -d), (0, d)]:
                                ax.text(j + dx, r + dy, label, ha='center',
                                        va='center', color='k',
                                        fontsize=7, fontweight='bold')
                            ax.text(j, r, label, ha='center', va='center',
                                    color='w', fontsize=7, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        title = (_value_label(value)
                 + (f' -- S{sector}' if sector is not None else '')
                 + (' (per cut)' if per_cut else ''))
        ax.set_title(title, fontsize=9)
        # colorbar the same height as the image axes
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label=_value_label(value))
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, bbox_inches='tight')
            print(f'Saved: {savepath}')
    return fig, ax


def zp_cut_map(df, sector, cam, ccd, value='zp_ab', savepath=None):
    """
    Per-cut ZP map within one CCD: an n x n grid over the cut positions, showing
    spatial ZP variation across a single CCD.
    """
    d = df[(df['sector'] == sector) & (df['cam'] == cam) & (df['ccd'] == ccd)]
    if len(d) == 0:
        raise ValueError('No cuts for that sector/cam/ccd.')
    n = int(d['n'].iloc[0])
    grid = np.full((n, n), np.nan)
    for _, r in d.iterrows():
        k = int(r['cut']) - 1                 # cut index 1..n^2, row-major
        grid[k // n, k % n] = r[value]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(grid, origin='lower', cmap='cividis', aspect='auto')
    ax.set_xlabel('cut column'); ax.set_ylabel('cut row')
    ax.set_title(f'{value} across cuts — S{sector} Cam{cam} Ccd{ccd}')
    plt.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print(f'Saved: {savepath}')
    return fig, ax


def zp_vs_time(df, value='zp_ab', per_ccd=True, savepath=None):
    """
    ZP vs sector (sectors are time-ordered).  With per_ccd, one line per
    (cam, ccd) of the per-sector median; otherwise the overall per-sector median
    with the cut-to-cut scatter as error bars.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if per_ccd:
        for (cam, ccd), sub in df.groupby(['cam', 'ccd']):
            s = sub.groupby('sector')[value].median()
            ax.plot(s.index, s.values, '-o', ms=3, lw=0.8, alpha=0.6,
                    label=f'C{cam}c{ccd}')
        ax.legend(fontsize=6, ncol=4, loc='best')
    else:
        s = df.groupby('sector')[value]
        med = s.median()
        scat = s.agg(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))
        ax.errorbar(med.index, med.values, yerr=scat.values, fmt='-o',
                    capsize=3, color='k')
    ax.set_xlabel('Sector (time-ordered)')
    ax.set_ylabel(value)
    ax.set_title(f'{value} across sectors')
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print(f'Saved: {savepath}')
    return fig, ax
