"""
Evaluate PSF-calibration zeropoints across the TESS focal plane and over time.

The calibration writes one psf_calibration_zp.csv per cut
(<data_path>/Sector{S}/Cam{C}/Ccd{D}/Cut{K}of{n^2}/calibration/), each with
zp_ab, e_zp_ab, zp_scatter, n_stars.  This module collects them and plots:

  collect_zeropoints() : gather all per-cut zeropoints into one table.
  zp_focal_plane()     : ZP laid out in the physical TESS focal-plane geometry
                         (4 cameras stacked, each a 2x2 CCD block, correct
                         orientations) -- combined value per CCD or the full
                         per-cut image; pass sector=None (default) or a list
                         of sectors to stack multiple sectors together.
  zp_cut_map()         : per-cut ZP map within one CCD (spatial across the cut
                         grid); also stackable over a list of sectors.
  zp_vs_time()         : ZP vs sector (sectors are time-ordered) per cam/ccd.

Zeropoints are a *log* (magnitude) quantity -- a flux ratio in disguise -- so
combining several independent estimates (stacking sectors, multiple cuts in
one CCD, etc.) must not median/average the magnitudes directly: that biases
the result relative to the true combine in linear flux space. See
_combine_value / _LOG_VALUE_COLUMNS.
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


# Columns that are magnitudes on a *log* scale (a multiplicative flux ratio in
# disguise).  Combining several independent estimates of one of these -- e.g.
# stacking sectors, or several cuts in one CCD -- must not average/median the
# magnitudes directly: log-space is nonlinear, so a combine there is biased
# relative to the true (linear-flux-space) combine, especially for anything
# weighted.  Convert to a linear flux-scale factor, combine, convert back.
_LOG_VALUE_COLUMNS = {'zp_ab'}


def _zp_to_scale(zp):
    """AB zeropoint (mag) -> linear flux-scale factor (flux = counts * this)."""
    return 10 ** (0.4 * np.asarray(zp, dtype=float))


def _scale_to_zp(f):
    """Linear flux-scale factor -> AB zeropoint (mag)."""
    return 2.5 * np.log10(f)


def _combine_value(vals, value):
    """
    Robust central value of `vals` for stacking (across sectors, cuts, ...).

    For log-scale columns (zp_ab) the median is taken in linear flux space
    and converted back, not on the magnitudes directly -- see
    _LOG_VALUE_COLUMNS.  Everything else (already-linear quantities, or
    magnitude *differences* like e_zp_ab/zp_scatter) is medianed as-is.
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    if value in _LOG_VALUE_COLUMNS:
        return float(_scale_to_zp(np.median(_zp_to_scale(v))))
    return float(np.median(v))

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
    """Per (sector, cam, ccd) robust-combined median / scatter / count of `value`."""
    g = df.groupby(['sector', 'cam', 'ccd'])[value]
    out = g.agg(median=lambda x: _combine_value(x.values, value),
                mad=lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))),
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
    """
    (block x block) array for one CCD, origin lower; rotated 180 if inverted.

    Combines with `_combine_value`, not a raw median: `sub` may hold several
    rows per cell (multiple sectors stacked onto the same cut/CCD), and for a
    log-scale column like zp_ab those must be combined in linear flux space.
    """
    block = n if per_cut else 1
    g = np.full((block, block), np.nan)
    if per_cut:
        buckets = {}
        for _, r in sub.iterrows():
            k = int(r['cut']) - 1            # cut 1..n^2, row-major (row 0 = low y)
            buckets.setdefault(k, []).append(r[value])
        for k, vals in buckets.items():
            g[k // n, k % n] = _combine_value(vals, value)
    elif len(sub):
        g[0, 0] = _combine_value(sub[value].values, value)
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

    per_cut=False : one combined value per CCD.
    per_cut=True  : fill each CCD with its n x n per-cut grid (intra-CCD structure),
                    rotating the inverted CCDs so the whole image is contiguous.
    sector : None stacks every sector present in `df`; an int selects one;
        an iterable of ints stacks that subset.  Stacking combines multiple
        sectors landing on the same CCD/cut with `_combine_value` -- for a
        log-scale column (zp_ab) that means the median in linear flux space,
        not a direct median of the magnitudes.
    north : hemisphere/camera order; if None, inferred from the sign of the median
        dec if a 'dec' column exists, else south ([1,2,3,4]).
    """
    if sector is None:
        d = df
    elif np.isscalar(sector):
        d = df[df['sector'] == sector]
    else:
        d = df[df['sector'].isin(list(sector))]
    sectors_used = sorted(d['sector'].unique().tolist()) if len(d) else []
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
                            off = 0.018
                            for dx, dy in [(-off, -off), (-off, off),
                                           (off, -off), (off, off),
                                           (-off, 0), (off, 0),
                                           (0, -off), (0, off)]:
                                ax.text(j + dx, r + dy, label, ha='center',
                                        va='center', color='k',
                                        fontsize=7, fontweight='bold')
                            ax.text(j, r, label, ha='center', va='center',
                                    color='w', fontsize=7, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if len(sectors_used) == 1:
            sec_label = f' -- S{sectors_used[0]}'
        elif len(sectors_used) > 1:
            sec_label = f' -- stack of {len(sectors_used)} sectors ' \
                        f'(S{sectors_used[0]}--S{sectors_used[-1]})'
        else:
            sec_label = ''
        title = _value_label(value) + sec_label + (' (per cut)' if per_cut else '')
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

    sector : an int selects one sector; an iterable of ints (or None, for
        every sector present) stacks that subset -- multiple sectors landing
        on the same cut are combined with `_combine_value` (linear flux
        space for a log-scale column such as zp_ab, not a raw median of the
        magnitudes).
    """
    if sector is None:
        d = df[(df['cam'] == cam) & (df['ccd'] == ccd)]
    elif np.isscalar(sector):
        d = df[(df['sector'] == sector) & (df['cam'] == cam) & (df['ccd'] == ccd)]
    else:
        d = df[df['sector'].isin(list(sector)) & (df['cam'] == cam) & (df['ccd'] == ccd)]
    if len(d) == 0:
        raise ValueError('No cuts for that sector/cam/ccd.')
    sectors_used = sorted(d['sector'].unique().tolist())
    n = int(d['n'].iloc[0])
    grid = np.full((n, n), np.nan)
    buckets = {}
    for _, r in d.iterrows():
        k = int(r['cut']) - 1                 # cut index 1..n^2, row-major
        buckets.setdefault(k, []).append(r[value])
    for k, vals in buckets.items():
        grid[k // n, k % n] = _combine_value(vals, value)

    with plt.rc_context(_TEX_RC):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(grid, origin='lower', cmap='cividis', aspect='auto')
        ax.set_xlabel('cut column'); ax.set_ylabel('cut row')
        if len(sectors_used) == 1:
            sec_label = f'S{sectors_used[0]}'
        else:
            sec_label = f'stack of {len(sectors_used)} sectors ' \
                        f'(S{sectors_used[0]}--S{sectors_used[-1]})'
        ax.set_title(f'{_value_label(value)} across cuts -- {sec_label} '
                     f'Cam{cam} Ccd{ccd}')
        fig.colorbar(im, ax=ax, label=_value_label(value))
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
    with plt.rc_context(_TEX_RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        if per_ccd:
            for (cam, ccd), sub in df.groupby(['cam', 'ccd']):
                s = sub.groupby('sector')[value].agg(
                    lambda x: _combine_value(x.values, value))
                ax.plot(s.index, s.values, '-o', ms=3, lw=0.8, alpha=0.6,
                        label=f'C{cam}c{ccd}')
            ax.legend(fontsize=6, ncol=4, loc='best')
        else:
            s = df.groupby('sector')[value]
            med = s.agg(lambda x: _combine_value(x.values, value))
            scat = s.agg(lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))
            ax.errorbar(med.index, med.values, yerr=scat.values, fmt='-o',
                        capsize=3, color='k')
        ax.set_xlabel('Sector (time-ordered)')
        ax.set_ylabel(_value_label(value))
        ax.set_title(f'{_value_label(value)} across sectors')
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, bbox_inches='tight')
            print(f'Saved: {savepath}')
    return fig, ax
