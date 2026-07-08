"""
Evaluate PSF-calibration zeropoints across the TESS focal plane and over time.

The calibration writes one psf_calibration_zp.csv per cut
(<data_path>/Sector{S}/Cam{C}/Ccd{D}/Cut{K}of{n^2}/calibration/), each with
zp_ab, e_zp_ab, zp_scatter, n_stars.  This module collects them and plots:

  collect_zeropoints() : gather all per-cut zeropoints into one table.
  zp_ccd_map()         : median ZP per (cam, ccd) -- focal-plane map for a sector.
  zp_cut_map()         : per-cut ZP map within one CCD (spatial across the cut grid).
  zp_vs_time()         : ZP vs sector (sectors are time-ordered) per cam/ccd.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def zp_ccd_map(df, sector=None, value='zp_ab', savepath=None):
    """
    Focal-plane map: median `value` per (cam, ccd) for a sector (4 cams x 4 ccds).
    """
    d = df if sector is None else df[df['sector'] == sector]
    grid = np.full((4, 4), np.nan)
    for cam in range(1, 5):
        for ccd in range(1, 5):
            v = d[(d['cam'] == cam) & (d['ccd'] == ccd)][value]
            if len(v):
                grid[cam - 1, ccd - 1] = np.median(v)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(grid, origin='upper', cmap='viridis', aspect='auto')
    for i in range(4):
        for j in range(4):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center',
                        color='w', fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'CCD{c}' for c in range(1, 5)])
    ax.set_yticks(range(4)); ax.set_yticklabels([f'Cam{c}' for c in range(1, 5)])
    ax.set_title(f'Median {value}'
                 + (f' — Sector {sector}' if sector is not None else ''))
    plt.colorbar(im, ax=ax, label=value)
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
    im = ax.imshow(grid, origin='lower', cmap='viridis', aspect='auto')
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
