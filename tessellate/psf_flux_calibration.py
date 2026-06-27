"""
PSF flux calibration for tessellate reference images.

Fits PSF photometry on bright, unsaturated, isolated Gaia DR3 sources in the
reference image and derives a TESS-to-Gaia-Rp AB magnitude zeropoint.

Each star is fit simultaneously for flux, sub-pixel position (dx, dy), and a
flat background offset.  Stars are processed in parallel via joblib.

Usage
-----
from psf_flux_calibration import run_calibration
from astropy.io import fits
from astropy.wcs import WCS

with fits.open('Cam1/Ccd1/wcs/ref/corrected.fits') as f:
    image = f[1].data
    wcs = WCS(f[1].header)

cut_corners, _, _, _ = processor.find_cuts(cam=1, ccd=1, n=8, plot=False)

zp_ab, zp_err, df = run_calibration(
    image, wcs,
    sector=73, cam=1, ccd=1,
    cut_corner=cut_corners[cut - 1],
    plot=True, savepath='.'
)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import fits
from astropy.wcs import WCS
import warnings
warnings.filterwarnings('ignore')

# Gaia Rp Vega-to-AB offset (Casagrande & VandenBerg 2018, MNRAS 479, L102)
GAIA_RP_AB_OFFSET = 0.152

TESS_PIX_SCALE = 21.0  # arcsec/pixel
GAIA_PATH_DEFAULT = '/fred/oz335/GAIAdata/full_gaia_cat.csv'
PRF_PATH_DEFAULT = '/fred/oz335/_local_TESS_PRFs'


# ---------------------------------------------------------------------------
# Gaia query
# ---------------------------------------------------------------------------

def _query_gaia(ra_centre, dec_centre, radius_arcsec, gaia_path, gmag_limit):
    import duckdb
    return duckdb.sql(f"""
        SELECT * FROM (
            SELECT *,
                2 * degrees(asin(sqrt(
                    pow(sin(radians(("dec" - {dec_centre}) / 2)), 2) +
                    cos(radians("dec")) * cos(radians({dec_centre})) *
                    pow(sin(radians((ra - {ra_centre}) / 2)), 2)
                ))) * 3600 AS dist_arcsec
            FROM read_csv('{gaia_path}', ignore_errors=true)
        )
        WHERE dist_arcsec < {radius_arcsec}
          AND Gmag < {gmag_limit}
    """).df()


# ---------------------------------------------------------------------------
# Star selection
# ---------------------------------------------------------------------------

def _select_isolated(gaia_all, gaia_cal, wcs, cut_corner, image_shape,
                     iso_radius_pix, edge_margin, delta_mag=2.0):
    """
    Return calibration stars that:
      - fall within the cut image (with edge_margin buffer)
      - have no Gaia neighbour within iso_radius_pix TESS pixels that is
        brighter than star_rp + delta_mag

    The WCS is the full CCD WCS; cut_corner=(x0,y0) gives the CCD pixel
    coordinate of the bottom-left corner of the cut image.
    """
    ny, nx = image_shape
    x0, y0 = cut_corner
    iso_arcsec = iso_radius_pix * TESS_PIX_SCALE

    ccd_x, ccd_y = wcs.all_world2pix(gaia_cal.ra.values, gaia_cal.dec.values, 0)
    loc_x = ccd_x - x0
    loc_y = ccd_y - y0

    ra_all = gaia_all.ra.values
    dec_all = gaia_all.dec.values
    rp_all = gaia_all.RPmag.values

    keep = []
    for i in range(len(gaia_cal)):
        lx, ly = loc_x[i], loc_y[i]
        if not (edge_margin < lx < nx - edge_margin and
                edge_margin < ly < ny - edge_margin):
            continue
        cos_dec = np.cos(np.radians(gaia_cal.dec.values[i]))
        dra = (ra_all - gaia_cal.ra.values[i]) * cos_dec
        ddec = dec_all - gaia_cal.dec.values[i]
        sep = np.sqrt(dra**2 + ddec**2) * 3600.0
        mag_limit = gaia_cal.RPmag.values[i] + delta_mag
        close = (sep > 0.5) & (sep < iso_arcsec) & (rp_all < mag_limit)
        if np.sum(close) == 0:
            keep.append(i)

    keep = np.array(keep, dtype=int)
    if len(keep) == 0:
        return gaia_cal.iloc[:0].reset_index(drop=True), np.array([]), np.array([]), np.array([]), np.array([])

    return (gaia_cal.iloc[keep].reset_index(drop=True),
            ccd_x[keep], ccd_y[keep],
            loc_x[keep], loc_y[keep])


# ---------------------------------------------------------------------------
# Per-star PSF fit worker  (top-level so joblib can pickle it)
# ---------------------------------------------------------------------------

def _fit_star_worker(stamp, ccd_x, ccd_y, cam, ccd, sector, prf_dir,
                     stamp_size, ra, dec, loc_x, loc_y, rp_vega, bp_rp=np.nan):
    """
    Fit one calibration star.  Constructs TESS_PRF internally so the object
    does not need to cross process boundaries.

    Returns a dict of results, or None on failure.
    """
    from PRF import TESS_PRF

    cent = stamp_size // 2

    # Per-pixel noise from stamp corners
    c = 2
    corners = np.concatenate([stamp[:c, :c].ravel(), stamp[:c, -c:].ravel(),
                               stamp[-c:, :c].ravel(), stamp[-c:, -c:].ravel()])
    bg0 = np.nanmedian(corners)
    sigma_bg = np.nanstd(corners)
    if not np.isfinite(sigma_bg) or sigma_bg <= 0:
        sigma_bg = max(np.sqrt(abs(bg0)), 1.0)
    var_pix = sigma_bg**2

    flux0 = max(float(np.nansum(stamp - bg0)), 1.0)

    try:
        prf = TESS_PRF(cam=cam, ccd=ccd, sector=sector,
                       colnum=int(np.clip(ccd_x, 44, 2090)),
                       rownum=int(np.clip(ccd_y,  1, 2040)),
                       localdatadir=prf_dir)
    except Exception:
        return None

    def _model(params):
        flux, dx, dy, bg = params
        p = prf.locate(cent + dx, cent + dy, (stamp_size, stamp_size))
        p = p / np.nansum(p)
        return flux * p + bg

    def _chi2(params):
        return float(np.nansum((stamp - _model(params))**2) / var_pix)

    try:
        res = minimize(_chi2, [flux0, 0.0, 0.0, bg0], method='BFGS',
                       options={'maxiter': 10000, 'gtol': 1e-6})
        flux, dx, dy, bg = res.x
        if not (flux > 0 and np.isfinite(flux)):
            return None
    except Exception:
        return None

    # Model and residual at the best fit
    p_fit = prf.locate(cent + float(dx), cent + float(dy), (stamp_size, stamp_size))
    p_fit = p_fit / np.nansum(p_fit)
    model_stamp = float(flux) * p_fit + float(bg)
    residual_stamp = stamp - model_stamp

    # Flux uncertainty from the analytic linear least-squares solution.
    # The model is linear in (flux, bg): data = flux * P + bg, with P the
    # normalised PSF.  The parameter covariance is s2 * inv(A^T A) for design
    # columns A = [P, 1], giving flux variance s2 * S11 / (Spp*S11 - Sp1^2).
    #
    # The per-pixel variance s2 is estimated from the fit residuals (not the
    # corners), so that an imperfect PSF model -- which biases the flux --
    # inflates the reported error.  This is the standard residual-variance
    # estimator (equivalent to scaling the covariance by reduced chi^2).
    #
    # Restrict to the inner 5x5 region around the centre, where essentially all
    # of the source flux and PSF-model power lies.
    inner = 2  # half-width of the 5x5 region
    inner_mask = np.zeros_like(stamp, dtype=bool)
    inner_mask[cent - inner:cent + inner + 1, cent - inner:cent + inner + 1] = True
    finite_pix = (inner_mask & np.isfinite(stamp) &
                  np.isfinite(p_fit) & np.isfinite(residual_stamp))
    P = p_fit[finite_pix]
    Spp = float(np.sum(P * P))
    Sp1 = float(np.sum(P))
    S11 = float(P.size)
    det = Spp * S11 - Sp1**2

    n_params = 4  # flux, dx, dy, bg
    dof = P.size - n_params
    if dof > 0:
        s2 = float(np.sum(residual_stamp[finite_pix]**2) / dof)
    else:
        s2 = var_pix
    if det > 0 and s2 > 0:
        e_flux = float(np.sqrt(s2 * S11 / det))
    else:
        e_flux = np.nan

    rp_ab = rp_vega + GAIA_RP_AB_OFFSET
    zp = rp_ab + 2.5 * np.log10(flux)
    e_zp = (2.5 / np.log(10)) * (e_flux / flux) if (np.isfinite(e_flux) and flux > 0) else np.nan

    return {
        'ra': ra, 'dec': dec,
        'x_pix': loc_x, 'y_pix': loc_y,
        'dx_fit': float(dx), 'dy_fit': float(dy),
        'flux_psf': float(flux), 'e_flux_psf': e_flux,
        'background': float(bg),
        'rp_vega': rp_vega, 'rp_ab': rp_ab,
        'bp_rp': bp_rp,
        'zp_ab': zp, 'e_zp_ab': e_zp,
        'stamp_data': stamp,
        'model_data': model_stamp,
        'residual_data': residual_stamp,
    }


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def run_calibration(image, wcs, sector, cam, ccd,
                    cut_corner=(0, 0),
                    gaia_path=GAIA_PATH_DEFAULT,
                    prf_path=PRF_PATH_DEFAULT,
                    mag_lo=11.0, mag_hi=15.5,
                    iso_radius_pix=4.0, stamp_size=9,
                    delta_mag=2.0, edge_margin=5,
                    n_jobs=-1,
                    plot=True, savepath=None):
    """
    Derive a TESS-to-Gaia-Rp AB zeropoint from PSF photometry on a reference image.

    Parameters
    ----------
    image : 2D ndarray
        Reference image for the cut (background fitted per-star).
    wcs : astropy.wcs.WCS
        Full CCD WCS from wcs/ref/corrected.fits.  Calling
        wcs.all_world2pix(ra, dec, 0) returns CCD-level pixel coordinates.
    sector, cam, ccd : int
        TESS identifiers.
    cut_corner : (x0, y0)
        CCD pixel coordinate of the bottom-left corner of this cut, taken
        directly from cut_corners[cut-1] returned by DataProcessor.find_cuts.
    gaia_path : str
        Path to the local Gaia DR3 CSV file.
    prf_path : str
        Root directory of local TESS PRF files.
    mag_lo, mag_hi : float
        Gaia Rp (Vega) magnitude range for calibration stars.
    iso_radius_pix : float
        Isolation radius in TESS pixels.
    stamp_size : int (odd)
        Pixel width of the stamp cutout for each PSF fit.
    delta_mag : float
        A calibration star is rejected if any Gaia neighbour within
        iso_radius_pix pixels is brighter than star_rp + delta_mag.
    edge_margin : int
        Minimum pixel distance from cut-image edge for a usable star.
    n_jobs : int
        Number of parallel workers passed to joblib.Parallel.
        -1 uses all available cores.
    plot : bool
        If True, write diagnostic figures to savepath.
    savepath : str or None
        Directory for output files.

    Returns
    -------
    zp_ab : float
        Error-weighted mean AB zeropoint: Rp_AB = -2.5*log10(flux) + zp_ab.
    zp_err : float
        Formal uncertainty: 1 / sqrt(sum(1/e_zp_i^2)).
    results : pd.DataFrame
        Full per-star table.
    Writes (if savepath is set):
        psf_calibration_zp.csv    — zp_ab, e_zp_ab, n_stars
        psf_calibration_stars.csv — x, y, ra, dec, flux, e_flux, background, Rp_ab
    """
    from joblib import Parallel, delayed

    ny, nx = image.shape
    x0, y0 = cut_corner

    # Field centre from the cut image centre in CCD coordinates
    ra_cen, dec_cen = wcs.all_pix2world(x0 + nx / 2.0, y0 + ny / 2.0, 0)
    radius_arcsec = np.hypot(nx / 2.0, ny / 2.0) * TESS_PIX_SCALE * 1.1

    print(f'Field centre: RA={ra_cen:.4f} Dec={dec_cen:.4f}')
    print(f'Querying Gaia DR3 within {radius_arcsec/60:.1f} arcmin ...')

    # Query deep enough to catch all relevant neighbours (faintest cal star + delta_mag)
    query_maglim = mag_hi + delta_mag
    gaia_all = _query_gaia(ra_cen, dec_cen, radius_arcsec, gaia_path, query_maglim)
    print(f'  {len(gaia_all)} sources (Gmag < {query_maglim})')

    if 'RPmag' not in gaia_all.columns:
        raise RuntimeError("'RPmag' column not found in Gaia catalog.")

    gaia_cal = gaia_all.dropna(subset=['RPmag'])
    gaia_cal = gaia_cal[(gaia_cal.RPmag >= mag_lo) & (gaia_cal.RPmag <= mag_hi)].copy()
    print(f'  {len(gaia_cal)} candidates with {mag_lo} <= Rp <= {mag_hi}')

    if len(gaia_cal) == 0:
        raise RuntimeError(f'No Gaia stars with {mag_lo} <= Rp <= {mag_hi} in field.')

    gaia_iso, ccd_xs, ccd_ys, loc_xs, loc_ys = _select_isolated(
        gaia_all, gaia_cal, wcs, cut_corner, image.shape, iso_radius_pix, edge_margin, delta_mag)
    print(f'  {len(gaia_iso)} isolated, on-image calibration stars')

    if len(gaia_iso) == 0:
        raise RuntimeError('No isolated calibration stars remain after cuts.')

    prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
    half = stamp_size // 2

    has_bp = 'BPmag' in gaia_iso.columns

    # Build per-star argument list; extract stamps here (serial, cheap)
    tasks = []
    for i in range(len(gaia_iso)):
        lx_int = int(np.round(loc_xs[i]))
        ly_int = int(np.round(loc_ys[i]))
        sy0, sy1 = ly_int - half, ly_int + half + 1
        sx0, sx1 = lx_int - half, lx_int + half + 1
        if sy0 < 0 or sx0 < 0 or sy1 > ny or sx1 > nx:
            continue
        stamp = image[sy0:sy1, sx0:sx1].copy().astype(float)
        if not np.isfinite(stamp).all():
            continue
        bp_rp = (float(gaia_iso.BPmag.values[i]) - float(gaia_iso.RPmag.values[i])
                 if has_bp else np.nan)
        tasks.append((
            stamp,
            ccd_xs[i], ccd_ys[i],
            cam, ccd, sector, prf_dir,
            stamp_size,
            float(gaia_iso.ra.values[i]),
            float(gaia_iso.dec.values[i]),
            float(loc_xs[i]), float(loc_ys[i]),
            float(gaia_iso.RPmag.values[i]),
            bp_rp,
        ))

    print(f'  Fitting {len(tasks)} stars in parallel (n_jobs={n_jobs}) ...')

    results_raw = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_fit_star_worker)(*args) for args in tasks
    )

    rows = [r for r in results_raw if r is not None]

    if len(rows) == 0:
        raise RuntimeError('PSF fitting failed for all calibration stars.')

    df = pd.DataFrame(rows)
    print(f'  {len(df)} successful PSF fits')

    # Sigma-clipped zeropoint (sigma=3)
    finite = np.isfinite(df.zp_ab.values)
    if finite.sum() < 2:
        raise RuntimeError('Fewer than 2 stars with finite zeropoints.')

    clip_mask = _sigma_clip_mask(df.zp_ab.values[finite], nsigma=3)
    zp_vals_clipped = df.zp_ab.values[finite][clip_mask]
    zp_ab = float(np.mean(zp_vals_clipped))
    zp_err = float(np.std(zp_vals_clipped))

    print(f'\nZeropoint (AB): {zp_ab:.4f} +/- {zp_err:.4f} mag  '
          f'(N={len(zp_vals_clipped)} stars, 3-sigma clipped mean)')
    print(f'  formula: Rp_AB = -2.5 * log10(flux) + {zp_ab:.4f}')
    print(f'  (Gaia Rp Vega-to-AB offset: +{GAIA_RP_AB_OFFSET:.3f} mag, '
          'Casagrande & VandenBerg 2018)')

    if savepath is not None:
        summary = pd.DataFrame({'zp_ab': [zp_ab], 'e_zp_ab': [zp_err],
                                'n_stars': [int(len(zp_vals_clipped))]})
        summary_path = f'{savepath}/psf_calibration_zp.csv'
        summary.to_csv(summary_path, index=False)
        print(f'  Zeropoint saved:  {summary_path}')

        stars = df[['x_pix', 'y_pix', 'ra', 'dec',
                    'flux_psf', 'e_flux_psf', 'background', 'rp_ab']].copy()
        stars.columns = ['x', 'y', 'ra', 'dec', 'flux', 'e_flux', 'background', 'Rp_ab']
        stars_path = f'{savepath}/psf_calibration_stars.csv'
        stars.to_csv(stars_path, index=False)
        print(f'  Star catalog saved: {stars_path}')

    if plot:
        for fn, args in [
            (_summary_figure,         (image, df, zp_ab, zp_err, savepath)),
            (_colour_magnitude_figure, (df, zp_ab, zp_err, savepath)),
            (_star_fits_pdf,           (df, savepath)),
        ]:
            try:
                fn(*args)
            except Exception:
                import traceback
                print(f'  WARNING: {fn.__name__} failed:')
                traceback.print_exc()

    return zp_ab, zp_err, df


# ---------------------------------------------------------------------------
# Per-frame detection limits
# ---------------------------------------------------------------------------

def _default_time_bins(sector):
    """Return default time bin strings for a given sector cadence."""
    if sector < 27:
        return ['30min', '1hr', '2hr', '6hr', '24hr']
    elif sector < 56:
        return ['10min', '30min', '1hr', '3hr', '12hr', '24hr']
    else:
        return ['200sec', '30min', '1hr', '6hr', '24hr']


def _parse_time_bins(time_bins, time_array):
    """Convert time bin strings to frame counts using the actual time array cadence."""
    import re
    conversions = {'sec': 1/86400, 'min': 1/1440, 'hr': 1/24, 'day': 1}
    cadence_days = float(np.nanmedian(np.diff(time_array)))
    frame_bins = []
    labels = []
    for b in time_bins:
        v, u = re.match(r'([\d.]+)(\w+)', b.strip()).groups()
        resolution_days = float(v) * conversions[u]
        n = max(1, int(round(resolution_days / cadence_days)))
        frame_bins.append(n)
        labels.append(b.strip())
    return frame_bins, labels


def _bin_flux(flux, n):
    """Average flux in blocks of n frames. Returns shape (n_bins, ny, nx)."""
    n_bins = flux.shape[0] // n
    return flux[:n_bins * n].reshape(n_bins, n, flux.shape[1], flux.shape[2]).mean(axis=1)


def _frame_noise(flux_3d):
    """Spatial MAD per frame → per-pixel sigma. Returns shape (n_frames,)."""
    sigma = np.zeros(flux_3d.shape[0])
    for i in range(flux_3d.shape[0]):
        frame = flux_3d[i]
        finite = frame[np.isfinite(frame)]
        if finite.size == 0:
            sigma[i] = np.nan
        else:
            med = np.median(finite)
            sigma[i] = np.median(np.abs(finite - med)) * 1.4826
    return sigma


def compute_detection_limits(reduced_flux, time_array, zp_ab,
                              sector, cam, ccd,
                              cut_corner=(0, 0),
                              prf_path=PRF_PATH_DEFAULT,
                              stamp_size=9,
                              time_bins=None,
                              savepath=None):
    """
    Compute PSF detection limits for one or more time bin sizes.

    For each bin size N, N consecutive frames are averaged and the background
    noise is measured from the actual binned data via spatial MAD.  The PSF
    matched-filter noise factor converts per-pixel noise to point-source flux
    uncertainty.

    Parameters
    ----------
    reduced_flux : ndarray, shape (n_frames, ny, nx)
        Background-subtracted flux cube (e.g. ReducedFlux.npy).
    time_array : ndarray, shape (n_frames,)
        MJD timestamps for each frame (e.g. Times.npy).
    zp_ab : float
        AB zeropoint from run_calibration.
    sector, cam, ccd : int
    cut_corner : (x0, y0)
        CCD pixel of the cut's bottom-left corner (from find_cuts).
    prf_path : str
    stamp_size : int
    time_bins : list of str or None
        Time bin sizes as strings, e.g. ['10min', '30min', '1hr', '12hr'].
        Uses the same format as the tessellate transient search.
        None uses cadence-appropriate defaults.
    savepath : str or None
        If given, saves ``detection_limits.csv`` here.

    Returns
    -------
    results : dict  keyed by n_bin (int)
        Each value is a dict with keys sigma_bg, mag_lim_3sigma,
        mag_lim_5sigma, mag_lim_10sigma — all 1D arrays over binned frames.
    """
    from PRF import TESS_PRF

    reduced_flux = np.asarray(reduced_flux, dtype=float)
    time_array = np.asarray(time_array, dtype=float)
    n_frames, ny, nx = reduced_flux.shape

    if time_bins is None:
        time_bins = _default_time_bins(sector)

    frame_bins, labels = _parse_time_bins(time_bins, time_array)

    # PRF at cut centre — matched-filter noise factor
    x0, y0 = cut_corner
    ccd_x = x0 + nx // 2
    ccd_y = y0 + ny // 2

    prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
    prf = TESS_PRF(cam=cam, ccd=ccd, sector=sector,
                   colnum=int(np.clip(ccd_x, 44, 2090)),
                   rownum=int(np.clip(ccd_y,  1, 2040)),
                   localdatadir=prf_dir)

    cent = stamp_size // 2
    p = prf.locate(float(cent), float(cent), (stamp_size, stamp_size))
    p = p / np.nansum(p)
    prf_noise_factor = 1.0 / np.sqrt(np.nansum(p**2))

    def _limits(sigma_bg, n_bins):
        out = {}
        for nsig, col in [(3, 'mag_lim_3sigma'), (5, 'mag_lim_5sigma'), (10, 'mag_lim_10sigma')]:
            flux = nsig * sigma_bg * prf_noise_factor
            mag = np.full(n_bins, np.nan)
            ok = flux > 0
            mag[ok] = zp_ab - 2.5 * np.log10(flux[ok])
            out[col] = mag
        return out

    results = {}

    print(f'\nDetection limits ({n_frames} frames):')

    for label, n_bin in zip(labels, frame_bins):
        binned = _bin_flux(reduced_flux, n_bin)
        n_bins = binned.shape[0]
        sigma_bg = _frame_noise(binned)
        lims = _limits(sigma_bg, n_bins)

        results[label] = {'n_frames_binned': n_bin, 'sigma_bg': sigma_bg, **lims}

        med5 = np.nanmedian(lims['mag_lim_5sigma'])
        print(f'  {label:>8s}  ({n_bin} frames)  →  5-sigma median: {med5:.3f} AB mag')

        if savepath is not None:
            df_lim = pd.DataFrame({
                'bin_index': np.arange(n_bins),
                'sigma_bg': sigma_bg,
                'mag_lim_3sigma': lims['mag_lim_3sigma'],
                'mag_lim_5sigma': lims['mag_lim_5sigma'],
                'mag_lim_10sigma': lims['mag_lim_10sigma'],
            })
            fout = f'{savepath}/detection_limits_{label}.csv'
            df_lim.to_csv(fout, index=False)
            print(f'  Saved: {fout}')

    return results


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _sigma_clip_mask(values, nsigma=3, maxiter=5):
    """Return boolean mask of inliers after iterative sigma clipping."""
    mask = np.isfinite(values)
    for _ in range(maxiter):
        med = np.median(values[mask])
        std = np.std(values[mask])
        new_mask = np.isfinite(values) & (np.abs(values - med) < nsigma * std)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return mask


def _summary_figure(image, df, zp_ab, zp_err, savepath):
    import matplotlib.gridspec as gridspec

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 9,
    })

    zp_vals = df.zp_ab.values
    e_zp = df.e_zp_ab.values
    x_pix = df.x_pix.values
    y_pix = df.y_pix.values

    inliers = _sigma_clip_mask(zp_vals)
    zp_in = zp_vals[inliers]
    med_zp = float(np.median(zp_in))
    std_zp = float(np.std(zp_in))
    n_in = int(inliers.sum())

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.32)

    # ---- Panel 1: ZP histogram ------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    bins = max(6, n_in // 3)
    ax1.hist(zp_in, bins=bins, color='C0', alpha=0.75, edgecolor='white', linewidth=0.5)
    ax1.axvline(zp_ab,  color='k',   ls='--', lw=1.5, label=f'Weighted mean = {zp_ab:.3f}')
    ax1.axvline(med_zp, color='C3',  ls=':',  lw=1.5, label=f'Median = {med_zp:.3f}')
    ax1.axvspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.15, color='k',
                label=f'$\\pm$1$\\sigma$ = {zp_err:.4f}')
    stats_txt = (f'N = {n_in}  (of {len(df)} fitted)\n'
                 f'std = {std_zp:.4f} mag')
    ax1.text(0.97, 0.96, stats_txt, transform=ax1.transAxes,
             ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax1.set_xlabel('Per-star ZP (AB mag)')
    ax1.set_ylabel('N stars')
    ax1.set_title('Zeropoint distribution (3$\\sigma$ clipped)')
    ax1.legend(fontsize=7, loc='upper left')

    # ---- Panel 2: Star locations on reference image --------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    nonan = np.isfinite(image)
    vlo = float(np.percentile(image[nonan], 2))
    vhi = float(np.percentile(image[nonan], 98))
    im2 = ax2.imshow(image, origin='lower', cmap='gray', vmin=vlo, vmax=vhi, aspect='auto')
    sc2 = ax2.scatter(x_pix[inliers],  y_pix[inliers],
                      c=zp_vals[inliers], cmap='RdYlGn_r',
                      vmin=med_zp - 3*std_zp, vmax=med_zp + 3*std_zp,
                      s=60, edgecolors='white', linewidths=0.5, zorder=3,
                      label='Inliers')
    ax2.scatter(x_pix[~inliers], y_pix[~inliers],
                marker='x', s=40, color='C3', linewidths=1.0, zorder=4,
                label='Outliers')
    plt.colorbar(sc2, ax=ax2, label='ZP (AB mag)')
    ax2.set_xlabel('Column (px)')
    ax2.set_ylabel('Row (px)')
    ax2.set_title('Calibration star locations')
    ax2.legend(fontsize=7, loc='upper right')

    # ---- Panel 3: ZP vs x_pix ------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.errorbar(x_pix[inliers], zp_vals[inliers], yerr=e_zp[inliers],
                 fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7, elinewidth=0.8,
                 capsize=2)
    if n_in >= 2:
        cx = np.polyfit(x_pix[inliers], zp_vals[inliers], 1)
        xg = np.linspace(x_pix[inliers].min(), x_pix[inliers].max(), 100)
        ax3.plot(xg, np.polyval(cx, xg), color='C3', lw=1.2,
                 label=f'slope={cx[0]*1e3:.2f}e-3 mag/px')
        ax3.legend(fontsize=7)
    ax3.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.6)
    ax3.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k')
    ax3.set_xlabel('x pixel (cut frame)')
    ax3.set_ylabel('ZP (AB mag)')
    ax3.set_title('Zeropoint vs detector column')

    # ---- Panel 4: ZP vs y_pix ------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.errorbar(y_pix[inliers], zp_vals[inliers], yerr=e_zp[inliers],
                 fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7, elinewidth=0.8,
                 capsize=2)
    if n_in >= 2:
        cy = np.polyfit(y_pix[inliers], zp_vals[inliers], 1)
        yg = np.linspace(y_pix[inliers].min(), y_pix[inliers].max(), 100)
        ax4.plot(yg, np.polyval(cy, yg), color='C3', lw=1.2,
                 label=f'slope={cy[0]*1e3:.2f}e-3 mag/px')
        ax4.legend(fontsize=7)
    ax4.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.6)
    ax4.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k')
    ax4.set_xlabel('y pixel (cut frame)')
    ax4.set_ylabel('ZP (AB mag)')
    ax4.set_title('Zeropoint vs detector row')

    fig.suptitle(f'PSF Flux Calibration  —  ZP = {zp_ab:.4f} ± {zp_err:.4f} AB mag  '
                 f'(N={n_in})', fontsize=11, y=1.01)

    if savepath is not None:
        fname = f'{savepath}/psf_calibration_diagnostic.pdf'
        fig.savefig(fname, bbox_inches='tight')
        print(f'  Summary diagnostic: {fname}')
    else:
        plt.show()
    plt.close(fig)


def _colour_magnitude_figure(df, zp_ab, zp_err, savepath):

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 9,
    })

    inliers = _sigma_clip_mask(df.zp_ab.values)
    rp_ab = df.rp_ab.values
    zp_vals = df.zp_ab.values
    e_zp = df.e_zp_ab.values
    bp_rp = df.bp_rp.values if 'bp_rp' in df.columns else np.full(len(df), np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), tight_layout=True)

    # ---- Top: source magnitude vs ZP -----------------------------------------
    ax1.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.7, zorder=1, label=f'ZP={zp_ab:.4f}')
    ax1.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k', zorder=1)
    ax1.errorbar(rp_ab[inliers], zp_vals[inliers], yerr=e_zp[inliers],
                 fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7,
                 elinewidth=0.8, capsize=2, zorder=3, label='Inliers')
    if inliers.sum() >= 2:
        cm = np.polyfit(rp_ab[inliers], zp_vals[inliers], 1)
        xg = np.linspace(rp_ab[inliers].min(), rp_ab[inliers].max(), 100)
        ax1.plot(xg, np.polyval(cm, xg), color='C3', lw=1.2, zorder=2,
                 label=f'slope={cm[0]:.4f} mag/mag')
    ax1.set_xlabel('Gaia Rp (AB mag)')
    ax1.set_ylabel('Per-star ZP (AB mag)')
    ax1.set_title('Zeropoint vs source magnitude')
    ax1.legend(fontsize=7)

    # ---- Bottom: ZP vs BP-RP colour ------------------------------------------
    has_colour = np.isfinite(bp_rp).any()
    if has_colour:
        ok = inliers & np.isfinite(bp_rp)
        ax2.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.7, zorder=1)
        ax2.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k', zorder=1)
        ax2.errorbar(bp_rp[ok], zp_vals[ok], yerr=e_zp[ok],
                     fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7,
                     elinewidth=0.8, capsize=2, zorder=3, label='Inliers')
        if ok.sum() >= 2:
            cc = np.polyfit(bp_rp[ok], zp_vals[ok], 1)
            cg = np.linspace(bp_rp[ok].min(), bp_rp[ok].max(), 100)
            ax2.plot(cg, np.polyval(cc, cg), color='C3', lw=1.2, zorder=2,
                     label=f'slope={cc[0]:.4f} mag/mag')
        ax2.set_xlabel('Gaia BP - RP (mag)')
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, 'BP-RP colour not available in catalog',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=9)
    ax2.set_ylabel('Per-star ZP (AB mag)')
    ax2.set_title('Zeropoint vs Gaia colour')

    fig.suptitle(f'PSF Calibration Colour Diagnostics  —  ZP = {zp_ab:.4f} ± {zp_err:.4f}',
                 fontsize=10)

    if savepath is not None:
        fname = f'{savepath}/psf_calibration_colour.pdf'
        fig.savefig(fname, bbox_inches='tight')
        print(f'  Colour diagnostic:  {fname}')
    else:
        plt.show()
    plt.close(fig)


def _star_fits_pdf(df, savepath):
    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 8,
    })

    stars_per_page = 3
    rows = list(df.itertuples())
    n_pages = int(np.ceil(len(rows) / stars_per_page))

    fname = (f'{savepath}/psf_calibration_star_fits.pdf'
             if savepath is not None else '/tmp/psf_calibration_star_fits.pdf')

    with PdfPages(fname) as pdf:
        for page in range(n_pages):
            chunk = rows[page * stars_per_page:(page + 1) * stars_per_page]
            n_rows = len(chunk)
            fig, axes = plt.subplots(n_rows, 3,
                                     figsize=(8.5, 3.0 * n_rows),
                                     squeeze=False)

            for row_i, r in enumerate(chunk):
                stamp = getattr(r, 'stamp_data', None)
                model = getattr(r, 'model_data', None)
                residual = getattr(r, 'residual_data', None)

                if stamp is None:
                    for col_i in range(3):
                        axes[row_i, col_i].axis('off')
                    continue

                # Scale the colour range to the inner 5x5 region used for the fit
                cent = stamp.shape[0] // 2
                inner = 2
                sl = np.s_[cent - inner:cent + inner + 1, cent - inner:cent + inner + 1]
                stamp_inner = stamp[sl]
                res_inner = residual[sl]
                vlo = float(np.nanmin(stamp_inner))
                vhi = float(np.nanmax(stamp_inner))
                res_lim = float(np.nanmax(np.abs(res_inner))) * 1.05

                titles = ['Data', 'PSF model', 'Residual']
                arrays = [stamp, model, residual]
                cmaps = ['gray', 'gray', 'RdBu_r']
                vlims = [(vlo, vhi), (vlo, vhi), (-res_lim, res_lim)]

                star_idx = page * stars_per_page + row_i + 1
                param_line1 = (f'Star {star_idx}  |  '
                               f'RA={r.ra:.5f}  Dec={r.dec:.5f}  '
                               f'(x,y)=({r.x_pix:.1f},{r.y_pix:.1f})')
                param_line2 = (f'flux={r.flux_psf:.1f}±{r.e_flux_psf:.1f}  '
                               f'dx={r.dx_fit:.3f}  dy={r.dy_fit:.3f}  '
                               f'bg={r.background:.2f}  '
                               f'Rp_AB={r.rp_ab:.3f}  '
                               f'ZP={r.zp_ab:.4f}±{r.e_zp_ab:.4f} mag')

                for col_i in range(3):
                    ax = axes[row_i, col_i]
                    im = ax.imshow(arrays[col_i], origin='lower', cmap=cmaps[col_i],
                                   vmin=vlims[col_i][0], vmax=vlims[col_i][1],
                                   interpolation='nearest')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(titles[col_i], fontsize=8)
                    ax.set_xlabel('px')
                    ax.set_ylabel('px')

                # Place parameter text inside the data panel to avoid layout issues
                axes[row_i, 1].set_title(
                    f'{titles[1]}\n{param_line1}\n{param_line2}',
                    fontsize=6, pad=4
                )

            fig.suptitle(f'Per-star PSF fits  (page {page+1}/{n_pages})',
                         fontsize=10, y=1.0)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'  Star fit pages:     {fname}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='PSF flux calibration to Gaia Rp AB zeropoint.')
    parser.add_argument('fits_path',
                        help='FITS file with image in HDU[1] and CCD WCS header.')
    parser.add_argument('--sector',   type=int, required=True)
    parser.add_argument('--cam',      type=int, required=True)
    parser.add_argument('--ccd',      type=int, required=True)
    parser.add_argument('--cut-x0',   type=float, default=0.0,
                        help='CCD x-pixel of cut bottom-left corner (from find_cuts).')
    parser.add_argument('--cut-y0',   type=float, default=0.0,
                        help='CCD y-pixel of cut bottom-left corner (from find_cuts).')
    parser.add_argument('--gaia-path', default=GAIA_PATH_DEFAULT)
    parser.add_argument('--prf-path',  default=PRF_PATH_DEFAULT)
    parser.add_argument('--mag-lo',   type=float, default=12.0)
    parser.add_argument('--mag-hi',   type=float, default=14.0)
    parser.add_argument('--iso-radius', type=float, default=5.0)
    parser.add_argument('--stamp-size', type=int,   default=9)
    parser.add_argument('--n-jobs',   type=int, default=-1)
    parser.add_argument('--plot',     action='store_true')
    parser.add_argument('--savepath', default=None)
    args = parser.parse_args()

    with fits.open(args.fits_path) as f:
        img = f[1].data.astype(float)
        w = WCS(f[1].header)

    zp, zp_e, _ = run_calibration(
        img, w,
        sector=args.sector, cam=args.cam, ccd=args.ccd,
        cut_corner=(args.cut_x0, args.cut_y0),
        gaia_path=args.gaia_path,
        prf_path=args.prf_path,
        mag_lo=args.mag_lo, mag_hi=args.mag_hi,
        iso_radius_pix=args.iso_radius,
        stamp_size=args.stamp_size,
        n_jobs=args.n_jobs,
        plot=args.plot, savepath=args.savepath,
    )
