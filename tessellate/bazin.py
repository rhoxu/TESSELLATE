"""
Bazin profile fitting for tessellate event light curves.

The Bazin (2009) function is a smooth-rise / exponential-decay model widely used
for transient light curves:

    f(t) = A * exp(-(t - t0) / tau_fall) / (1 + exp(-(t - t0) / tau_rise)) + c

This module fits it to a (PSF) light curve and reports the parameters, their
errors, and goodness-of-fit -- including a likelihood-ratio statistic against a
flat baseline so Bazin-shaped events can be ranked against noise.
"""

import numpy as np

_PARAMS = ['A', 't0', 'tau_rise', 'tau_fall', 'c']


def bazin(t, A, t0, tau_rise, tau_fall, c):
    """Bazin profile.  Exponent arguments are clipped for numerical stability."""
    dt = np.asarray(t, dtype=float) - t0
    rise = np.exp(np.clip(-dt / tau_rise, -50, 50))
    fall = np.exp(np.clip(-dt / tau_fall, -50, 50))
    return A * fall / (1.0 + rise) + c


def bazin_features(params):
    """
    Derived, physically meaningful descriptors of a Bazin fit, for clustering.

    Computes the shape numerically (baseline removed) and returns:
      tau_ratio   : tau_fall / tau_rise (rise/decay asymmetry)
      t_peak_off  : peak time relative to t0
      peak        : model peak above baseline
      fwhm        : full width at half maximum
      rise_time   : half-max(rise) -> peak
      decay_time  : peak -> half-max(decay)
      fluence     : integral of the shape (peak flux x time)
    """
    A = params['A']; t0 = params['t0']
    tr = params['tau_rise']; tf = params['tau_fall']
    out = dict(tau_ratio=(tf / tr) if tr > 0 else np.nan,
               t_peak_off=np.nan, peak=np.nan, fwhm=np.nan,
               rise_time=np.nan, decay_time=np.nan, fluence=np.nan)
    if not (np.isfinite(A) and A > 0 and tr > 0 and tf > 0):
        return out

    tt = np.linspace(t0 - 12 * tr, t0 + 18 * tf, 6000)
    shape = bazin(tt, A, t0, tr, tf, 0.0)   # baseline-subtracted
    ip = int(np.argmax(shape))
    peak = float(shape[ip]); tpk = float(tt[ip])
    out['peak'] = peak
    out['t_peak_off'] = tpk - t0
    out['fluence'] = float(np.trapz(shape, tt))
    half = peak / 2.0
    above = shape >= half
    if above.any():
        tlo = float(tt[above][0]); thi = float(tt[above][-1])
        out['fwhm'] = thi - tlo
        out['rise_time'] = tpk - tlo
        out['decay_time'] = thi - tpk
    return out


def bazin_binned(t, A, t0, tau_rise, tau_fall, c, exp_time=0.0, supersample=7):
    """
    Bazin profile averaged over the exposure of each data point.

    Each observation is the mean flux over its integration, not the instantaneous
    model at the bin centre.  For fast events (timescale <~ exp_time) this matters:
    the model is supersampled across [t - exp_time/2, t + exp_time/2] and averaged.
    With exp_time <= 0 this reduces to the instantaneous Bazin.
    """
    t = np.asarray(t, dtype=float)
    if exp_time and exp_time > 0 and supersample and supersample > 1:
        off = (np.arange(supersample) / (supersample - 1) - 0.5) * exp_time
        tt = t[None, :] + off[:, None]
        return bazin(tt, A, t0, tau_rise, tau_fall, c).mean(axis=0)
    return bazin(t, A, t0, tau_rise, tau_fall, c)


def fit_bazin(t, f, ferr=None, p0=None, maxfev=20000, fit_mask=None, fix_c=None,
              exp_time=0.0, supersample=7):
    """
    Fit the Bazin profile to (t, f) with optional errors.  The amplitude A is
    constrained >= 0 (only brightening events are modelled).

    fit_mask : optional boolean array selecting the points to fit (the rest are
        ignored).  fix_c : if given, the baseline offset c is held fixed and only
        the 4 shape parameters (A, t0, tau_rise, tau_fall) are fit.

    Returns a dict with the fitted params/errors, chi2/dof/reduced-chi2, amplitude
    S/N, the number of free parameters, and the (cleaned, fitted) data + model
    arrays -- or None on failure.
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    if ferr is not None:
        ferr = np.asarray(ferr, dtype=float)
        m &= np.isfinite(ferr) & (ferr > 0)
    if fit_mask is not None:
        m &= np.asarray(fit_mask, dtype=bool)
    t, f = t[m], f[m]
    sig = ferr[m] if ferr is not None else None

    n_free = 4 if fix_c is not None else 5
    if len(t) < n_free + 1:
        return None

    c0 = float(fix_c) if fix_c is not None else float(np.median(f))
    imax = int(np.argmax(f - c0))
    A0 = float((f - c0)[imax]) or 1.0
    t0_0 = float(t[imax])
    span = float(t.max() - t.min()) or 1.0

    # Amplitude is constrained non-negative (brightening events only)
    if fix_c is not None:
        func = lambda tv, A, t0, tr, tf: bazin_binned(tv, A, t0, tr, tf, fix_c,
                                                      exp_time, supersample)
        guess = [max(A0, 0.0), t0_0, 0.05 * span, 0.2 * span]
        lb = [0.0, t.min() - span, 1e-3, 1e-3]
        ub = [np.inf, t.max() + span, 5 * span, 10 * span]
    else:
        func = lambda tv, A, t0, tr, tf, cc: bazin_binned(tv, A, t0, tr, tf, cc,
                                                          exp_time, supersample)
        guess = [max(A0, 0.0), t0_0, 0.05 * span, 0.2 * span, c0]
        lb = [0.0, t.min() - span, 1e-3, 1e-3, -np.inf]
        ub = [np.inf, t.max() + span, 5 * span, 10 * span, np.inf]
    if p0 is None:
        p0 = guess
    p0 = [min(max(v, lo + 1e-6), hi - 1e-6) for v, lo, hi in zip(p0, lb, ub)]

    try:
        popt, pcov = curve_fit(func, t, f, p0=p0, sigma=sig,
                               absolute_sigma=sig is not None,
                               bounds=(lb, ub), maxfev=maxfev)
    except Exception:
        return None

    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    if fix_c is not None:
        popt = np.append(popt, fix_c)
        perr = np.append(perr, 0.0)
    model = bazin_binned(t, *popt, exp_time=exp_time, supersample=supersample)
    resid = f - model
    chi2 = float(np.sum((resid / sig) ** 2)) if sig is not None else float(np.sum(resid ** 2))
    dof = len(t) - n_free

    return {
        'params': dict(zip(_PARAMS, [float(v) for v in popt])),
        'perr': dict(zip(_PARAMS, [float(v) for v in perr])),
        'chi2': chi2, 'dof': dof,
        'redchi2': chi2 / dof if dof > 0 else np.nan,
        'n': len(t), 'n_free': n_free,
        'A_snr': float(popt[0] / perr[0]) if perr[0] > 0 else np.nan,
        't': t, 'f': f, 'model': model,
    }


def bazin_detection(t, f, ferr=None, event_window=None, n_durations=3.0,
                    above_frac=0.05, **kwargs):
    """
    Fit Bazin simultaneously (all 5 params, including the baseline offset c) over
    a window of n_durations event durations centred on the detected event window,
    then evaluate the detection statistics over the event region.

    The fit window (default 3 event durations) gives one event duration of
    baseline shoulder on each side, which constrains the global flux offset c
    (the difference-image zero level) in the same simultaneous fit.

    The statistics region = (where the model is above baseline by > above_frac of
    its peak) UNION (the detected event window).

    Adds to the fit dict:
      fit_window        : (t_lo, t_hi) of the fitted region
      offset            : baseline flux offset c
      n_region          : points in the event region
      chi2_region       : Bazin chi2 over the region
      chi2_flat_region  : flat (offset) chi2 over the region
      redchi2_region    : region Bazin reduced chi2
      delta_chi2        : chi2_flat_region - chi2_region   (>0 favours Bazin)
      delta_bic         : BIC_bazin - BIC_flat over region (<0 favours Bazin)
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)

    # Fit window: n_durations event durations, centred on the event window
    if event_window is not None:
        t0w, t1w = event_window
        finite_t = np.sort(t[np.isfinite(t)])
        cad = float(np.median(np.diff(finite_t))) if finite_t.size > 1 else 0.0
        dur = max(t1w - t0w, cad)                      # at least one cadence
        if not (dur > 0):
            dur = 0.05 * (np.nanmax(t) - np.nanmin(t))
        centre = 0.5 * (t0w + t1w)
        half = 0.5 * n_durations * dur
        roi = np.isfinite(t) & (t >= centre - half) & (t <= centre + half)
        fit_window = (centre - half, centre + half)
    else:
        roi = np.isfinite(t)
        fit_window = (float(np.nanmin(t)), float(np.nanmax(t)))

    fit = fit_bazin(t, f, ferr, fit_mask=roi, **kwargs)
    if fit is None:
        return None

    tc, fc, model = fit['t'], fit['f'], fit['model']
    c = fit['params']['c']

    # Errors aligned to the fitted (roi) data
    m = np.isfinite(t) & np.isfinite(f) & roi
    sig = None
    if ferr is not None:
        ferr = np.asarray(ferr, dtype=float)
        m &= np.isfinite(ferr) & (ferr > 0)
        sig = ferr[m]

    # Event region (within the fit window): model above baseline + detected window
    bump = model - c
    peak = np.nanmax(bump) if abs(np.nanmax(bump)) >= abs(np.nanmin(bump)) else np.nanmin(bump)
    region = (bump / peak) > above_frac if peak != 0 else np.zeros(len(tc), dtype=bool)
    if event_window is not None:
        region |= (tc >= t0w) & (tc <= t1w)
    if region.sum() < 3:
        region = np.ones(len(tc), dtype=bool)

    fr, mr = fc[region], model[region]
    if sig is not None:
        wr = 1.0 / sig[region] ** 2
        chi2_b = float(np.sum(wr * (fr - mr) ** 2))
        chi2_0 = float(np.sum(wr * (fr - c) ** 2))
    else:
        chi2_b = float(np.sum((fr - mr) ** 2))
        chi2_0 = float(np.sum((fr - c) ** 2))

    nreg = int(region.sum())
    k_extra = 4   # Bazin shape params over the flat-offset null (c is shared)
    fit['fit_window'] = fit_window
    fit['offset'] = float(c)
    fit['n_region'] = nreg
    fit['chi2_region'] = chi2_b
    fit['chi2_flat_region'] = chi2_0
    fit['redchi2_region'] = chi2_b / (nreg - len(_PARAMS)) if nreg > len(_PARAMS) else np.nan
    fit['delta_chi2'] = chi2_0 - chi2_b
    fit['delta_bic'] = (chi2_b - chi2_0) + k_extra * np.log(nreg)
    return fit
