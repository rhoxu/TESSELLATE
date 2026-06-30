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


def fit_bazin(t, f, ferr=None, p0=None, maxfev=20000):
    """
    Fit the Bazin profile to (t, f) with optional errors.

    Returns a dict with the fitted params/errors, chi2/dof/reduced-chi2, the
    amplitude S/N, and the (cleaned) data + model arrays -- or None on failure.
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    sig = None
    if ferr is not None:
        ferr = np.asarray(ferr, dtype=float)
        m &= np.isfinite(ferr) & (ferr > 0)
    t, f = t[m], f[m]
    if ferr is not None:
        sig = ferr[m]
    if len(t) < 6:
        return None

    # Initial guess
    c0 = float(np.median(f))
    imax = int(np.argmax(f - c0))
    A0 = float((f - c0)[imax]) or 1.0
    t0_0 = float(t[imax])
    span = float(t.max() - t.min()) or 1.0
    if p0 is None:
        p0 = [A0, t0_0, 0.05 * span, 0.2 * span, c0]

    lb = [-np.inf, t.min() - span, 1e-3, 1e-3, -np.inf]
    ub = [np.inf, t.max() + span, 5 * span, 10 * span, np.inf]
    p0 = [min(max(v, lo + 1e-6), hi - 1e-6) for v, lo, hi in zip(p0, lb, ub)]

    try:
        popt, pcov = curve_fit(bazin, t, f, p0=p0, sigma=sig,
                               absolute_sigma=sig is not None,
                               bounds=(lb, ub), maxfev=maxfev)
    except Exception:
        return None

    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    model = bazin(t, *popt)
    resid = f - model
    chi2 = float(np.sum((resid / sig) ** 2)) if sig is not None else float(np.sum(resid ** 2))
    dof = len(t) - len(_PARAMS)

    return {
        'params': dict(zip(_PARAMS, [float(v) for v in popt])),
        'perr': dict(zip(_PARAMS, [float(v) for v in perr])),
        'chi2': chi2, 'dof': dof,
        'redchi2': chi2 / dof if dof > 0 else np.nan,
        'n': len(t),
        'A_snr': float(popt[0] / perr[0]) if perr[0] > 0 else np.nan,
        't': t, 'f': f, 'model': model,
    }


def bazin_detection(t, f, ferr=None, **kwargs):
    """
    Fit Bazin and compare it to a flat baseline.

    Adds to the fit dict:
      chi2_flat   : chi2 of the best constant model
      delta_chi2  : chi2_flat - chi2_bazin   (>0 favours Bazin)
      delta_bic   : BIC_bazin - BIC_flat     (<0 favours Bazin)

    delta_chi2 / delta_bic are the detection statistics for ranking how
    convincingly a Bazin-shaped signal is present.
    """
    fit = fit_bazin(t, f, ferr, **kwargs)
    if fit is None:
        return None

    # Rebuild the same cleaning mask used inside fit_bazin, to align the errors
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    sig = None
    if ferr is not None:
        ferr = np.asarray(ferr, dtype=float)
        m &= np.isfinite(ferr) & (ferr > 0)
        sig = ferr[m]
    ff = f[m]

    if sig is not None:
        w = 1.0 / sig ** 2
        c = float(np.sum(w * ff) / np.sum(w))
        chi2_flat = float(np.sum(w * (ff - c) ** 2))
    else:
        c = float(np.mean(ff))
        chi2_flat = float(np.sum((ff - c) ** 2))

    n = fit['n']
    fit['chi2_flat'] = chi2_flat
    fit['delta_chi2'] = chi2_flat - fit['chi2']
    fit['delta_bic'] = (fit['chi2'] + len(_PARAMS) * np.log(n)) - (chi2_flat + 1 * np.log(n))
    return fit
