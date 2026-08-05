from scipy.stats import exponnorm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.spatial import cKDTree
from scipy.ndimage import shift
from astropy.stats import sigma_clipped_stats
import os
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from tessellate.tessellate import Tessellate
from tessellate.dataprocessor import DataProcessor
from tessellate.navigator import Navigator
from tessellate.tools import RoundToInt, _Print_buff

def _emg_placement(K_internal, duty_frac, threshold=0.05, margin=0.05, n_fine=5000):
    """
    Finds (loc, scale) so that the EMG shape's above-`threshold` region sits
    at tau in [margin, margin + target_width] on the normalized [0,1] domain,
    where target_width = duty_frac * (1 - 2*margin).

    Only used to calibrate the shape's placement once per event - the fine
    grid here is on an abstract, resolution-independent tau axis and has
    nothing to do with real cadence/gaps, so it can't reintroduce the old bug.
    """
    ref_tau = np.linspace(-5, 5 + 10 * K_internal, n_fine)
    ref_flux = exponnorm.pdf(ref_tau, K=K_internal, loc=0, scale=1)
    ref_flux /= ref_flux.max()

    above = ref_flux >= threshold
    if not above.any():
        raise ValueError("threshold too high for given K - no region above it")
    ref_width = ref_tau[above].max() - ref_tau[above].min()
    ref_left = ref_tau[above].min()

    target_width = duty_frac * (1 - 2 * margin)
    scale = target_width / ref_width
    loc = margin - ref_left * scale
    return loc, scale, target_width


def _emg_bin_flux(edges_lo, edges_hi, K_internal, loc, scale, mirror):
    """
    Exact analytic integral of the (possibly mirrored) EMG pdf over each
    [edges_lo, edges_hi] bin, via the exponnorm CDF. Works for arbitrarily
    irregular bin spacing - no interpolation, no fine grid needed here.
    """
    if mirror:
        # mirrored shape g(tau) = f(1-tau); integral over [a,b] of g
        # = integral over [1-b,1-a] of f = F(1-a) - F(1-b)
        cdf_lo = exponnorm.cdf(1 - edges_lo, K=K_internal, loc=loc, scale=scale)
        cdf_hi = exponnorm.cdf(1 - edges_hi, K=K_internal, loc=loc, scale=scale)
        return cdf_lo - cdf_hi
    else:
        cdf_lo = exponnorm.cdf(edges_lo, K=K_internal, loc=loc, scale=scale)
        cdf_hi = exponnorm.cdf(edges_hi, K=K_internal, loc=loc, scale=scale)
        return cdf_hi - cdf_lo


def Flare_Shape(K, duty_frac=0.3, n_fine=2000, threshold=0.05, margin=0.05):
    """
    Kept for previewing/plotting a shape on a regular tau grid - no longer
    used internally by Gen_Event (which evaluates the exact analytic CDF
    directly at real timestamps instead).
    """
    K = max(K, 1e-6)
    loc, scale, _ = _emg_placement(K, duty_frac, threshold, margin, n_fine)
    tau_grid = np.linspace(0, 1, n_fine)
    raw = exponnorm.pdf(tau_grid, K=K, loc=loc, scale=scale)
    flux_norm = raw / raw.max()
    return tau_grid, flux_norm

def Gen_Event(K, cadence_min, time_days, event_time_min, duty_frac=0.6,
              threshold=0.05, margin=0.05, n_fine=5000):
    """
    Evaluates the flare/dip shape exactly (analytic EMG CDF) at the actual
    observation times - no fine grid, no interpolation, no assumption of
    regular spacing.

    K: shape in [-1, 1]
    cadence_min: nominal exposure duration of one frame, minutes (bin width)
    time_days: actual MJD timestamps to generate flux for. These should
               already be just the "active" (above-threshold) window - e.g.
               self.nav.time[frame_start:frame_end] as chosen by the
               gap-aware scheduler below.
    event_time_min: the INTENDED active-region duration in minutes (what
               draw_duration_days() drew) - NOT the padded window, and NOT
               shortened by any gap truncation. This must match what was
               used at scheduling time so the shape placement agrees.
    duty_frac, threshold, margin: must match scheduling-time values.

    Returns:
        flux: array, same length as time_days, peak abs deviation = 1
    """
    K = np.clip(K, -1, 1)
    K_max = 8.0
    K_internal = abs(K) * K_max
    mirror = K < 0

    loc, scale, target_width = _emg_placement(K_internal, duty_frac, threshold, margin, n_fine)
    total_window_min = event_time_min / duty_frac
    tau_lo = margin if not mirror else 1 - (margin + target_width)

    t_min = (time_days - time_days[0]) * 1440.0
    tau = tau_lo + t_min / total_window_min
    half_w_tau = (cadence_min / 2.0) / total_window_min

    flux = _emg_bin_flux(tau - half_w_tau, tau + half_w_tau, K_internal, loc, scale, mirror)

    peak = np.max(np.abs(flux))
    if peak > 0:
        flux = flux / peak
    return flux


def Gen_Sinusoid(time_days, period_min, cadence_min, phase=0.0):
    """
    Evaluates the sinusoid's cadence-integrated flux exactly (closed-form
    integral of sin) at the actual observation times. Handles any gaps for
    free - missing frames are simply absent from time_days, not extrapolated
    across.

    time_days: actual MJD timestamps to generate flux for (can span the
               full baseline, gaps and all).
    period_min: oscillation period, minutes.
    cadence_min: nominal exposure duration of one frame, minutes (bin width -
                 NOT inferred from spacing between frames; a gap means
                 missing frames, not a wider exposure).
    phase: starting phase (radians), referenced to time_days[0].

    Returns:
        flux: array, same length as time_days, peak abs deviation = 1
    """
    t_min = (time_days - time_days[0]) * 1440.0
    omega = 2 * np.pi / period_min
    half_w = cadence_min / 2.0

    antideriv = lambda t: -np.cos(omega * t + phase) / omega
    flux = antideriv(t_min + half_w) - antideriv(t_min - half_w)

    peak = np.max(np.abs(flux))
    if peak > 0:
        flux = flux / peak
    return flux

def _Shift_One(frame, s):
	if np.nansum(abs(frame)) > 0:
		return shift(frame, [s[0], s[1]], mode='nearest', order=5)
	return frame

class SourceInjector():

    def __init__(self,sector,cam,ccd,n=8,job_output_path='.',working_path='.',num_cores=None,
                 data_path='/fred/oz335/TESSdata',prf_path='/fred/oz335/_local_TESS_PRFs'):

        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.n = n

        self.job_output_path = job_output_path
        self.working_path = working_path
        self.data_path = data_path
        self.prf_path = prf_path

        self.num_cores = multiprocessing.cpu_count() if num_cores is None else num_cores

        self.path = f'{self.data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}'
        self.nav = Navigator(sector,cam,ccd,data_path,n)



    def _find_injection_sites(self, min_sep=5, edge_buffer=5, grid_step=1):
        """
        Finds valid pixel locations for source injection, defined as points at
        least `min_sep` pixels from every existing detected object and at least
        `edge_buffer` pixels from the cube edge.

        objects_df: DataFrame with existing detected object positions - expects
                    'x' and 'y' columns (pixel coordinates); adjust names below
                    if your Navigator uses different column labels
        cube_shape: (n_frames, ny, nx) or (ny, nx) of the data cube
        min_sep: minimum allowed distance (pixels) from any existing object
        edge_buffer: minimum allowed distance (pixels) from the cube edge
        grid_step: spacing (pixels) of the candidate grid searched over - 1 for
                full resolution, >1 to speed up the search on large cubes

        Returns:
            valid_sites: array of shape (n_valid, 2), (x, y) integer pixel
                        positions satisfying both constraints
        """

        print('    finding injection sites')

        ny, nx = self.nav.flux.shape[-2], self.nav.flux.shape[-1]

        xs = np.arange(edge_buffer, nx - edge_buffer, grid_step)
        ys = np.arange(edge_buffer, ny - edge_buffer, grid_step)
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)

        obj_xy = self.nav.objects[['xcentroid', 'ycentroid']].dropna().values
        if len(obj_xy) == 0:
            return candidates.astype(int)

        tree = cKDTree(obj_xy)
        dist, _ = tree.query(candidates, k=1)

        valid_sites = candidates[dist >= min_sep].astype(int)
        return valid_sites

    def _frame_window_with_gap_cutoff(self, frame_start, max_duration_days, cadence_min, gap_factor=10.0):
        """
        Walks forward from frame_start through the real self.nav.time array
        and returns frame_end (exclusive) such that:
          - elapsed real time from frame_start doesn't exceed max_duration_days
          - no gap between consecutive included frames exceeds
            gap_factor * cadence_min
        i.e. the window is cut short - not padded or rejected - the moment
        it runs into a big gap or the end of the baseline.
        """
        time = self.nav.time
        n_frames = len(time)
        if frame_start >= n_frames - 1:
            return min(frame_start + 1, n_frames)

        gap_thresh_days = gap_factor * (cadence_min / 1440.0)
        t0 = time[frame_start]

        frame_end = frame_start + 1
        while frame_end < n_frames:
            if time[frame_end] - time[frame_end - 1] > gap_thresh_days:
                break
            if time[frame_end] - t0 > max_duration_days:
                break
            frame_end += 1
        return frame_end


    def schedule_injections(self, valid_sites, n_events,
                        duration_range_min=(10, 1440), duration_skew=(1.0, 2.5),
                        K_range=(-1, 1), K_skew=(1.0, 1.0), p_negative_K=0.05,
                        duty_frac_range=(0.05, 0.95), duty_frac_skew=(1.0, 1.0),
                        stamp_area=25, max_frame_fill_frac=0.25,
                        overlap_dist_px=5, max_attempts_per_event=200, rng=None,
                        type_probs=(0.7, 0.2, 0.10),
                        K_negative_range=(-0.2, 0.2),
                        period_range_min=(20, None), period_mode_min=120.0,
                        period_concentration=6.0,
                        gap_factor=10.0):
        """
        Randomly schedules n_events injections across space and time, allowing
        overlap (spatial and/or temporal) but capping how much of any single
        frame's pixel area is occupied by injected sources at once. Flags events
        within `overlap_dist_px` of another event during an overlapping time
        window. Assigns a uniform random sub-pixel offset within the chosen pixel.

        Each event is one of three types, drawn according to `type_probs`
        (probabilities for 'flare', 'negative', 'sinusoid' respectively - must
        sum to 1):
            'flare'    : the existing EMG flare/variable-peak shape (positive-going)
            'negative' : a dip - same EMG machinery, but K is drawn from the
                        narrow `K_negative_range` (near-Gaussian) and the
                        injected flux is negative-going
            'sinusoid' : a continuous oscillation, always spanning the FULL
                        TESS temporal baseline (mjd_start/mjd_end = the first/
                        last available timestamps) with hard edges, at a
                        period drawn skewed toward a couple hours but ranging
                        from `period_range_min[0]` up to (potentially) well
                        beyond the baseline length itself

        Duration, K, and duty_frac are each drawn via a Beta(a, b) distribution
        over their respective ranges (Beta(1,1) = uniform), letting you bias
        toward particular regimes by default. Period is drawn via a Beta(a, b)
        parameterized directly by a target mode (`period_mode_min`) and a
        concentration (`period_concentration`), rather than raw (a, b), since
        the period range spans orders of magnitude (20 min to the full
        multi-week baseline) and a fixed (a, b) tuned for a narrow range
        wouldn't reliably land the peak near a couple hours once the range
        changes with the sector's baseline length.

        valid_sites: array (n_sites, 2), (x, y) candidate positions
        n_events: number of events to schedule
        duration_range_min: (min, max) event duration IN MINUTES - used for
                    'flare'/'negative' events only; sinusoids ignore this and
                    always span the full baseline
        duration_skew: (a, b) Beta params on normalized log-duration axis;
                    a < b biases toward shorter events
        K_range: (min, max) shape parameter range, used for 'flare' events
        K_skew: (a, b) Beta params on the positive-K portion, normalized to
                [0,1] then mapped to [0, K_range[1]]; a > b biases toward high K
        p_negative_K: probability a 'flare' event is drawn from the negative K
                    range (i.e. left-skewed shape - NOT the 'negative' type)
        duty_frac_range: (min, max) allowed duty_frac, used for 'flare'/'negative'
        duty_frac_skew: (a, b) Beta params on normalized duty_frac axis;
                        (1,1) = uniform
        stamp_area: approximate PSF footprint (pixels)
        max_frame_fill_frac: max fraction of a frame's pixels occupied at once
        overlap_dist_px: distance (pixels) defining spatial overlap flag
        max_attempts_per_event: retries per event before giving up
        rng: np.random.Generator, or None to create a default one
        type_probs: (p_flare, p_negative, p_sinusoid), must sum to 1
        K_negative_range: (min, max) K range for 'negative' events - kept
                        narrow/near-zero so dips are close to Gaussian
        period_range_min: (min, max) oscillation period IN MINUTES, used for
                    'sinusoid' events. If max is None, it's auto-set to 2x
                    the sector's temporal baseline (in minutes), so some
                    periods extend past the full baseline.
        period_mode_min: the period (minutes) the distribution peaks at -
                    default 120 (2 hr). Must lie within period_range_min.
        period_concentration: >2, controls how peaked the distribution is
                    around period_mode_min. Larger = tighter clustering
                    around the mode; near 2 = closer to uniform-in-log.

        Returns:
            schedule_df: DataFrame with columns:
                eventid, event_type, xcentroid, ycentroid, snr, K, duty_frac,
                period_min, phase, frame_start, frame_end, frame_duration,
                mjd_start, mjd_end, mjd_duration, overlap
        """

        print('    generating injection properties')

        if rng is None:
            rng = np.random.default_rng()

        assert abs(sum(type_probs) - 1.0) < 1e-6, "type_probs must sum to 1"

        n_frames, ny, nx = self.nav.flux.shape
        frame_area = valid_sites.shape[0]
        max_occupied_px = max_frame_fill_frac * frame_area

        min_to_day = 1 / 1440.0
        log_dur_min = np.log10(duration_range_min[0] * min_to_day)
        log_dur_max = np.log10(duration_range_min[1] * min_to_day)

        t_min, t_max = self.nav.time.min(), self.nav.time.max()
        baseline_days = t_max - t_min
        baseline_min = baseline_days * 1440.0
        cadence_min = np.nanmedian(np.diff(self.nav.time)) * 1440

        period_lo, period_hi = period_range_min
        if period_hi is None:
            period_hi = baseline_min * 2.0  # allow periods well past the baseline
        log_per_min = np.log10(period_lo)
        log_per_max = np.log10(period_hi)

        log_mode = np.log10(period_mode_min)
        assert log_per_min <= log_mode <= log_per_max, \
            "period_mode_min must lie within period_range_min"
        u_mode = (log_mode - log_per_min) / (log_per_max - log_per_min)

        k = max(period_concentration, 2.0001)  # keep a,b > 1 so mode formula holds
        a_per = 1 + u_mode * (k - 2)
        b_per = 1 + (1 - u_mode) * (k - 2)

        occupancy = np.zeros(n_frames)

        def draw_event_type():
            return rng.choice(['flare', 'negative', 'sinusoid'], p=type_probs)

        def draw_duration_days():
            u = rng.beta(duration_skew[0], duration_skew[1])
            log_dur = log_dur_min + u * (log_dur_max - log_dur_min)
            return 10 ** log_dur

        def draw_K(event_type):
            if event_type == 'negative':
                return rng.uniform(K_negative_range[0], K_negative_range[1])
            if rng.uniform() < p_negative_K:
                return rng.uniform(K_range[0], 0)
            u = rng.beta(K_skew[0], K_skew[1])
            return u * K_range[1]

        def draw_duty_frac():
            u = rng.beta(duty_frac_skew[0], duty_frac_skew[1])
            return duty_frac_range[0] + u * (duty_frac_range[1] - duty_frac_range[0])

        def draw_period_min():
            u = rng.beta(a_per, b_per)
            log_per = log_per_min + u * (log_per_max - log_per_min)
            return 10 ** log_per

        def draw_snr():
            if rng.random() < 0.60:          # 75% of draws concentrated in 3-10
                return rng.uniform(1,10)
            else:                             # 25% draws from full range (covers tails)
                return 10**(rng.uniform(1, 2))
        
        rows = []
        eventid = 0 
        while eventid < n_events:
            placed = False
            event_type = draw_event_type()
            for _attempt in range(max_attempts_per_event):
                site = valid_sites[rng.integers(0, len(valid_sites))]
                xfrac, yfrac = rng.uniform(0, 1, size=2)
                xcentroid = site[0] + xfrac
                ycentroid = site[1] + yfrac

                K = np.nan
                duty_frac = np.nan
                period_min = np.nan
                phase = np.nan
                event_time_min = np.nan

                if event_type == 'sinusoid':
                    period_min = draw_period_min()
                    phase = rng.uniform(0, 2 * np.pi)
                    frame_start, frame_end = 0, n_frames
                    mjd_start, mjd_end = t_min, t_max
                else:
                    event_time_days = draw_duration_days()
                    K = draw_K(event_type)
                    duty_frac = draw_duty_frac()
                    event_time_min = event_time_days * 1440.0

                    mjd_start_raw = t_min + rng.uniform(0, 1) * (t_max - t_min)
                    frame_start = int(np.searchsorted(self.nav.time, mjd_start_raw, side='left'))
                    if frame_start >= n_frames:
                        continue

                    frame_end = self._frame_window_with_gap_cutoff(
                        frame_start, event_time_days, cadence_min, gap_factor=gap_factor
                    )

                    mjd_start = self.nav.time[frame_start]
                    mjd_end = self.nav.time[frame_end - 1]

                projected = occupancy[frame_start:frame_end] + stamp_area
                if np.any(projected > max_occupied_px):
                    continue

                occupancy[frame_start:frame_end] += stamp_area
                snr = draw_snr()

                conflict = False
                for r in rows:
                    time_overlap = (
                        frame_start < r['frame_end']
                        and frame_end > r['frame_start']
                    )

                    if not time_overlap:
                        continue

                    dist = np.hypot(
                        xcentroid - r['xcentroid'],
                        ycentroid - r['ycentroid']
                    )

                    if dist <= overlap_dist_px:
                        conflict = True
                        break

                if conflict:
                    continue 

                rows.append({
                    'event_type': event_type,
                    'xcentroid': xcentroid,
                    'ycentroid': ycentroid,
                    'snr': snr,
                    'K': K,
                    'duty_frac': duty_frac,
                    'period_min': period_min,
                    'phase': phase,
                    'event_time_min': event_time_min,   # NEW: intended active duration (flare/negative only)
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'frame_duration': frame_end - frame_start,
                    'mjd_start': mjd_start,              # now the ACTUAL timestamp of frame_start
                    'mjd_end': mjd_end,                  # now the ACTUAL timestamp of frame_end-1
                    'mjd_duration': mjd_end - mjd_start,  # now the ACTUAL injected span
                })
                placed = True
                eventid += 1
                break

            if not placed:
                print(f"Warning: event {eventid} ({event_type}) could not be placed within "
                    f"{max_attempts_per_event} attempts under the fill-fraction cap")

        schedule_df = pd.DataFrame(rows)
        return schedule_df

    def inject_sources(self,cut,n_events,
                    min_sep=5,edge_buffer=5,grid_step=1,big_size=15,small_size=5,
                    duration_range_min=(10, 1440), duration_skew=(1.0, 2.5),
                        K_range=(-1, 1), K_skew=(1.0, 1.0), p_negative_K=0.05,
                        duty_frac_range=(0.05, 0.95), duty_frac_skew=(1.0, 1.0),
                        stamp_area=5, max_frame_fill_frac=0.25,
                        overlap_dist_px=5, max_attempts_per_event=200,
                        type_probs=(0.6, 0.2, 0.2),
                        K_negative_range=(-0.2, 0.2),
                        period_range_min=(20, None), period_mode_min=240.0,
                        period_concentration=3.0):


        from PRF import TESS_PRF
            
        self.nav.gather_data(cut=cut,flux=True,time=True,bkg=True,verbose=False)
        self.nav.gather_results(cut=cut,sources=False,events=True,objects=True)
        raw_cube = self.nav.flux #+ self.nav.bkg

        # -- Generate PRF -- #
        dp = DataProcessor(self.sector,data_path=self.data_path)
        cutCornerPx, cutCentrePx, _, _ = dp.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        column = cutCentrePx[cut-1][0]
        row = cutCentrePx[cut-1][1]
        if self.sector < 4:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{self.prf_path}/Sectors1_2_3')
        else:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{self.prf_path}/Sectors4+')
                

        valid_sites = self._find_injection_sites(min_sep,edge_buffer,grid_step)

        injections = self.schedule_injections(valid_sites,n_events,
                                                duration_range_min,duration_skew,
                                                K_range, K_skew, p_negative_K,
                                                duty_frac_range, duty_frac_skew,
                                                stamp_area, max_frame_fill_frac,
                                                overlap_dist_px, max_attempts_per_event,
                                                type_probs=type_probs,
                                                K_negative_range=K_negative_range,
                                                period_range_min=period_range_min,
                                                period_mode_min=period_mode_min,
                                                period_concentration=period_concentration)

        injections['frame_max'] = 0
        cadence_min = np.nanmedian(np.diff(self.nav.time)) * 1440
        lcs = []
        for i in tqdm(range(n_events), desc='    injecting events into cube', position=0, leave=True, dynamic_ncols=False, ascii=True):
            source = injections.iloc[i]

            time_days = self.nav.time[source.frame_start:source.frame_end]
            if len(time_days) == 0:
                continue

            if source.event_type == 'sinusoid':
                flux = Gen_Sinusoid(time_days, source.period_min, cadence_min, phase=source.phase)
                sign = 1.0
            else:
                flux = Gen_Event(source.K, cadence_min, time_days, source.event_time_min, source.duty_frac)
                sign = -1.0 if source.event_type == 'negative' else 1.0

            frames = np.arange(source.frame_start, source.frame_end)
            ref_idx = np.argmax(np.abs(flux))
            max_frame = frames[ref_idx]
            injections.iloc[i, injections.columns.get_loc('frame_max')] = max_frame
            
            xint = RoundToInt(source.xcentroid)
            yint = RoundToInt(source.ycentroid)

            half_big = big_size // 2
            h, w = self.nav.flux.shape[1], self.nav.flux.shape[2]

            y1 = yint - half_big        # Desired bounds in full image
            y2 = yint + half_big + 1
            x1 = xint - half_big
            x2 = xint + half_big + 1
        
            yy1, yy2 = max(0, y1), min(h, y2)   # Clip to image bounds
            xx1, xx2 = max(0, x1), min(w, x2)
        
            cut = np.full((big_size, big_size), np.nan, dtype=np.float32)   # Create NaN-padded cut
    
            cy1 = yy1 - y1
            cy2 = cy1 + (yy2 - yy1)
            cx1 = xx1 - x1
            cx2 = cx1 + (xx2 - xx1)
    
            cut[cy1:cy2, cx1:cx2] = self.nav.flux[max_frame, yy1:yy2, xx1:xx2] 
        
            valid = cut[~np.isnan(cut)]     # Compute noise only on valid pixels
            if valid.size == 0:
                continue
            _, _, noise = sigma_clipped_stats(valid, sigma=3)

            npix = 9
            b = -source.snr**2 / 600
            c = -source.snr**2 * npix * noise**2
            peak_flux = (-b + np.sqrt(b**2 - 4*c)) / 2
            peak_flux *= sign

            flux *= peak_flux
            lcs.append(np.array([frames,flux]))

            image = prf.locate(2 + (source.xcentroid - RoundToInt(source.xcentroid)),
                                2 + (source.ycentroid - RoundToInt(source.ycentroid)),
                                (5, 5))
            
            for j, f in enumerate(flux):

                image_frame = image.copy() * f / np.nansum(image[1:4, 1:4])

                raw_cube[frames[j], yint-2:yint+3, xint-2:xint+3] += image_frame

        return raw_cube,injections,lcs

    def apply_shifts(self,shifts,cube):

        print('    applying shifts')        

        result = Parallel(n_jobs=self.num_cores)(
					delayed(_Shift_One)(cube[i], -1*shifts[i])
					for i in tqdm(range(len(cube)), position=0, leave=True,dynamic_ncols=False,ascii=True))

        return np.array(result)

    def run(self,cut,n_events,overwrite=False,
            min_sep=5,edge_buffer=5,grid_step=1,big_size=15,small_size=5,
            duration_range_min=(10, 1440), duration_skew=(1.0, 2.5),
            K_range=(-1, 1), K_skew=(1.0, 1.0), p_negative_K=0.05,
            duty_frac_range=(0.05, 0.95), duty_frac_skew=(1.0, 1.0),
            stamp_area=25, max_frame_fill_frac=0.25,
            overlap_dist_px=5, max_attempts_per_event=200,
            type_probs=(0.6, 0.2, 0.2),
            K_negative_range=(-0.2, 0.2),
            period_range_min=(20, None), period_mode_min=240.0,
            period_concentration=3.0):

        

        if cut == 'all':
            cuts = np.arange(1,self.n**2+1).astype(int)
        else:
            cuts = [cut]

        _Print_buff(60,f'Running Source Injection for Sector{self.sector} Cam{self.cam} Ccd{self.ccd}')
        print('\n')

        for cut in cuts: 

            print(f'Cut {cut}')     

            directory = f'{self.path}/Cut{cut}of{self.n**2}'
            base_name = f'sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}'      

            go = False
            if not os.path.exists(f'{directory}/source_injection/{base_name}_RawFlux.npy'): 
                go = True
            elif overwrite:
                os.system(f'rm -r {directory}/source_injection')
                go = True

            if go:
                rawcube,injections,lcs = self.inject_sources(cut,n_events,
                        min_sep,edge_buffer,grid_step,big_size,small_size,
                        duration_range_min, duration_skew,
                            K_range, K_skew, p_negative_K,
                            duty_frac_range, duty_frac_skew,
                            stamp_area, max_frame_fill_frac,
                            overlap_dist_px, max_attempts_per_event,
                            type_probs=type_probs,
                            K_negative_range=K_negative_range,
                            period_range_min=period_range_min,
                            period_mode_min=period_mode_min,
                            period_concentration=period_concentration)

                orbit_segments = np.load(f'{directory}/{base_name}_OrbitSegments.npy')
                orbit_refs = np.load(f'{directory}/{base_name}_OrbitRefs.npz')
                orbit_refs = {int(k): orbit_refs[k] for k in orbit_refs.files}
                ref = np.load(f'{directory}/{base_name}_Ref.npy')
                shifts = np.load(f'{directory}/{base_name}_Shifts.npy')

                rawcube[orbit_segments==1] += orbit_refs[1]
                rawcube[orbit_segments==2] += orbit_refs[2]
                rawcube += ref

                shifted_cube = self.apply_shifts(shifts,rawcube)

                lcs_arr = np.empty(len(lcs), dtype=object)
                for i, lc in enumerate(lcs):
                    lcs_arr[i] = lc

                os.makedirs(f'{directory}/source_injection',exist_ok=True)
                np.savez(f'{directory}/source_injection/lightcurves.npz', lcs=lcs_arr)
                injections.to_csv(f'{directory}/source_injection/injected_events.csv',index=False)
                np.save(f'{directory}/source_injection/{base_name}_RawFlux.npy',shifted_cube)

            run = Tessellate(data_path=self.data_path,working_path=self.working_path,job_output_path=self.job_output_path,
                                sector=self.sector,cam=self.cam,ccd=self.ccd,n=self.n,cuts=cut,
                                download=False,make_cube=False,fix_wcs=False,make_cuts=False,calibrate=False,
                                reduce=True,search=True,injection=True,plot=False,delete=False,
                                reset_logs=False,overwrite=False,ask_config=False,save_config=False,use_suggestions=True)



    def match_results_to_injections(self,centroid_match_radius=1):

        self.injections['ev_match'] = '-'
        self.injections['detected'] = 'n'

        for i, ev in self.injections.iterrows():
            if ev.event_type != 'sinusoid':     # cant deal with variables just yet

                start = int(ev.frame_start)
                end = int(ev.frame_end)

                # check for detection in isolated single frames
                mask = (
                    (self.nav.isolated.frame >= start)
                    & (self.nav.isolated.frame <= end)   # or < end if you want exclusive, but be explicit
                    & (abs(self.nav.isolated.xcentroid - ev.xcentroid)<centroid_match_radius)
                    & (abs(self.nav.isolated.ycentroid - ev.ycentroid)<centroid_match_radius)
                )

                isolated_sources = self.nav.isolated[mask]
                if len(isolated_sources) > 0:
                    self.injections.loc[i, 'detected'] = 'iso'

                mask = (
                        (self.nav.events.frame_bin == 1)
                        & (self.nav.events.frame_max >= start)
                        & (self.nav.events.frame_max <= end)   # or < end if you want exclusive, but be explicit
                        & (abs(self.nav.events.xcentroid - ev.xcentroid)<centroid_match_radius)
                        & (abs(self.nav.events.ycentroid - ev.ycentroid)<centroid_match_radius)
                    )

                events = self.nav.events[mask]
                if len(events) > 0:
                    if len(events) == 1:
                        loc = 0
                    else:
                        duration_diffs = np.fmax(events.frame_duration, ev.frame_duration) / np.fmin(events.frame_duration, ev.frame_duration)
                        loc = np.argmin(duration_diffs)

                    self.injections.loc[i, 'ev_match'] = f"{events.iloc[loc].objid}_{events.iloc[loc].eventid}"
                    self.injections.loc[i, 'detected'] = 'y'

                else:
                    if len(isolated_sources) == 0:
                        found = False
                        frame_bins = np.unique(self.nav.events.frame_bin)
                        frame_bins = frame_bins[frame_bins>1]
                        i = 0
                        while not found:
                            frame_bin = frame_bins[i]
                            mask = (
                                (self.nav.events.frame_bin == frame_bin)
                                & (self.nav.events.frame_max * frame_bin >= start)
                                & (self.nav.events.frame_max * frame_bin <= end)   # or < end if you want exclusive, but be explicit
                                & (abs(self.nav.events.xcentroid - ev.xcentroid)<centroid_match_radius)
                                & (abs(self.nav.events.ycentroid - ev.ycentroid)<centroid_match_radius)
                            )

                            events = self.events[mask]
                            if len(events) > 0:
                                found = True
                                if len(events) == 1:
                                    loc = 0
                                else:
                                    duration_diffs = np.fmax(events.frame_duration*frame_bin, ev.frame_duration) / np.fmin(events.frame_duration*frame_bin, ev.frame_duration)
                                    loc = np.argmin(duration_diffs)

                                self.injections.loc[i, 'ev_match'] = f"{events.iloc[loc].objid}_{events.iloc[loc].eventid}"
                                self.injections.loc[i, 'detected'] = str(frame_bin)


    def match_vars_to_injections(self,min_samples_per_cycle=3):

        sin_mask = self.injections.event_type == 'sinusoid'

        for i in self.injections.index[sin_mask]:
            ev = self.injections.loc[i]

            t = self.nav.time[ev.frame_start:ev.frame_end]
            elapsed_min = (t - ev.mjd_start) * 1440.0

            half_period_min = ev.period_min / 2.0
            phase_offset_min = (ev.phase / (2 * np.pi)) * ev.period_min

            extremum_idx = np.floor((elapsed_min + phase_offset_min) / half_period_min).astype(int)

            counts = np.bincount(extremum_idx - extremum_idx.min())
            min_samples_per_cycle = min(min_samples_per_cycle,np.nanmax(counts))

            n_visible = np.sum(counts >= min_samples_per_cycle)


        

    def gather_results(self,cut):

        self.nav = Navigator(self.sector,self.cam,self.ccd,self.data_path,self.n,injection=True)
        self.nav.gather_data(cut=cut)
        self.nav.gather_results(cut=cut,isolated=True)

        directory = f'{self.path}/Cut{cut}of{self.n**2}'

        self.injections = pd.read_csv(f'{directory}/source_injection/injected_events.csv')

        self.match_results_to_injections()
        self.match_vars_to_injections()
