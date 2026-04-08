from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.optimize import lsq_linear, minimize_scalar
from copy import deepcopy

# ------------ Functions for Finding Asteroids ------------ # 

def Straight_Line_Asteroid_Checker(time,flux,events):
    """
    Check if a stacked image makes an identifiable straight line. Only operates on maximum resolution events.
    """
    from astropy.stats import sigma_clipped_stats
    import cv2

    events = deepcopy(events)
    for i in range(len(events)):
        event = events.iloc[i]
        if (event['classification'] != 'Asteroid') & (event['frame_bin']==events['frame_bin'].min()):                
            frameStart = event['frame_start']
            frameEnd = event['frame_end']
            x = event['xint']; y = event['yint']
            h, w = flux.shape[1], flux.shape[2]

            if (x-5<0)|(y-5<0)|(x+5>=w)|(y+5>=h):
                continue

            xl = x - 5; xu = x + 6
            yl = y - 5; yu = y + 6
            fs = np.max((frameStart - 5, 0))
            fe = np.min((frameEnd + 5, len(time)-1))

            fs = int(fs)
            fe = int(fe)
            xl = int(xl)
            yl = int(yl)
            yu = int(yu)
            xu = int(xu)

            image = flux[fs:fe,yl:yu,xl:xu]
            image = np.nanmax(image,axis=0)
            image = image / image[5,5] * 255
            
            image[image > 255] = 255
            mean, med, std = sigma_clipped_stats(image,maxiters=10,sigma_upper=2)
            edges = (image > med + 5*std).astype('uint8')

            lines = cv2.HoughLinesP(edges, # Input edge image
                                    1, # Distance resolution in pixels
                                    np.pi/180, # Angle resolution in radians
                                    threshold=10, # Min number of votes for valid line
                                    minLineLength=8, # Min allowed length of line
                                    maxLineGap=0 # Max allowed gap between line for joining them
                                    )

            if (lines is not None) & (event['psf_like']>=0.8):
                events.iloc[i, events.columns.get_loc('classification')] = 'Asteroid'
                # events.iloc[i, events.columns.get_loc('prob')] = 0.5
        
    return events


def Calculate_COM_Motion(flux,candidates):

    from .tools import Distance
    from scipy.ndimage import center_of_mass as COM

    distances = []
    for i in range(len(candidates)):
        event = candidates.iloc[i]
        x = int(event['xint'])
        y = int(event['yint'])
        frameStart = int(event['frame_start'])
        frameEnd = int(event['frame_end'])

        f = flux[frameStart:frameEnd+1,y-1:y+2,x-1:x+2]

        coms = []
        maxflux = np.max(np.nansum(f,axis=(1,2)))
        for frame in f:
            if np.sum(frame) >= maxflux/2:
                com = COM(frame)
                coms.append(com)
        if len(coms)>1:
            distances.append(Distance(coms[-1],coms[0]))
        else:
            distances.append(0)
    candidates['com_motion'] = distances
    return candidates

def Gaussian_Score(time,flux,candidates):

    from .tools import Gaussian, Generate_LC
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    r2s = []
    for i in range(len(candidates)):
        try:
            event = candidates.iloc[i]
            x = int(event['xint'])
            y = int(event['yint'])
            frameStart = int(event['frame_start'])
            frameEnd = int(event['frame_end'])

            frameStart = np.max([frameStart-10,0])
            frameEnd = np.min([frameEnd+11,len(time)-1])

            t,f = Generate_LC(time,flux,x,y,frameStart,frameEnd,radius=1.5)

            p0 = [
                np.max(f) - np.min(f),                     # A: positive height of the bump
                t[np.argmax(f)],                           # t0: time of peak flux
                (np.max(t) - np.min(t)) / 2,              # sigma: rough width guess
                np.min(f)                                     # offset: estimated baseline
            ]

            bounds = (
                [0, np.min(t), 10/24/60, -np.inf],       # lower bounds: A ≥ 0, σ ≥ 15
                [np.inf, np.max(t), np.inf, np.inf]  # upper bounds
            )

            params, _ = curve_fit(Gaussian, t, f, p0=p0, bounds=bounds)
            fit_flux_gaussian = Gaussian(t, *params)
            r2 = r2_score(f, fit_flux_gaussian)

            r2s.append(r2)
        except:
            r2s.append(0)
        
    r2s = np.array(r2s)
    r2s[r2s<0]=0

    candidates['gaussian_score']=r2s

    return candidates


def Threshold_Asteroid_Checker(time,flux,events,com_motion_thresholds=[1, 0.75, 0.5], gaussian_score_thresholds=[0, 0.7, 0.9]):
    """
    Check the centre of mass motion and gaussianity of light curve. Only searches fastest time resolution.
    """

    events = deepcopy(events)

    # -- Identify candidates to actually compute on -- #
    candidates = events[(events['frame_duration']>=2)&
                        (events['frame_duration']<=50)&
                        (events['lc_sig_max']>=5)&
                        (events['flux_sign']==1)& 
                        (events['frame_bin']==events.frame_bin.min())]

    candidate_indices = candidates.index

    # -- Run motion and Gaussian score only on those candidates -- #
    candidates = Calculate_COM_Motion(flux,candidates)
    candidates = Gaussian_Score(time,flux,candidates)

    # -- Add results back to full events -- #
    events['com_motion'] = np.nan
    events['gaussian_score'] = np.nan
    events.loc[candidate_indices, 'com_motion'] = candidates['com_motion'].values
    events.loc[candidate_indices, 'gaussian_score'] = candidates['gaussian_score'].values

    # -- Flagging loop -- #
    events['classification'] = '-'
    for com_thresh, gauss_thresh in zip(com_motion_thresholds, gaussian_score_thresholds):
        mask = (
            (events['com_motion'] >= com_thresh) &
            (events['gaussian_score'] >= gauss_thresh)
        )
        events.loc[mask, 'classification'] = 'Asteroid'
        # events.loc[mask, 'prob'] = 0.8

    return events



# ------------ Functions for Organising Asteroids Into Separate Tracks ------------ # 

@dataclass
class AsteroidTrack:
    asteroid_id: int
    t_ref: float
    x0: float
    vx: float
    y0: float
    vy: float
    t_min: float
    t_max: float
    n_points: int
    residual_px: float
    ax: float = 0.0
    ay: float = 0.0

    def predict(self, t):
        dt = np.asarray(t, dtype=float) - self.t_ref
        x = self.x0 + self.vx * dt + 0.5 * self.ax * dt**2
        y = self.y0 + self.vy * dt + 0.5 * self.ay * dt**2
        return x, y

    def distance_to(self, x, y, t):
        px, py = self.predict(t)
        return np.sqrt((x - px)**2 + (y - py)**2)
    

def cluster_tracks(df_ast, eps_spatial, eps_temporal, min_samples):
    """
    Runs DBSCAN on known asteroids as an initial clustering
    """

    if df_ast.empty: return pd.Series(dtype=int)
    t_c = df_ast['mjd_max'].values - df_ast['mjd_max'].mean()

    X = np.column_stack([df_ast['xcentroid'].values / eps_spatial, 
                         df_ast['ycentroid'].values / eps_spatial, 
                         t_c / eps_temporal])
    
    return pd.Series(DBSCAN(eps=1.0, min_samples=min_samples).fit_predict(X), index=df_ast.index, name='asteroid_id')


def fit_track(tid, group, method, max_accel):
    """
    Fits a linear/quadratic to events with the same track ID.
    """

    t0 = group['mjd_max'].mean()
    dt = (group['mjd_max'] - t0).values
    A = np.column_stack([np.ones_like(dt), dt, 0.5 * dt**2])
    
    if method == 'linear' or len(group) < 3:
        res_x = lsq_linear(A[:, :2], group['xcentroid'])
        res_y = lsq_linear(A[:, :2], group['ycentroid'])
        x0, vx = res_x.x
        y0, vy = res_y.x
        ax, ay = 0.0, 0.0
    else:
        lb = [-np.inf, -np.inf, -max_accel]
        ub = [np.inf, np.inf, max_accel]
        res_x = lsq_linear(A, group['xcentroid'], bounds=(lb, ub))
        res_y = lsq_linear(A, group['ycentroid'], bounds=(lb, ub))
        x0, vx, ax = res_x.x
        y0, vy, ay = res_y.x

    pred_x = x0 + vx * dt + 0.5 * ax * dt**2
    pred_y = y0 + vy * dt + 0.5 * ay * dt**2
    res_px = np.sqrt(np.mean((group['xcentroid'] - pred_x)**2 + (group['ycentroid'] - pred_y)**2))

    return AsteroidTrack(tid, t0, x0, vx, y0, vy, group['mjd_max'].min(), group['mjd_max'].max(), len(group), res_px, ax, ay)


def get_min_spacetime_dist(tr1, tr2, t_range):
    """Finds the minimum distance between two tracks in 3D (X, Y, T)."""

    def separation(t):
        x1, y1 = tr1.predict(t)
        x2, y2 = tr2.predict(t)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    res = minimize_scalar(separation, bounds=t_range, method='bounded')
    return res.fun, res.x


def grow_tracks(df_ast, tracks, eps_pixel):
    """
    Expands track to absorb non clustered asteroids.
    """
    df_grown = df_ast.copy()
    noise_idx = df_grown[df_grown['asteroid_id'] == -1].index
    if len(noise_idx) == 0: return df_grown
    for track in tracks:
        noise_pts = df_grown.loc[noise_idx]
        dist = track.distance_to(noise_pts['xcentroid'], noise_pts['ycentroid'], noise_pts['mjd_max'])
        matches = noise_idx[dist < eps_pixel]
        if len(matches) > 0:
            df_grown.loc[matches, 'asteroid_id'] = track.asteroid_id
            noise_idx = noise_idx.difference(matches)
    return df_grown


def merge_and_refit_tracks(df, tracks, method, max_accel, eps_pixel):
    """
    Merge tracks by average distance of each track's with respect to the other track trajectories.
    """

    df_merged = df.copy()
    sorted_tracks = sorted(tracks, key=lambda t: t.n_points, reverse=True)
    consumed_ids = set()

    for i, big_track in enumerate(sorted_tracks):
        if big_track.asteroid_id in consumed_ids: continue

        for j, small_track in enumerate(sorted_tracks):
            if i == j or small_track.asteroid_id in consumed_ids: continue
            
            small_group = df_merged[df_merged['asteroid_id'] == small_track.asteroid_id]
            dist = big_track.distance_to(small_group['xcentroid'], small_group['ycentroid'], small_group['mjd_max'])
            
            if np.mean(dist) < eps_pixel:
                df_merged.loc[small_group.index, 'asteroid_id'] = big_track.asteroid_id
                consumed_ids.add(small_track.asteroid_id)
            
    return df_merged, [fit_track(tid, g, method, max_accel) for tid, g in df_merged.groupby('asteroid_id') if tid != -1]


def cross_tracks(df_ast,method,max_accel,crossing_dist,spatial_lim):
    """
    Check if tracks get very close to each other in space and time. If they do, merge them.
    Doesnt handle cases where two distinct asteroids actually cross, but that has to be rare
    """

    df_crossed = df_ast.copy()

    tracks = [fit_track(tid, g, method, max_accel) for tid, g in df_ast.groupby('asteroid_id') if tid != -1]
    t_window = (df_crossed['mjd_max'].min(), df_crossed['mjd_max'].max())
    
    # Check for kissing tracks in X,Y,T
    merged_ids = {t.asteroid_id: t.asteroid_id for t in tracks}
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            tr1, tr2 = tracks[i], tracks[j]
            if merged_ids[tr1.asteroid_id] == merged_ids[tr2.asteroid_id]: continue
            
            min_dist, t_cross = get_min_spacetime_dist(tr1, tr2, t_window)

            if min_dist < crossing_dist:
                                # Calculate where the tracks actually are at the crossing time
                x_c, y_c = tr1.predict(t_cross)
                
                # Verify the crossing happens within the sensor bounds
                in_bounds = (0 <= x_c <= spatial_lim) and (0 <= y_c <= spatial_lim)
                
                if in_bounds:
                    old, new = tr2.asteroid_id, tr1.asteroid_id
                    for k, v in merged_ids.items():
                        if v == old: merged_ids[k] = new

    df_crossed['asteroid_id'] = df_crossed['asteroid_id'].map(lambda x: merged_ids.get(x, x))

    return df_crossed


def link_detections(df_unk, tracks, match_radius_px, spatial_lim):
    """
    Follow along each track and give asteroid classification and track id to detections very close to tracks.
    """
    if df_unk.empty or not tracks:
        return df_unk.assign(asteroid_id=-1, residual_px=np.nan)

    x, y, t = df_unk['xcentroid'].values, df_unk['ycentroid'].values, df_unk['mjd_max'].values
    
    # Initialize result arrays
    best_track = np.full(len(df_unk), -1, dtype=int)
    best_dist  = np.full(len(df_unk), np.inf)

    for tr in tracks:
        # 1. Temporal Bound Check: Is the detection time inside the track's life?
        is_inside_time = (t >= tr.t_min) & (t <= tr.t_max)
        
        # 2. Spatial Bound Check: Is the detection inside the CCD area?
        # (Though df_unk usually is by default, we verify against the model)
        px, py = tr.predict(t)
        is_inside_space = (px >= 0) & (px <= spatial_lim) & \
                          (py >= 0) & (py <= spatial_lim)

        # 3. Distance Check
        dist = tr.distance_to(x, y, t)
        
        # Combined Mask: Must be inside time, inside space, and close to the path
        valid_match = is_inside_time & is_inside_space & (dist < match_radius_px)
        
        # Update if this track is the closest one found so far
        better = valid_match & (dist < best_dist)
        best_track[better] = tr.asteroid_id
        best_dist[better]  = dist[better]

    result = df_unk.copy()
    result['asteroid_id'] = best_track
    
    return result


def Tag_Asteroids(df, min_samples=3, spatial_eps=6, temporal_eps=0.5, method='quadratic', 
                          max_accel=1, growth_eps=5,niter=5, merge_eps=10, 
                          min_track_events=10, crossing_dist=2.0,spatial_lim=256, match_radius=2):
    
    # -- Initial Seeds -- #
    df_ast = df[df.classification == 'Asteroid'].copy()
    df_ast['asteroid_id'] = cluster_tracks(df_ast, spatial_eps, temporal_eps, min_samples)

    # -- Grow Asteroid Tracks to absorb unclustered asteroids -- #
    for _ in range(niter):
        tracks = [fit_track(tid, g, method, max_accel) for tid, g in df_ast.groupby('asteroid_id') if tid != -1]
        if not tracks: break
        df_ast = grow_tracks(df_ast, tracks, growth_eps)

    # -- Merge Tracks -- #
    for _ in range(niter):
        tracks = [fit_track(tid, g, method, max_accel) for tid, g in df_ast.groupby('asteroid_id') if tid != -1]
        if not tracks: break
        df_ast, _ = merge_and_refit_tracks(df_ast, tracks, method, max_accel, merge_eps)
    df_ast = cross_tracks(df_ast,method,max_accel,crossing_dist,spatial_lim)

    # -- Final Filter & ID Mapping -- #
    counts = df_ast[df_ast['asteroid_id'] != -1].groupby('asteroid_id').size()
    survivors = sorted(counts[counts >= min_track_events].index.tolist())
    id_map = {old: new for new, old in enumerate(survivors, start=1)}
    df_ast['asteroid_id'] = df_ast['asteroid_id'].apply(lambda x: id_map.get(x, -1))
    final_tracks = [fit_track(tid, g, method, max_accel) for tid, g in df_ast.groupby('asteroid_id') if tid != -1]

    # -- Link Non Asteroid Detections -- #
    df_notast = df[df.classification != 'Asteroid'].copy()
    df_notast = link_detections(df_notast,final_tracks,match_radius,spatial_lim)
    df_notast.loc[df_notast.flux_sign == -1,'asteroid_id'] = -1
    df_notast.loc[df_notast.asteroid_id != -1,'classification'] = 'Asteroid'

    final_df = pd.concat([df_ast,df_notast])
    final_df.loc[final_df.asteroid_id == -1,'asteroid_id'] = '-'

    return final_df