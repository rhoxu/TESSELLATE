import numpy as np
import pandas as pd
from time import time as clock
from copy import deepcopy
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from .tools import RoundToInt


# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- # 
# ------------------------------------------------- Source Detection functions ------------------------------------------------ #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #

def _Spatial_group(result,colname='objid',min_samples=1,distance=0.5,njobs=-1):
    """
    Groups events based on proximity.
    """

    from sklearn.cluster import DBSCAN

    pos = np.array([result.xcentroid,result.ycentroid]).T
    cluster = DBSCAN(eps=distance,min_samples=min_samples,n_jobs=njobs).fit(pos)
    labels = cluster.labels_
    unique_labels = set(labels)
    for label in unique_labels:
        result.loc[label == labels,colname] = label + 1
    result[colname] = result[colname].astype(int)
    return result

def _Spatio_temporal_group(result,colname='objid',min_samples=1,distance=0.5,frame_gap=5,
                           median_distance=None,median_scale_x=1.0,median_scale_y=1.0,njobs=-1):
    """
    Groups events based on proximity in both space and time, 
    then groups their median positions to identify repeating sources.
    This helps prevent moving items from creating large trailed artifact groups.

    Parameters
    ----------
    distance : float
        DBSCAN eps for the first-stage 3D (x, y, scaled_frame) clustering.
    frame_gap : int
        Number of frames that equals one spatial pixel unit for the time scaling.
    median_distance : float, optional
        DBSCAN eps for the second-stage 2D median clustering. If None, uses `distance`.
        Set larger than `distance` to merge spatially nearby event groups (e.g. CCD bleed columns).
    median_scale_x : float
        Scale factor applied to X axis before the second-stage clustering (default 1.0).
        Increase to compress X differences (makes X less important for merging).
    median_scale_y : float
        Scale factor applied to Y axis before the second-stage clustering (default 1.0).
        Decrease to make Y differences more important; increase to allow broader Y merging.
        E.g. set median_scale_y=0.3 to allow groups ~3x further apart in Y to merge.
    """
    if len(result) == 0:
        return result

    from sklearn.cluster import DBSCAN

    if median_distance is None:
        median_distance = distance

    # -- 1. Spatio-Temporal Clustering --
    # Scale frame distance so that a gap of `frame_gap` equals `distance` in DBSCAN 
    # Use max to avoid zero division if frame_gap is set to 0
    frame_scale = distance / max(1e-5, frame_gap)
    
    # Fallback to regular spatial group if no frame info is available
    if 'frame' not in result.columns:
        return _Spatial_group(result, colname=colname, min_samples=min_samples, distance=distance, njobs=njobs)
        
    scaled_frames = result['frame'].values * frame_scale
    
    pos_3d = np.array([result['xcentroid'].values, result['ycentroid'].values, scaled_frames]).T
    
    st_cluster = DBSCAN(eps=distance, min_samples=min_samples, n_jobs=njobs).fit(pos_3d)
    st_labels = st_cluster.labels_.copy() # Make a copy to mutate
    
    # Handle noise (-1) from DBSCAN by giving them distinct unique labels 
    # so they can still participate in median clustering individually
    max_label = st_labels.max()
    noise_mask = st_labels == -1
    if noise_mask.sum() > 0:
        st_labels[noise_mask] = np.arange(max_label + 1, max_label + 1 + noise_mask.sum())
    
    result = result.copy()
    result['st_group'] = st_labels

    # -- 2. Cluster the median positions (with optional anisotropic axis scaling) --
    medians = result.groupby('st_group')[['xcentroid', 'ycentroid']].median()
    
    # Apply axis scaling: divide coords by scale factors so DBSCAN eps acts differently per axis
    pos_2d = np.column_stack([
        medians['xcentroid'].values * median_scale_x,
        medians['ycentroid'].values * median_scale_y
    ])
    spatial_cluster = DBSCAN(eps=median_distance, min_samples=1, n_jobs=njobs).fit(pos_2d)
    
    medians['final_label'] = spatial_cluster.labels_
    
    # Join labels back using st_group as the index mapping to medians index
    result = result.merge(medians[['final_label']], left_on='st_group', right_index=True, how='left')
    
    # Apply 1-indexed labeling identical to the original strategy
    result[colname] = (result['final_label'] + 1).fillna(0).astype(int)
    
    # Strip intermediate fields to maintain identical output structure
    result = result.drop(columns=['st_group', 'final_label'])
    
    return result



def _Classify_and_Merge_Asteroids(result, colname='objid', std_threshold=1.0, corr_threshold=0.8):

    """
    Classifies objects based on spatial variance and time-correlation natively from table properties.
    Merges disjoint asteroid segments belonging to the same moving object.
    Requires 'frame' column.
    """
    if len(result) == 0 or 'frame' not in result.columns:
        return result
        
    result = result.copy()
    if 'candidate_class' not in result.columns:
        result['candidate_class'] = 'Unknown/Static'
    
    asteroid_segments = []
    
    # -- 1. Classification --
    for oid, sub in result.groupby(colname):
        if len(sub) > 5:
            std_x = sub['xcentroid'].std()
            std_y = sub['ycentroid'].std()
            std_t = sub['frame'].std()
            
            corr_x_t = abs(sub['xcentroid'].corr(sub['frame'])) if std_x > 0 and std_t > 0 else 0
            corr_y_t = abs(sub['ycentroid'].corr(sub['frame'])) if std_y > 0 and std_t > 0 else 0
                
            max_std = max(std_x, std_y)
            max_corr = max(corr_x_t, corr_y_t)
            
            # Asteroid Heuristic
            if max_std > std_threshold and max_corr > corr_threshold:
                result.loc[result[colname] == oid, 'candidate_class'] = 'Asteroid'
                
                # Fit linear trajectory for later merging
                vx, x0 = np.polyfit(sub['frame'], sub['xcentroid'], 1)
                vy, y0 = np.polyfit(sub['frame'], sub['ycentroid'], 1)
                
                asteroid_segments.append({
                    'id': oid,
                    'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0,
                    'med_frame': sub['frame'].median(),
                    'med_x': sub['xcentroid'].median(),
                    'med_y': sub['ycentroid'].median()
                })
                
            # Saturated Star Heuristic
            elif max_std > std_threshold and max_corr < 0.6:
                det_per_frame = len(sub) / sub['frame'].nunique() if sub['frame'].nunique() > 0 else 0
                max_val = sub['max_value'].median()
                if det_per_frame > 2 and max_val > 1000:
                    result.loc[result[colname] == oid, 'candidate_class'] = 'Saturated_Star'

    # -- 2. Merging Disjoint Segments --
    if len(asteroid_segments) > 1:
        parent = {seg['id']: seg['id'] for seg in asteroid_segments}
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(len(asteroid_segments)):
            for j in range(i + 1, len(asteroid_segments)):
                si = asteroid_segments[i]
                sj = asteroid_segments[j]
                
                # Predict cross positions
                pred_x_i_at_j = si['vx'] * sj['med_frame'] + si['x0']
                pred_y_i_at_j = si['vy'] * sj['med_frame'] + si['y0']
                dist_i_to_j = np.sqrt((pred_x_i_at_j - sj['med_x'])**2 + (pred_y_i_at_j - sj['med_y'])**2)
                
                pred_x_j_at_i = sj['vx'] * si['med_frame'] + sj['x0']
                pred_y_j_at_i = sj['vy'] * si['med_frame'] + sj['y0']
                dist_j_to_i = np.sqrt((pred_x_j_at_i - si['med_x'])**2 + (pred_y_j_at_i - si['med_y'])**2)
                
                # Bi-directional error within 2 pixels
                if dist_i_to_j < 2.0 and dist_j_to_i < 2.0:
                    union(si['id'], sj['id'])
                    
        # Apply merged IDs
        for seg in asteroid_segments:
            root_id = find(seg['id'])
            if root_id != seg['id']:
                result.loc[result[colname] == seg['id'], colname] = root_id

    return result

def _Star_finding_procedure(data,prf,sig_limit = 2):
    """
    Use StarFinder to find stars with different PSF shapes depending on subpixel shift.
    """

    from photutils.detection import StarFinder
    from astropy.stats import sigma_clipped_stats

    mean, med, std = sigma_clipped_stats(data, sigma=5.0)

    psfCentre = prf.locate(5,5,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfCentre)
    res1 = finder.find_stars(deepcopy(data))

    psfUR = prf.locate(5.25,5.25,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfUR)
    res2 = finder.find_stars(deepcopy(data))

    psfUL = prf.locate(4.75,5.25,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfUL)
    res3 = finder.find_stars(deepcopy(data))

    psfDR = prf.locate(5.25,4.75,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfDR)
    res4 = finder.find_stars(deepcopy(data))

    psfDL = prf.locate(4.75,4.75,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfDL)
    res5 = finder.find_stars(deepcopy(data))

    tables = [res1, res2, res3, res4, res5]
    good_tables = [table.to_pandas() for table in tables if table is not None]
    if len(good_tables)>0:
        total = pd.concat(good_tables)
        total = total[~pd.isna(total['xcentroid'])]
        if len(total) > 0:
            grouped = _Spatial_group(total,distance=2)
            res = grouped.groupby('objid').head(1)
            res = res.reset_index(drop=True)
            res = res.drop(['id','objid'],axis=1)
        else:
            res=None
    else:
        res = None

    return res


def _Find_stars(data,prf,fwhmlim=7,siglim=2.5,bkgstd_lim=50,negative=False):
    """
    Find stars in image.
    """

    from scipy.signal import fftconvolve
    from photutils.aperture import RectangularAperture, RectangularAnnulus, ApertureStats, aperture_photometry
    from astropy.stats import sigma_clip

    # -- Look for negative detections by flipping sign of data -- #
    if negative:
        data = data * -1

    # -- Use StarFinder to find sources with similar psf to TESS PRF -- #
    star = _Star_finding_procedure(data,prf,sig_limit=siglim)

    if star is None:
        return None
    
    # -- Initial cutoff, requiring fwhm to be star like and the source to be at least 3 pixels in from the edge -- #
    ind = (star['fwhm'].values < fwhmlim) & (star['fwhm'].values > 0.8)
    pos_ind = ((star.xcentroid.values >=3) & (star.xcentroid.values < data.shape[1]-3) & 
                (star.ycentroid.values >=3) & (star.ycentroid.values < data.shape[0]-3))
    star = star.iloc[ind & pos_ind]

    # -- Generate Aperture photometry for each source -- #
    x = RoundToInt(star.xcentroid.values); y = RoundToInt(star.ycentroid.values)
    pos = list(zip(x, y))
    if len(pos)<1:
        return None
    aperture = RectangularAperture(pos, 3.0, 3.0)
    annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
    m = sigma_clip(data,masked=True,sigma=5).mask
    mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
    aperstats_sky = ApertureStats(data, annulus_aperture,mask = mask)
    annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=40,h_out=40)
    aperstats_sky_no_mask = ApertureStats(data, annulus_aperture)
    aperstats_source = ApertureStats(data, aperture)
    phot_table = aperture_photometry(data, aperture)
    phot_table = phot_table.to_pandas()

    # -- Readout aperture photometry -- #
    bkg_std = aperstats_sky.std
    bkg_std[bkg_std==0] = aperstats_sky_no_mask.std[bkg_std==0] # assign a value without mask using a larger area of sky
    bkg_std[bkg_std==0] = np.nanmean(bkg_std[bkg_std>0])
    negative_ind = aperstats_source.min >= aperstats_sky.median - 3* bkg_std
    star['sig'] = phot_table['aperture_sum'].values / (aperture.area * bkg_std)
    star['flux'] = phot_table['aperture_sum'].values
    star['mag'] = -2.5*np.log10(phot_table['aperture_sum'].values)
    star['bkgstd'] = aperture.area * bkg_std
    star = star.iloc[negative_ind]
    star = star.loc[(star['sig'] > siglim) & (star['bkgstd'] < bkgstd_lim)]

    if negative:
        star['flux_sign'] = -1
        star['flux'] = star['flux'].values * -1
        star['max_value'] = star['max_value'].values * -1
    else:
        star['flux_sign'] = 1

    return star


def _Frame_detection(data,prf,frameNum):
    """
    Acts on a frame of data. Uses StarFinder to find bright sources, then on each source peform correlation check.
    """
    star = None
    if np.nansum(data) != 0.0:
        p = _Find_stars(data,prf)
        n = _Find_stars(deepcopy(data),prf,negative=True)
        if (p is not None) | (n is not None):
            star = pd.concat([p,n])
            star['frame'] = frameNum
        else:
            star = None
    return star

def _Source_mask(res,mask):
    """
    Add source mask value at each source location.
    """

    xInts = res['xint'].values
    yInts = res['yint'].values

    res['source_mask'] = mask[yInts,xInts]

    return res

def _Count_detections(result):
    """
    Count number of source detections per objid.
    """

    ids = result['objid'].values
    unique = np.unique(ids, return_counts=True)
    unique = list(zip(unique[0],unique[1]))

    array = np.zeros_like(ids)

    for id,count in unique:
        index = (result['objid'] == id).values
        array[index] = count

    result['n_detections'] = array

    return result


def _Do_photometry(star,data,siglim=3,bkgstd_lim=50):
    """
    Do aperture photometry on each source.
    """

    from scipy.signal import fftconvolve
    from photutils.aperture import RectangularAnnulus, CircularAperture, ApertureStats, aperture_photometry
    from astropy.stats import sigma_clip, SigmaClip

    # -- Photometry on source detect stars -- #
    x = RoundToInt(star.xcentroid.values); y = RoundToInt(star.ycentroid.values)
    pos = list(zip(x, y))
    aperture = CircularAperture(pos, 1.5)
    annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
    m = sigma_clip(data,masked=True,sigma=5).mask
    mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
    aperstats_sky = ApertureStats(data, annulus_aperture,mask = mask,sigma_clip=SigmaClip(sigma=3,cenfunc='median'))
    annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=40,h_out=40)
    aperstats_sky_no_mask = ApertureStats(data, annulus_aperture,sigma_clip=SigmaClip(sigma=3,cenfunc='median'))
    aperstats_source = ApertureStats(data, aperture)
    phot_table = aperture_photometry(data, aperture)
    phot_table = phot_table.to_pandas()

    # -- Readout results -- #
    bkg_std = aperstats_sky.std
    bkg_std[bkg_std==0] = aperstats_sky_no_mask.std[bkg_std==0] # assign a value without mask using a larger area of sky
    bkg_std[(~np.isfinite(bkg_std)) | (bkg_std == 0)] = 100
    negative_ind = aperstats_source.min >= aperstats_sky.median - aperture.area * aperstats_sky.std
    star['sig'] = phot_table['aperture_sum'].values / (aperture.area * aperstats_sky.std)
    star['flux'] = phot_table['aperture_sum'].values
    star['mag'] = -2.5*np.log10(phot_table['aperture_sum'].values)
    star['bkgstd'] = aperture.area * bkg_std #* aperstats_sky.std
    
    star = star.loc[(star['sig'] >= siglim) & (star['bkgstd'] <= bkgstd_lim)]

    return star 


def _Source_detect(flux,cpu,siglim=2,bkgstd=50,maxattempts=5):
    """
    Run source detect on each image.
    """
    
    from sourcedetect import SourceDetect
    from joblib import Parallel, delayed 

    attempt = 0
    passed = False
    while (not passed) & (attempt <= maxattempts):
        try:
            res = SourceDetect(flux,run=True,train=False).result
            passed = True
        except:
            attempt += 1

    res = res.drop('psflike',axis=1)

    frames = res['frame'].unique()
    stars = Parallel(n_jobs=cpu)(delayed(_Do_photometry)(res.loc[res['frame'] == frame],flux[frame],siglim,bkgstd) for frame in frames)
    res = pd.concat(stars)

    ind = (res['xint'].values > 3) & (res['xint'].values < flux.shape[2]-3) & (res['yint'].values >3) & (res['yint'].values < flux.shape[1]-3)
    res = res[ind]

    return res

def _Make_dataframe(results,data):
    """
    Collate results into a dataframe.
    """

    # -- Collates results into a dataframe -- #
    frame = None
    for result in results:
        if frame is None:
            frame = result
        else:
            frame = pd.concat([frame,result])
    
    # -- Again checks to make sure all sources are at least 3px from the edge -- # 
    frame['xint'] = deepcopy(RoundToInt(frame['xcentroid'].values))
    frame['yint'] = deepcopy(RoundToInt(frame['ycentroid'].values))
    ind = ((frame['xint'].values >3) & (frame['xint'].values < data.shape[1]-3) & 
           (frame['yint'].values >3) & (frame['yint'].values < data.shape[0]-3))
    frame = frame[ind]

    return frame

def _Collate_frame(results):
    """
    Collate result dataframe to be more managable.
    """
    

    frame_sf = results[results['method'] == 'SF'].copy()
    frame_sd = results[results['method'] == 'SD'].copy()

    frame_sf = frame_sf.rename(columns={
        'xcentroid': 'xcentroid_SF',
        'ycentroid': 'ycentroid_SF'
    })
    frame_sd = frame_sd.rename(columns={
        'xcentroid': 'xcentroid_SD',
        'ycentroid': 'ycentroid_SD'
    })

    frame_merged = pd.merge(
        frame_sf,
        frame_sd,
        on=['objid', 'frame'],        
        how='outer',
        suffixes=('_SFtemp', '_SDtemp'),
        sort=False
    )

    for col in frame_sf.columns:
        if col in ['objid', 'xcentroid_SF', 'ycentroid_SF']:
            continue
        sf_col = f"{col}_SFtemp"
        sd_col = f"{col}_SDtemp"
        if sf_col in frame_merged.columns and sd_col in frame_merged.columns:
            # Prefer SD values when available
            frame_merged[col] = frame_merged[sd_col].combine_first(frame_merged[sf_col])
            frame_merged.drop(columns=[sf_col, sd_col], inplace=True)

    frame_merged = frame_merged.drop(columns=['method_SFtemp', 'method_SDtemp','method'], errors='ignore')

    frame_merged['xcentroid'] = frame_merged['xcentroid_SD'].combine_first(frame_merged['xcentroid_SF'])
    frame_merged['ycentroid'] = frame_merged['ycentroid_SD'].combine_first(frame_merged['ycentroid_SF'])

    frame_merged['xint'] = RoundToInt(frame_merged['xcentroid'])
    frame_merged['yint'] = RoundToInt(frame_merged['ycentroid'])

    return frame_merged

def _Brightest_Px(flux,frame):
    """
    Find the brightest pixel.
    """

    brightest_xint = []
    brightest_yint = []
    for i in range(len(frame)):
        x = frame.loc[i, 'xint']
        y = frame.loc[i, 'yint']
        im = frame.loc[i, 'frame']
        image = flux[im,y-1:y+2,x-1:x+2]
        iy, ix = np.unravel_index(np.nanargmax(image), image.shape)
        brightesty = y + (iy - 1)  # shift from 3x3 center
        brightestx = x + (ix - 1)
        brightest_xint.append(brightestx)
        brightest_yint.append(brightesty)

    frame['xint_brightest'] = brightest_xint
    frame['yint_brightest'] = brightest_yint

    return frame




# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- # 
# -------------------------------------------------- Event finding functions -------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #


def _Get_temporal_events(df, max_gap=2, frame_col='frame', id_col='eventid',startingID=1):
    """
    Labels temporally clustered events in a DataFrame by assigning an event ID.

    Parameters:
        df: Pandas DataFrame with a time/frame column
        max_gap: maximum allowed gap (in frames) between temporally connected sources
        frame_col: name of the column containing frame/time info
        id_col: name of the new event ID column to add

    Returns:
        Modified DataFrame with a new column for event ID
    """
    df = df.copy()
    df = df.sort_values(frame_col).reset_index(drop=False)  # preserve original index

    event_ids = []
    current_event = startingID
    event_ids.append(current_event)

    for i in range(1, len(df)):
        frame_diff = df.loc[i, frame_col] - df.loc[i - 1, frame_col]
        if frame_diff <= max_gap:
            event_ids.append(current_event)
        else:
            current_event += 1
            event_ids.append(current_event)

    df[id_col] = event_ids
    df = df.sort_values('index').set_index('index')  # return to original order
    return df


def _Check_LC_significance(time,flux,start,end,pos,flux_sign,buffer = 0.5,base_range=1):

    from .tools import Generate_LC

    time_per_frame = time[1] - time[0]
    buffer = int(buffer/time_per_frame)
    base_range = int(base_range/time_per_frame)
    ap = np.zeros_like(flux[0])
    y = pos[1] 
    x = pos[0] 
    _,lc = Generate_LC(time,flux,x,y,radius=1.5)
    # lc = np.nansum(flux[:,y-1:y+2,x-1:x+2],axis=(1,2))
    fs = start - buffer
    fe = end + buffer
    if fs < 0:
        fs = 0
    if fe > len(lc):
        fe = len(lc) - 1 
    baseline = lc
    bs = fs - base_range
    be = fe + base_range
    if bs < 0:
        bs = 0
    if be > len(lc):
        be = len(lc) - 1 
    frames = np.arange(0,len(lc))
    ind = ((frames > bs) & (frames < fs)) | ((frames < be) & (frames > fe))
    #mean,med, std = sigma_clipped_stats(lc[ind])
    med = np.nanmedian(lc[ind])
    std = np.nanstd(lc[ind])

    lcevent = lc[start:end+1]
    lc_sig = (lcevent - med) / std
    
    max_flux = np.nanmax(lcevent)
    max_frame = np.argmax(lcevent)+start

    if flux_sign >= 0:
        sig_max = np.nanmax(lc_sig)
        sig_med = np.nanmean(lc_sig)
        
    else:
        sig_max = abs(np.nanmin(lc_sig))
        sig_med = abs(np.nanmean(lc_sig))
    
    lc_sig = (lc - med) / std
    return sig_max, sig_med, lc_sig * flux_sign, max_flux, max_frame


def _Lightcurve_event_checker(lc_sig,triggers,siglim=3,maxsep=5):

    triggers = list(triggers)
    start = np.nanmin(triggers)
    end = np.nanmax(triggers)

    sig_ind = np.where(lc_sig>= siglim)[0]

    new_frames = (frame for frame in sig_ind if start <= frame <= end and frame not in triggers)
    triggers.extend(new_frames)

    doneBack = False
    frame_ind = start-1
    count = 0
    while not doneBack:
        if frame_ind in sig_ind:
            triggers.append(frame_ind)
        else:
            count += 1
            if count == maxsep:
                doneBack = True
        frame_ind -= 1


    doneFront = False
    frame_ind = end+1
    count = 0
    while not doneFront:
        if frame_ind in sig_ind:
            triggers.append(frame_ind)
        else:
            count += 1
            if count == maxsep:
                doneFront = True
        frame_ind += 1

    new_start = np.nanmin(triggers)
    new_end = np.nanmax(triggers)
    n_detections = len(triggers)

    return new_start,new_end,n_detections,sorted(triggers)


def _Fit_psf(flux, event, prf, frames, uncertainty_func, big_size=15, small_size=5):
    """
    Generate an cutout around an event and fit PSF. 
    Chooses the frame based on the highest SNR between stack through event and individual frames.
    """

    from .localisation import PSF_Fitter
    from astropy.stats import sigma_clipped_stats

    x0 = RoundToInt(event['xcentroid_det'])
    y0 = RoundToInt(event['ycentroid_det'])
    sign = event['flux_sign']

    half_big = big_size // 2
    half_small = small_size // 2

    h, w = flux.shape[1], flux.shape[2]

    # -- Generate a stacked 3x3 aperture to identify and lock onto brightest pixel -- #
    stacked_flux_3x3 = np.zeros((3, 3), dtype=np.float32) 
    for i in frames: 
        y1, y2 = y0 - 1, y0 + 2 
        x1, x2 = x0 - 1, x0 + 2 
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w: 
            continue 
        cut = flux[i, y1:y2, x1:x2] * sign 
        stacked_flux_3x3 += cut 
        
    iy, ix = np.unravel_index(np.nanargmax(stacked_flux_3x3), stacked_flux_3x3.shape)

    brightest_y = y0 + (iy - 1)
    brightest_x = x0 + (ix - 1)

    event['xint_brightest'] = brightest_x
    event['yint_brightest'] = brightest_y


    # --- Produce big_size cutouts with NaN padding and add to stack --- #
    stacked_big = np.zeros((big_size, big_size), dtype=np.float32)
    cuts = []
    snrs = []

    y1 = brightest_y - half_big        # Desired bounds in full image
    y2 = brightest_y + half_big + 1
    x1 = brightest_x - half_big
    x2 = brightest_x + half_big + 1

    yy1, yy2 = max(0, y1), min(h, y2)   # Clip to image bounds
    xx1, xx2 = max(0, x1), min(w, x2)

    for i in frames:
        cut = np.full((big_size, big_size), np.nan, dtype=np.float32)   # Create NaN-padded cut

        cy1 = yy1 - y1
        cy2 = cy1 + (yy2 - yy1)
        cx1 = xx1 - x1
        cx2 = cx1 + (xx2 - xx1)

        cut[cy1:cy2, cx1:cx2] = flux[i, yy1:yy2, xx1:xx2] * sign

        valid = cut[~np.isnan(cut)]     # Compute noise only on valid pixels
        if valid.size == 0:
            continue
        _, _, noise = sigma_clipped_stats(valid, sigma=3)

        core = cut[                         
            half_big-half_small:half_big+half_small+1,      # 3x3 aperture around core
            half_big-half_small:half_big+half_small+1,
        ]

        flux_sum = np.nansum(core)      # Estimate the snr from core
        snr = flux_sum / (9 * noise)

        cuts.append(cut)
        snrs.append(snr)
        stacked_big += np.nan_to_num(cut, nan=0.0)

    if len(cuts) == 0:
        raise ValueError("All PSF cutouts were out of bounds.")


    # --- Stacked SNR (ignore NaNs) --- #
    valid = stacked_big[~np.isnan(stacked_big)]
    _, _, noise = sigma_clipped_stats(valid, sigma=3)

    stacked_core = stacked_big[
        half_big-half_small:half_big+half_small+1,
        half_big-half_small:half_big+half_small+1,
    ]

    stacked_flux_sum = np.nansum(stacked_core)
    stacked_snr = stacked_flux_sum / (9 * noise)

    # --- Choose best image or stacked through event --- #
    if np.max(snrs) > stacked_snr:
        idx = int(np.argmax(snrs))
        centred_flux = cuts[idx][
            half_big-half_small:half_big+half_small+1,
            half_big-half_small:half_big+half_small+1,
        ]
        snr = snrs[idx]
        stacked_psf_fit = 0
    else:
        centred_flux = stacked_core
        snr = stacked_snr
        stacked_psf_fit = 1

    # --- PSF fit --- #
    unc = uncertainty_func(snr)

    fitter = PSF_Fitter(small_size, prf)
    fitter.fit_psf(centred_flux, limx=0.5, limy=0.5)

    event['xcentroid_psf'] = fitter.source_x + brightest_x
    event['ycentroid_psf'] = fitter.source_y + brightest_y
    event['centroid_err_psf'] = unc
    event['snr_psf'] = snr

    r = np.corrcoef(centred_flux.flatten(), fitter.psf.flatten())[0, 1]
    event['psf_like'] = r

    norm_flux = centred_flux / np.nansum(centred_flux)
    event['psf_diff'] = np.nansum(np.abs(norm_flux - fitter.psf))
    event['psf_stacked'] = stacked_psf_fit

    return event



def _Isolate_events(objid,time,flux,sources,sector,cam,ccd,cut,prf,
                    snr_to_localisation_func,nan_frames,
                    frame_buffer=5,buffer=1,base_range=1):
    """
    Groups sources for given objid into temporally separated events.
    """

    from .tools import pandas_weighted_avg

    # -- Select all sources grouped to this objid -- #
    source = sources[sources['objid']==objid]
    
    all_labelled_sources = []

    # -- For this objid, separate positive and negative detections -- #
    startingID=1
    for sign in source['flux_sign'].unique():
        signed_sources = source[source['flux_sign']==sign]

        # weighted_signedsources = pandas_weighted_avg(signed_sources)

        # # -- Look for any regular variability -- #
        # peak_freq, peak_power = _Fit_period(time,flux,weighted_signedsources.iloc[0])

        # # -- Run RFC Classification -- #
        # cf_classification, cf_prob = _Check_classifind(time,flux,weighted_signedsources.iloc[0])

        labelled_sources = _Get_temporal_events(signed_sources,max_gap=frame_buffer,startingID=startingID)
        all_labelled_sources.append(labelled_sources)
        startingID = np.nanmax(labelled_sources['eventid'])+1
    
    all_labelled_sources = pd.concat(all_labelled_sources, ignore_index=True)
    
    # -- Iterate through eventids -- #
    dfs = []
    for eventID in all_labelled_sources['eventid'].unique():
        event = {}
        eventsources = deepcopy(all_labelled_sources[all_labelled_sources['eventid']==eventID])
        weighted_eventsources = pandas_weighted_avg(eventsources)

        xint = RoundToInt(weighted_eventsources.iloc[0]['xint_brightest'])
        yint = RoundToInt(weighted_eventsources.iloc[0]['yint_brightest'])

        # -- Calculate significance of detection above background and local light curve -- #
        _, _, sig_lc, _, _ = _Check_LC_significance(time,flux,eventsources['frame'].min(),eventsources['frame'].max(),
                                                                [xint,yint],sign,buffer=buffer,base_range=base_range)
        

        sig_lc[nan_frames] = np.nan
        
        # -- Extend the event duration based on time spent above significance threshold -- #
        frame_start,frame_end,n_detections,frames = _Lightcurve_event_checker(sig_lc,eventsources['frame'].values,siglim=3,maxsep=5)

        # -- Initialise event information -- #
        event['frame_bin'] = int(eventsources.iloc[0].frame_bin)
        event['objid'] = int(objid)
        event['eventid'] = int(eventID)
        event['sector'] = int(sector)
        event['camera'] = int(cam)
        event['ccd'] = int(ccd)
        event['cut'] = int(cut)
        event['frame_start'] = int(frame_start)
        event['frame_end'] = int(frame_end)
        event['frame_duration'] = int(event['frame_end']-event['frame_start']+1)
        event['flux_sign'] = int(eventsources.iloc[0]['flux_sign'])
        event['n_detections'] = int(n_detections)
        event['bkg_level'] = weighted_eventsources.iloc[0]['bkg_level']
        event['bkg_std'] = weighted_eventsources.iloc[0]['bkgstd']

        event['xcentroid_det'] = weighted_eventsources.iloc[0]['xcentroid']
        event['ycentroid_det'] = weighted_eventsources.iloc[0]['ycentroid']

        # -- Fit PSF -- #
        event = _Fit_psf(flux,event,prf,frames,snr_to_localisation_func)
        
        # -- If event is quite PSF-like, centroid likely good -- #
        if event['psf_like']>0.6:
            event['xcentroid'] = event['xcentroid_psf']
            event['ycentroid'] = event['ycentroid_psf']
            event['centroid_err'] = event['centroid_err_psf']
        else:
            event['xcentroid'] = event['xcentroid_det']
            event['ycentroid'] = event['ycentroid_det']
            event['centroid_err'] = 0.5

        event['xint'] = RoundToInt(event['xcentroid'])
        event['yint'] = RoundToInt(event['ycentroid'])

        # -- Calculate and save LC statistics -- #
        sig_max, sig_med, _, max_flux, max_frame = _Check_LC_significance(time,flux,
                                                                event['frame_start'],event['frame_end'],
                                                                [event['xint'],event['yint']],
                                                                sign,buffer=buffer,base_range=base_range)

        event['frame_max'] = int(max_frame)
        event['flux_max'] = int(max_flux)
        event['image_sig_max'] = np.nanmax(eventsources['sig'].values)
        event['lc_sig_max'] = sig_max
        event['lc_sig_med'] = sig_med

        # -- Miscellaneous info -- #
        # event['peak_freq'] = peak_freq[0]
        # event['peak_power'] = peak_power[0]
        event['source_mask'] = eventsources.iloc[0]['source_mask']
        # event['cf_class'] = cf_classification
        # event['cf_prob'] = cf_prob

        df = pd.DataFrame([event])
        dfs.append(df)

    events_df = pd.concat(dfs, ignore_index=True)

    # -- Add to each dataframe row the number of events in the total object -- #
    events_df['total_events'] = int(len(events_df))
    
    return events_df 


    
def _Straight_line_asteroid_checker(time,flux,events):
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


def _Calculate_xcom_motion(flux,candidates):

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

def _Gaussian_score(time,flux,candidates):

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

def _Threshold_asteroid_checker(time,flux,events,com_motion_thresholds=[1, 0.75, 0.5], gaussian_score_thresholds=[0, 0.7, 0.9]):
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
    candidates = _Calculate_xcom_motion(flux,candidates)
    candidates = _Gaussian_score(time,flux,candidates)

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


def _Recheck_asteroid_lcs(time,flux,events):
    """
    Asteroids move their centroids a lot in PSF fitting, so often their light curve is now off. This correts for that.
    """

    asteroids = events[events['classification']=='Asteroid']
    if len (asteroids) > 0:
        for _, ast in asteroids.iterrows():
            objid = int(ast['objid'])
            eventid = int(ast['eventid'])
            frame_bin = int(ast['frame_bin'])
            xint = int(ast['xint'])
            yint = int(ast['yint'])
            frame_start = int(ast['frame_start'])
            frame_end = int(ast['frame_end'])

            
            _, _, lc_sig, _, _ = _Check_LC_significance(
                time, flux, frame_start, frame_end, [xint, yint], 1, 0.5, 1
            )
            

            start, end, _, _ = _Lightcurve_event_checker(
                lc_sig, np.arange(frame_start, frame_end+1)
            )

            n_detections = np.nansum([frame_start - start, end - frame_end])

            # Update self.events
            event_mask = (events['objid'] == objid) & (events['eventid'] == eventid) &  (events['frame_bin'] == frame_bin)
            events.loc[event_mask, 'frame_start'] = start
            events.loc[event_mask, 'frame_end'] = end
            events.loc[event_mask, 'mjd_start'] = time[start]
            events.loc[event_mask, 'mjd_end'] = time[end]
            events.loc[event_mask, 'frame_duration'] = end - start
            events.loc[event_mask, 'mjd_duration'] = time[end] - time[start]
            events.loc[event_mask, 'n_detections'] += n_detections

    return events


class Detector():

    def __init__(self,sector,cam,ccd,data_path='/fred/oz335/TESSdata',n=8,
                 match_variables=True,mode='both',part=None,cpu=multiprocessing.cpu_count()):
        
        """
        Tessellate Detection Class.
        """

        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.data_path = data_path
        self.n = n
        self.match_variables = match_variables
        self.mode = mode

        self.cpu = cpu

        self.flux = None
        self.ref = None
        self.time = None
        self.mask = None
        self.sources = None  #raw detection results
        self.events = None   #temporally located with same object id
        self.objects = None   #temporally and spatially combined
        self.cut = None
        self.bkg = None

        if part is None:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}'
        elif part == 1:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Part1'
        elif part == 2:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Part2'
        else:
            e = 'Invalid Part Parameter!'
            raise AttributeError(e)


        
    # ------------------------------ Gathering data and cached results ------------------------------ #

    def gather_results(self,cut):
        """
        Gather pre-existing sources / events csvs.
        """

        import os

        self.objects = None
        path = f'{self.path}/Cut{cut}of{self.n**2}'

        if os.path.exists(f'{path}/detected_sources.csv'):
            self.sources = pd.read_csv(f'{path}/detected_sources.csv')    # raw detection results
        else:
            print('No detected sources file found')
            self.sources = None

        if os.path.exists(f'{path}/detected_events.csv'):
            self.events = pd.read_csv(f'{path}/detected_events.csv')    # raw detection results
        else:
            print('No detected events file found')
            self.events = None
            

    def gather_data(self,cut,flux=True,time=True,bkg=False,mask=False,ref=False,verbose=True):
        """
        Gather reduced data.
        """

        from .localisation import CutWCS
        
        if verbose:
            ts = clock()
            print(f'Loading Cut {cut} Data...',end='\r')

        base = f'{self.path}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}'

        if flux:
            self.flux = np.load(base + '_ReducedFlux.npy')
            self.cut = cut

        if time:
            self.time = np.load(base + '_Times.npy')

        if ref:
            self.ref = np.load(base + '_Ref.npy')

        if bkg:
            self.bkg = np.load(base + '_Background.npy')

        if mask:
            self.mask = np.load(base + '_Mask.npy')

        
        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=cut,n=self.n)

        if verbose:
            print(f'Loading Cut {cut} Data -- done! ({clock()-ts:.0f}s)')

    


    # ------------------------------ Source finding functions ------------------------------ #

    def _find_sources_in_images(self,flux,column,row,inputNums=None,
                                isolate_single_detections=True,datadir='/fred/oz335/_local_TESS_PRFs'):
        """
        Detect sources in flux and collate information.
        """

        from joblib import Parallel, delayed 
        from tqdm import tqdm
        from PRF import TESS_PRF

        if inputNums is not None:
            flux = flux[inputNums]
            inputNum = inputNums[-1]
        else:
            inputNum = 0
                
        if self.sector < 4:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{datadir}/Sectors1_2_3')
        else:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{datadir}/Sectors4+')

        # -- Run source detection on each frame -- #
        length = np.linspace(0,flux.shape[0]-1,flux.shape[0]).astype(int)
        if self.mode == 'starfind':
            sources = Parallel(n_jobs=self.cpu)(delayed(_Frame_detection)(flux[i],prf,inputNum+i) for i in tqdm(length))
            
            sources = _Make_dataframe(sources,flux[0])
            sources['method'] = 'starfind'

        elif self.mode == 'sourcedetect':
            sources = _Source_detect(flux,self.cpu)
            sources['method'] = 'sourcedetect'
            sources = sources[~pd.isna(sources['xcentroid'])]

        elif self.mode == 'both':
            results = Parallel(n_jobs=self.cpu)( delayed(_Frame_detection)(flux[i],prf,inputNum+i) for i in tqdm(length))
            
            star = _Make_dataframe(results,flux[0])
            star['method'] = 'starfind'

            machine = _Source_detect(flux,self.cpu)
            machine = machine[~pd.isna(machine['xcentroid'])]
            machine['method'] = 'sourcedetect'

            sources = pd.concat([star.assign(method='SF'), machine.assign(method='SD')], ignore_index=True)

        # -- Group based on distance -- #
        sources = _Spatial_group(sources,distance=0.5,min_samples=1)        

        # -- Prefer SourceDetect results -- #
        sources = _Collate_frame(sources)                  

        sources = sources.drop('objid',axis=1)

        # -- Group based on distance -- #
        sources = _Spatial_group(sources,distance=0.5,min_samples=2)   # prefer SourceDetect results

        # -- Separate source detections with no companion -- #
        single_isolated_detections = None
        if isolate_single_detections:
            single_isolated_detections = sources[sources['objid']==0]
            sources = sources[sources['objid']>0].reset_index(drop=True)

        # --  Find brightest pixels around each source -- #
        sources = _Brightest_Px(flux,sources)               

        # -- Add in TessReduce source mask value -- #
        sources = _Source_mask(sources,self.mask)                

        # -- Count num detections for each objid -- #
        sources = _Count_detections(sources)               

        return sources,single_isolated_detections


    def _wcs_time_info(self,result):
        """
        Physical units for sources.
        """
        
        result['ra'],result['dec'] = self.wcs.all_pix2world(result['xcentroid'],result['ycentroid'],0)
        result['mjd'] = self.time[result['frame']]
    
        return result

    def _run_find_sources(self,frame_bin):
        """
        Run the source finding on this specific frame_binning.
        """

        from .dataprocessor import DataProcessor
        # from .catalog_queries import match_result_to_cat #,find_variables, gaia_stars,
        from .tools import pandas_weighted_avg,Frame_Bin
        
        save_folder = f'{self.path}/Cut{self.cut}of{self.n**2}'

        # -- Access information about the cut with respect to the original ccd -- #        
        processor = DataProcessor(sector=self.sector,data_path=self.data_path,verbose=2)
        cut_corners, cut_centre_px, _, _ = processor.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        column = cut_centre_px[self.cut-1][0]
        row = cut_centre_px[self.cut-1][1]

        # -- Rebin time and flux by given factor -- #
        if frame_bin > 1:
            time,flux = Frame_Bin(self.time,self.flux,frame_bin)
            isolate_single_detections = False
        else:
            time = self.time; flux = self.flux
            isolate_single_detections = True

        # -- Run the detection algorithm, generates dataframe -- #
        results,single_isolated_detections = self._find_sources_in_images(flux,column=column,row=row,datadir=self.prf_path,
                                                                 isolate_single_detections=isolate_single_detections)
        
        # -- Save out single detections which are isolated in space in time, probably noise, maybe cool -- #
        if isolate_single_detections:
            single_isolated_detections.to_csv(f'{save_folder}/single_isolated_detections.csv',index=False)
        
        # -- Add wcs, time, ccd info to the results dataframe -- #
        results = self._wcs_time_info(results)
        results['xccd'] = deepcopy(results['xcentroid'] + cut_corners[self.cut-1][0])
        results['yccd'] = deepcopy(results['ycentroid'] + cut_corners[self.cut-1][1])
        
        # -- For each source, finds the average position based on the weighted average of the flux -- #
        av_var = pandas_weighted_avg(results[['objid','sig','xcentroid','ycentroid','ra','dec','xccd','yccd']])
        av_var = av_var.rename(columns={'xcentroid':'x_source',
                                        'ycentroid':'y_source',
                                        'e_xcentroid':'e_x_source',
                                        'e_ycentroid':'e_y_source',
                                        'ra':'ra_source',
                                        'dec':'dec_source',
                                        'e_ra':'e_ra_source',
                                        'e_dec':'e_dec_source',
                                        'xccd':'xccd_source',
                                        'yccd':'yccd_source',
                                        'e_xccd':'e_xccd_source',
                                        'e_yccd':'e_yccd_source'})
        av_var = av_var.drop(['sig','e_sig'],axis=1)
        results = results.merge(av_var, on='objid', how='left')

        # -- Calculates the background level for each source -- #
        results['bkg_level'] = 0
        if self.bkg is not None:
            f = results['frame'].values
            x = results['xint'].values; y = results['yint'].values
            b = []
            for i in range(3):
                i-=1
                for j in range(3):
                    j-=1
                    b += [self.bkg[f,y-i,x+j]]
            b = np.array(b)
            b = np.nansum(b,axis=0)
            results['bkg_level'] = b

        results['frame_bin'] = frame_bin

        return results
    
    def _parse_time_bins(self, bins):
        import re

        conversions = {'sec':1/86400,'min': 1/1440, 'hr': 1/24, 'day': 1}
        max_resolution = np.nanmedian(np.diff(self.time))
        
        frame_bins = []
        for bin in bins:
            v, u = re.match(r'([\d.]+)(\w+)', bin).groups()
            resolution = float(v) * conversions[u]
            frame_bins.append(RoundToInt(resolution/max_resolution))

        return frame_bins

    def find_sources(self,time_bins):
        """
        Find sources.
        """

        # -- Turn time_bins into frame_bins -- #
        frame_bins = self._parse_time_bins(time_bins)

        # -- Iteratate over frame_bins and run source detection -- #
        sources = pd.DataFrame()
        for frame_bin in frame_bins:
            ts = clock()
            sources = pd.concat([sources,self._run_find_sources(frame_bin)]) 
            print(f'    Run with time bin = {frame_bin} ({clock()-ts:.0f}s)')
            
        # -- Reset objid so each objid,frame_bin pair is unique -- #
        matched = sources[sources['objid'] != 0]
        unmatched = sources[sources['objid'] == 0]
        matched['objid'] = pd.factorize(matched['objid'].astype(str) + '_' + matched['frame_bin'].astype(str))[0]+1
        unmatched['objid'] = np.arange(matched['objid'].max() + 1, matched['objid'].max() + 1 + len(unmatched))

        sources = pd.concat([matched, unmatched])

        self.sources = sources

        self.sources.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_sources.csv',index=False)




    # ------------------------------ Event finding functions ------------------------------ #

    def _get_all_independent_events(self,frame_buffer=10,buffer=0.5,base_range=1):
        """
        Isolate sources into individual temporal events.
        """

        from joblib import Parallel, delayed 
        from tqdm import tqdm
        from .dataprocessor import DataProcessor
        from .localisation import get_snr_to_localisation_func, get_wcs_uncertainty
        from .tools import Frame_Bin
        from PRF import TESS_PRF
        
    
        # -- Generate PRF -- #
        dp = DataProcessor(self.sector,data_path=self.data_path)
        cutCornerPx, cutCentrePx, _, _ = dp.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        column = cutCentrePx[self.cut-1][0]
        row = cutCentrePx[self.cut-1][1]
        if self.sector < 4:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{self.prf_path}/Sectors1_2_3')
        else:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{self.prf_path}/Sectors4+')
        
        # -- Retrieve cut localisation quality -- #
        snr_to_localisation = get_snr_to_localisation_func(self.data_path,self.sector,self.cam,self.ccd,self.cut,self.n)

        # -- Iterate over frame bins -- #
        frame_bins = np.unique(self.sources.frame_bin)
        events = pd.DataFrame()
        for frame_bin in frame_bins:
            sources = self.sources[self.sources.frame_bin == frame_bin]
            time,flux = Frame_Bin(self.time,self.flux,frame_bin)

            has_data = np.any(np.isfinite(flux), axis=(1, 2))      # Identify nan_frames 
            nan_frames = np.where(~has_data)[0]

            # -- Iterate over objids to separate into discrete events -- #
            objids = np.unique(sources['objid'].values).astype(int)
            if self.cpu > 1:
                length = np.arange(0,len(objids)).astype(int)
                bin_events = Parallel(n_jobs=self.cpu)(delayed(_Isolate_events)(objids[i],time,flux,sources,
                                                                    self.sector,self.cam,self.ccd,self.cut,prf,snr_to_localisation,nan_frames,
                                                                    frame_buffer,buffer,base_range) for i in tqdm(length))
            else:            
                bin_events = []
                for objid in objids:
                    e = _Isolate_events(objid,time,flux,sources,self.sector,self.cam,
                                        self.ccd,self.cut,prf,snr_to_localisation,nan_frames,frame_buffer=frame_buffer,
                                        buffer=buffer,base_range=base_range)
                    bin_events += [e]
            
            events = pd.concat([events,pd.concat(bin_events)],ignore_index=True)

        # -- Provide CCD-relative location -- #
        events['xccd'] = RoundToInt(events['xint'] + cutCornerPx[self.cut-1][0])
        events['yccd'] = RoundToInt(events['yint'] + cutCornerPx[self.cut-1][1])

        # -- Pull the uncertainty on WCS and combine with PSF fit centroid error -- #
        wcs_unc = get_wcs_uncertainty(self.data_path,self.sector,self.cam,self.ccd,self.cut,self.n)
        if np.isnan(wcs_unc).any():
            events['xcentroid_err'] = 0.5
            events['ycentroid_err'] = 0.5
        else:
            events['xcentroid_err'] = np.sqrt(events['centroid_err']**2 + wcs_unc[0]**2)
            events['ycentroid_err'] = np.sqrt(events['centroid_err']**2 + wcs_unc[1]**2)

        # -- Remove all events with single frame durations -- #
        fake_events = events[(events.frame_duration==1)&(events.frame_bin==1)].copy()
        fake_events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/single_frame_events.csv')

        real_events = events[(events.frame_duration>1)|(events.frame_bin>1)].copy()
        real_events['total_events'] = real_events.groupby('objid')['objid'].transform('size')
        real_events['eventid'] = real_events.groupby('objid').cumcount() + 1

        self.events = real_events 

    def _events_physical_units(self):
        """
        Give physical values to event.
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        events = deepcopy(self.events)

        events['ra'],events['dec'] = self.wcs.all_pix2world(events['xcentroid'],events['ycentroid'],0)

        delta = 0.1
        # Derivatives w.r.t x
        ra_px, dec_px = self.wcs.all_pix2world(events['xcentroid'] + delta, events['ycentroid'], 0)
        dra_dx = (ra_px - events['ra']) / delta
        ddec_dx = (dec_px - events['dec']) / delta
        
        # Derivatives w.r.t y
        ra_py, dec_py = self.wcs.all_pix2world(events['xcentroid'],  events['ycentroid'] + delta, 0)
        dra_dy = (ra_py - events['ra']) / delta
        ddec_dy = (dec_py - events['dec']) / delta
        
        events['ra_err'] = np.sqrt((dra_dx * events['xcentroid_err'])**2 + (dra_dy * events['ycentroid_err'])**2)
        events['dec_err'] = np.sqrt((ddec_dx * events['xcentroid_err'])**2 + (ddec_dy * events['ycentroid_err'])**2)
        
        events['dec_err'] = np.abs(events['ra_err'] * np.cos(np.radians(events['dec'])))   # account for cos(dec) factor in RA

        coords = SkyCoord(ra=events['ra'].values*u.degree,dec=events['dec'].values*u.degree)
        events['gal_l'] = coords.galactic.l.value
        events['gal_b'] = coords.galactic.b.value

        events['mjd_start'] = self.time[events['frame_start']]
        events['mjd_end'] = self.time[events['frame_end']]
        events['mjd_duration'] = events['mjd_end'] - events['mjd_start']
        events['mjd_max'] = self.time[events['frame_max']]

        events['mag_min'] = -2.5*np.log10(events['flux_max'])

        self.events = events

    def _flag_asteroids(self):
        """
        Flag asteroids in events.
        """

        from .tools import Frame_Bin

        events = deepcopy(self.events)

        # -- Correct light curves of asteroids whose locations change -- #
        frame_bins = np.unique(events.frame_bin)
        events_list = []
        for frame_bin in frame_bins:
            evs = events[events.frame_bin == frame_bin]
            time,flux = Frame_Bin(self.time,self.flux,frame_bin)
            evs = _Threshold_asteroid_checker(time,flux,evs)        # Generally best, checks for centre of mass movement and light curve Gaussianity 
            evs = _Straight_line_asteroid_checker(time,flux,evs)    # Picks up events with weirdly long event boundaries  
            evs = _Recheck_asteroid_lcs(time,flux,evs)              # Correct light curves of asteroids whose locations change -- #
            events_list.append(evs)
        
        
        # # -- Generally best, checks for centre of mass movement and light curve Gaussianity -- #
        # events = _Threshold_asteroid_checker(self.time,self.flux,events)

        # # -- Picks up events with weirdly long event boundaries -- # 
        # events = _Straight_line_asteroid_checker(self.time,self.flux,events)

        # # -- Correct light curves of asteroids whose locations change -- #
        # frame_bins = np.unique(events.frame_bin)
        # events_list = []
        # for frame_bin in frame_bins:
        #     evs = events[events.frame_bin == frame_bin]
        #     time,flux = Frame_Bin(self.time,self.flux,frame_bin)
        #     evs = _Recheck_asteroid_lcs(time,flux,evs)
        #     events_list.append(evs)

        self.events = pd.concat(events_list)

    def _catalogue_crossmatch(self,sigma=3):
        """
        Crossmatch events with stars / variables.
        """
        
        events = deepcopy(self.events)

        # -- Cross matches location to Gaia -- #
        gaia = pd.read_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/local_gaia_cat.csv')
        events['GaiaID'] = '-'
        for i,event in events.iterrows():
            if event.classification != 'Asteroid':
                inside = gaia[(abs(gaia.ra-event.ra) < sigma*event.ra_err)&
                            (abs(gaia.dec-event.dec) < sigma*event.dec_err)]
                if len(inside) > 0:
                    events.loc[i,'GaiaID'] = inside[inside.mag==inside.mag.min()].iloc[0].Source
        
        # -- Cross matches location to variable catalog -- #
        variables = pd.read_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/variable_catalog.csv')
        for i,event in events.iterrows():
            if event.classification != 'Asteroid':
                inside = variables[(abs(variables.ra-event.ra) < sigma*event.ra_err)&
                            (abs(variables.dec-event.dec) < sigma*event.dec_err)]
                if len(inside) > 0:
                    events.loc[i,'classification'] = inside.iloc[0].Type

        self.events = events

    def _crossmatch_framebin(self):
        """
        Crossmatch between time_bins.
        """

        events = deepcopy(self.events)
        events['crossbin_ids'] = [[] for _ in range(len(events))]

        # -- Group spatially -- #
        events = _Spatial_group(events, colname='crossbin_group', distance=1, min_samples=1)
        events = _Spatial_group(events, colname='asteroid_crossbin_group', distance=4, min_samples=1)

        # -- Iterate through spatial groups -- # 
        for group in np.unique(events.crossbin_group):
            group_df = events[events.crossbin_group == group].sort_values('frame_bin')

            for i, row in group_df.iterrows():
                # If this event has no ID yet, assign its own
                if not events.at[i, 'crossbin_ids']:
                    events.at[i, 'crossbin_ids'] = [i]

                current_ids = events.at[i, 'crossbin_ids']

                # For asteroids, use the wider spatial group to find coarser bin candidates
                if row['classification'] == 'Asteroid':
                    asteroid_group = events.at[i, 'asteroid_crossbin_group']
                    coarser = events[
                        (events['asteroid_crossbin_group'] == asteroid_group) &
                        (events['frame_bin'] > row['frame_bin'])
                    ].sort_values('frame_bin')
                else:
                    coarser = group_df[group_df['frame_bin'] > row['frame_bin']]

                for j, upper in coarser.iterrows():
                    consistent = (
                        (row['frame_start'] * row['frame_bin'] <= upper['frame_end'] * upper['frame_bin'] + upper['frame_bin']) &
                        (row['frame_end'] * row['frame_bin'] >= upper['frame_start'] * upper['frame_bin'] - upper['frame_bin'])
                    )
                    if consistent:
                        # Merge IDs, avoiding duplicates
                        merged = list(set(events.at[j, 'crossbin_ids'] + current_ids))
                        events.at[j, 'crossbin_ids'] = merged

        events = events.drop(columns=['crossbin_group', 'asteroid_crossbin_group'])

        # -- Find which indices are present more than once -- #
        all_ids = [i for ids in events['crossbin_ids'] for i in ids]
        all_referenced = set(i for i in all_ids if all_ids.count(i) > 1)

        # -- Clear solo crossbin_ids -- #
        events['crossbin_ids'] = events.apply(
            lambda row: [] if not any(i in all_referenced for i in row['crossbin_ids']) else row['crossbin_ids'], axis=1
        )

        # -- Remap crossbin_ids -- #
        id_map = {old: new for new, old in enumerate(sorted(all_referenced))}
        events['crossbin_ids'] = events['crossbin_ids'].apply(
            lambda x: [id_map[i] for i in x if i in id_map]
        )

        self.events = events
 
    def _order_events_columns(self):
        """
        Order events columns.
        """

        ordered_cols = [
            # Primary Identification
            'objid', 'eventid', 'classification',
            'sector', 'camera', 'ccd', 'cut','xcentroid', 'ycentroid',
            'frame_max',

            # Centroid Positions
            'xcentroid_err','ycentroid_err',
            'xint', 'yint','xccd', 'yccd',
            'xcentroid_det', 'ycentroid_det', 
            'xcentroid_psf', 'ycentroid_psf','centroid_err_psf',

            # Astrometry
            'ra', 'dec','ra_err','dec_err',
            'gal_l', 'gal_b',

            # Time and Frame Info
            'frame_start', 'frame_end', 'frame_duration',
            'mjd_start', 'mjd_end', 'mjd_max', 'mjd_duration',

            # Photometry
            'flux_max', 'mag_min','flux_sign',
            'image_sig_max', 'lc_sig_max', 'lc_sig_med','bkg_level',
            'snr_psf',

            # Morphology
            'psf_like', 'psf_diff','psf_stacked','com_motion','gaussian_score',

            # # Frequency Domain
            # 'peak_freq', 'peak_power',

            # Secondary Identification
            'source_mask', 'GaiaID', 'crossbin_ids', # 'prob', 'GaaID', 'cf_class', 'cf_prob', 

            # Miscellaneous
            'n_detections','total_events','frame_bin', 'TSS Catalogue'
        ]

        # Safely apply ordering
        self.events = self.events[[col for col in ordered_cols if col in self.events.columns]]

    def _TSS_catalogue_names(self):
        """
        Generate a TSS style catalogue name for each event.
        """

        from astropy.coordinates import SkyCoord
        import astropy.units as u

        tss_names = []
        for _,event in self.events.iterrows():
            c = SkyCoord(ra=event['ra'] * u.deg, dec=event['dec'] * u.deg)

            ra_hms = c.ra.hms
            dec_dms = c.dec.dms

            RAh = f'{int(ra_hms.h):02d}'
            RAm = f'{int(ra_hms.m):02d}'
            RAs = f'{ra_hms.s:05.2f}'

            sign = '+' if dec_dms.d >= 0 else '-'
            DECd = f'{abs(int(dec_dms.d)):02d}'
            DECm = f'{abs(int(dec_dms.m)):02d}'
            DECs = f'{abs(dec_dms.s):05.2f}'

            tss_name = f'TSS {RAh}{RAm}{RAs}{sign}{DECd}{DECm}{DECs}'
            if event['total_events']==1:
                tss_name += f"T{int(event['mjd_start'])}"
            tss_names.append(tss_name)
        self.events['TSS Catalogue'] = tss_names


    def find_events(self):

        # -- Group these sources into unique objects based on the objid -- #
        ts = clock()
        self._get_all_independent_events()
        print(f'   Separated into individual events -- done! ({(clock()-ts):.0f}s)')

        # -- Get physical units for events -- #
        self._events_physical_units()
        print(f'   Getting time/coords/flux information -- done!')

        # -- Tag asteroids -- #
        ts = clock()
        self._flag_asteroids()
        print(f'   Checking for asteroids -- done! ({(clock()-ts):.0f}s)')

        self.events = self.events.drop_duplicates(subset=['frame_bin','xint','yint','frame_max'],keep='first')

        # -- Tag asteroids -- #
        ts = clock()
        self._catalogue_crossmatch()
        print(f'   Crossmatching with Gaia and Variables -- done! ({(clock()-ts):.0f}s)')

        # -- Tag asteroids -- #
        ts = clock()
        self._crossmatch_framebin()
        print(f'   Crossmatching between time bins -- done! ({(clock()-ts):.0f}s)')

        # -- Get TSS Catalogue Names -- #        
        self._TSS_catalogue_names()
        print(f'   Getting TSS Catalogue Names -- done!')

        # -- Order nicely -- #
        self._order_events_columns()  

        # -- Save out results to csv file -- #
        self.events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_events.csv',index=False)
        


    # ------------------------------ Object finding function ------------------------------ #

    def find_objects(self):
        """
        Extract spatial objects.
        """

        objids = self.events['objid'].unique()

        columns = [
            'frame_bin','objid', 'sector', 'cam', 'ccd', 'cut', 'xcentroid', 'ycentroid', 
            'ra', 'dec', 'gal_l', 'gal_b', 'xcentroid_err','ycentroid_err','ra_err','dec_err',
            'lc_sig_max', 'flux_maxsig', 'frame_maxsig',
            'mjd_maxsig','psf_maxsig','flux_sign', 'n_events',
            'min_eventlength_frame', 'max_eventlength_frame',
            'min_eventlength_mjd','max_eventlength_mjd','GaiaID','classification','TSS Catalogue'
        ]
        objects = pd.DataFrame(columns=columns)

        for objid in objids:
            obj = self.events[self.events['objid'] == objid]

            maxevent = obj.iloc[obj['image_sig_max'].argmax()]

            if maxevent['classification'] == 'Asteroid' and len(obj) < 2:
                classification = 'Asteroid'
            else:
                classification = obj['classification'].mode()[0]

            if classification == 'RRLyrae':
                classification = 'VRRLyr'

            row_data = {
                'frame_bin':  maxevent['frame_bin'],
                'objid': objid,
                'xcentroid': maxevent['xcentroid'],
                'ycentroid': maxevent['ycentroid'],
                'ra': maxevent['ra'],
                'dec': maxevent['dec'],
                'gal_l': maxevent['gal_l'],
                'gal_b': maxevent['gal_b'],
                'xcentroid_err': maxevent['xcentroid_err'],
                'ycentroid_err': maxevent['ycentroid_err'],
                'ra_err': maxevent['ra_err'],
                'dec_err': maxevent['dec_err'],
                'image_sig_max': maxevent['image_sig_max'],
                'lc_sig_max': maxevent['lc_sig_max'],
                'flux_maxsig': maxevent['flux_max'],
                'frame_maxsig': maxevent['frame_max'],
                'mjd_maxsig': maxevent['mjd_max'],
                'psf_maxsig': maxevent['psf_like'],
                'flux_sign': np.sum(obj['flux_sign'].unique()).astype(int),
                'sector': maxevent['sector'],
                'cam': maxevent['camera'],
                'ccd': maxevent['ccd'],
                'cut': maxevent['cut'],
                'classification': classification,             
                'n_events': len(obj),
                'min_eventlength_frame': obj['frame_duration'].min(),
                'max_eventlength_frame': obj['frame_duration'].max(),
                'min_eventlength_mjd': obj['mjd_duration'].min(),
                'max_eventlength_mjd': obj['mjd_duration'].max(),
                'TSS Catalogue' : maxevent['TSS Catalogue'],
                'GaiaID' : maxevent['GaiaID']
            }

            obj_row = pd.DataFrame([row_data])
            objects = pd.concat([objects, obj_row], ignore_index=True)

        self.objects = objects

        self.objects.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_objects.csv',index=False)
        
        
    # ------------------------------ Main search function ------------------------------ #

    def transient_search(self,cut,mode='starfind',prf_path='/fred/oz335/_local_TESS_PRFs',time_bins=['10min']):

        import os

        # -- Check if using starfinder and/or sourcedetect for detection -- #
        self.mode = mode
        self.prf_path = prf_path
        
        # -- Gather time/flux data for the cut -- #
        print('-------Preloading sources / events-------')
        if cut != self.cut:
            self.gather_data(cut,bkg=True,mask=True,ref=True)

        # -- Preload self.sources and self.events if they're already made, self.objects can't be made otherwise this function wouldn't be called -- #
        self.gather_results(cut=cut)
        print('\n') 

        if self.sources is None:
            print('-------Source finding (see progress in errors log file)-------')
            self.find_sources(time_bins)
            print('\n')

        if not os.path.exists(f'{self.path}/Cut{cut}of{self.n**2}/wcs_info/snr_localisation_coeffs.pkl'):
            from .localisation import simulate_cut_psf_fitting

            print('-------Simulating PSF fit accuracy (see progress in errors log file)-------')
            simulate_cut_psf_fitting(self.data_path,self.sector,self.cam,self.ccd,cut,n=self.n,nfits=10000,nMedians=10)
            print('\n')

        # -- self.events contains all individual events, grouped by time and space -- #  
        if self.events is None:
            print('-------Event finding (see progress in errors log file)-------')
            self.find_events()
            print('\n')

        # -- self.objects contains all individual spatial objects -- #  
        print('-------Object finding-------')
        self.find_objects() 