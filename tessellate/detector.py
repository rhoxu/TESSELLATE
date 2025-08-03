# -- A good number of functions are imported only in the functions they get utilised -- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from .tools import RoundToInt


global_flux = None

# Reused LC Generation Function

def Generate_LC(time,flux,xint,yint,frame_start=None,frame_end=None,buffer=1):

    t = time
    f = flux

    if frame_start is not None:
        if frame_end is not None:
            t = t[frame_start:frame_end+1]
            f = f[frame_start:frame_end+1]
        else:
            t = t[frame_start:]
            f = f[frame_start:]   
    elif frame_end is not None:
        t = t[:frame_end+1]
        f = f[:frame_end+1]     

    f = np.nansum(f[:,yint-buffer:yint+buffer+1,xint-buffer:xint+buffer+1],axis=(1,2))

    return t,f

# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- # 
# ------------------------------------------------- Source Detection functions ------------------------------------------------ #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #

def _Spatial_group(result,min_samples=1,distance=0.5,njobs=-1):
    """
    Groups events based on proximity.
    """

    from sklearn.cluster import DBSCAN

    pos = np.array([result.xcentroid,result.ycentroid]).T
    cluster = DBSCAN(eps=distance,min_samples=min_samples,n_jobs=njobs).fit(pos)
    labels = cluster.labels_
    unique_labels = set(labels)
    for label in unique_labels:
        result.loc[label == labels,'objid'] = label + 1
    result['objid'] = result['objid'].astype(int)
    return result

def _Star_finding_procedure(data,prf,sig_limit = 2):

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

    xInts = res['xint'].values
    yInts = res['yint'].values

    res['source_mask'] = mask[yInts,xInts]

    return res

def _Count_detections(result):

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
    
def _Main_detection(flux,prf,inputNum,mode='both'):

    from time import time as t
    import multiprocessing
    from joblib import Parallel, delayed 
    from tqdm import tqdm

    
    print('    Starting source detection')
    length = np.linspace(0,flux.shape[0]-1,flux.shape[0]).astype(int)
    if mode == 'starfind':
        results = Parallel(n_jobs=int(multiprocessing.cpu_count()*3/4))(delayed(_Frame_detection)(flux[i],prf,inputNum+i) for i in tqdm(length))
        print('found sources')
        results = _Make_dataframe(results,flux[0])
        results['method'] = 'starfind'
    elif mode == 'sourcedetect':
        results = _Source_detect(flux,int(multiprocessing.cpu_count()*3/4))
        results['method'] = 'sourcedetect'
        results = results[~pd.isna(results['xcentroid'])]
    elif mode == 'both':
        t1 = t()
        results = Parallel(n_jobs=int(multiprocessing.cpu_count()*2/3))(delayed(_Frame_detection)(flux[i],prf,inputNum+i) for i in tqdm(length))
        star = _Make_dataframe(results,flux[0])
        star['method'] = 'starfind'
        print(f'        Done Starfind: {(t()-t1):.1f} sec')
        t1 = t()
        machine = _Source_detect(flux,int(multiprocessing.cpu_count()*3/4))
        machine = machine[~pd.isna(machine['xcentroid'])]
        machine['method'] = 'sourcedetect'
        print(f'        Done Sourcedetect: {(t()-t1):.1f} sec')

        results = pd.concat([
            star.assign(method='SF'),
            machine.assign(method='SD')
        ], ignore_index=True)

    return results

def _Collate_frame(frame):

    frame_sf = frame[frame['method'] == 'SF'].copy()
    frame_sd = frame[frame['method'] == 'SD'].copy()

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

def Detect(flux,cam,ccd,sector,column,row,mask,inputNums=None,corlim=0.6,psfdifflim=0.7,mode='starfind',
            datadir='/fred/oz335/_local_TESS_PRFs/'):
    """
    Main Function.
    """

    from PRF import TESS_PRF
    from time import time as t

    if inputNums is not None:
        flux = flux[inputNums]
        inputNum = inputNums[-1]
    else:
        inputNum = 0
            
    if sector < 4:
        prf = TESS_PRF(cam,ccd,sector,column,row,localdatadir=datadir+'Sectors1_2_3')
    else:
        prf = TESS_PRF(cam,ccd,sector,column,row,localdatadir=datadir+'Sectors4+')

    t1 = t()
    frame = _Main_detection(flux,prf,inputNum,mode=mode)
    print(f'    Main Search: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _Spatial_group(frame,distance=0.5,min_samples=1)        # groupd based on distance
    print(f'    Spatial Group: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _Collate_frame(frame)                   # prefer SourceDetect results
    print(f'    Collate Frame: {(t()-t1):.1f} sec')

    frame = frame.drop('objid',axis=1)

    t1 = t()
    frame = _Spatial_group(frame,distance=0.5,min_samples=2)                   # prefer SourceDetect results
    print(f'    Collate Frame: {(t()-t1):.1f} sec')

    single_isolated_detections = frame[frame['objid']==0]
    frame = frame[frame['objid']>0].reset_index(drop=True)

    t1 = t()
    frame = _Brightest_Px(flux,frame)               # Find brightest pixels around each source
    print(f'    Brightest Px: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _Source_mask(frame,mask)                # Add in TessReduce source mask value
    print(f'    Source Mask: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _Count_detections(frame)                # Count num detections for each objid
    print(f'    Count Detections: {(t()-t1):.1f} sec')

    return frame,single_isolated_detections

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

    time_per_frame = time[1] - time[0]
    buffer = int(buffer/time_per_frame)
    base_range = int(base_range/time_per_frame)
    ap = np.zeros_like(flux[0])
    y = pos[1] 
    x = pos[0] 
    _,lc = Generate_LC(time,flux,x,y,buffer=1)
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


def _Fit_period(time,flux,source,significance=3):

    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    import lightkurve as lk
    from astropy.stats import sigma_clipped_stats
    import astropy.units as u
    from astropy.time import Time
    from .tools import Exp_func

    x = RoundToInt(source['xint_brightest'])
    y = RoundToInt(source['yint_brightest'])
    
    t,f = Generate_LC(time,flux,x,y,buffer=1)
    # f = np.nansum(flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
    # t = time
    finite = np.isfinite(f) & np.isfinite(t)
    
    #ap = CircularAperture([source.xcentroid,source.ycentroid],1.5)
    #phot_table = aperture_photometry(data, aperture)
    #phot_table = phot_table.to_pandas()
    unit = u.electron / u.s
    t = time
    finite = np.isfinite(f) & np.isfinite(t)
    light = lk.LightCurve(time=Time(t[finite], format='mjd'),flux=(f[finite] - np.nanmedian(f[finite]))*unit)
    period = light.to_periodogram()

    x = period.frequency.value
    y = period.power.value
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    ind = (x > 2)
    #ind[:np.where(x < 2)[0][-1]] = False
    try:
        for i in range(2):
            popt, pcov = curve_fit(Exp_func, x[ind], y[ind])
            fit = Exp_func(x, *popt)
            m,med,std = sigma_clipped_stats(y - fit)
            ind = (y - fit) < (5 * std + med)

        norm = y/Exp_func(x, *popt)
        a = find_peaks(norm,prominence=3,distance=50,wlen=300,height=significance)
        peak_power = y[a[0]]
        peak_freq = x[a[0]]
        peak_freq = peak_freq[peak_power>1] 
        peak_power = peak_power[peak_power>1] 
        if peak_power is None:
            peak_power = [0]
            peak_freq = [0]
        elif len(peak_power) < 1:
            peak_power = [0]
            peak_freq = [0]
    except:
        peak_power = [0]
        peak_freq = [0]
    if peak_power is None:
            peak_power = [0]
            peak_freq = [0]
    return peak_freq, peak_power

def _Check_classifind(time,flux,source):
    import joblib
    from .temp_classifind import classifind as cf 
    import os

    package_directory = os.path.dirname(os.path.abspath(__file__))

    x = RoundToInt(source['xint_brightest'])
    y = RoundToInt(source['yint_brightest'])
    t,f = Generate_LC(time,flux,x,y,buffer=1)
    # f = np.nansum(flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
    # t = time
    finite = np.isfinite(f) & np.isfinite(t)
    lc = [np.column_stack((t[finite],f[finite]))]
    
    classes = {'Eclipsing Binary':'EB','Delta Scuti':'DSCT','RR Lyrae':'RRLyr','Cepheid':'Cep','Long-Period':'LPV',
                'Non-Variable':'Non-V','Non-Variable-B':'Non-V','Non-Variable-N':'Non-V'}
    try:
        model_path = os.path.join(package_directory,'rfc_files','RFC_model.joblib')
        classifier = joblib.load(model_path)
        cmodel = cf(lc,model=classifier,classes=list(classes.keys()))
        classification = classes[cmodel.class_preds[0]]
        if classification in ['Non-Variable','Non-Variable-B','Non-Variable-N']:
            prob = np.sum(cmodel.class_probs[0][-3:])
        else:
            prob = np.max(cmodel.class_probs)
    except:
        classification = 'Non-V'
        prob = 0.80001
    return classification, prob

def _Lightcurve_event_checker(lc_sig,triggers,siglim=3,maxsep=5):
    from .tools import consecutive_points

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

def _Fit_psf(flux,event,prf,frames):
    from .tools import PSF_Fitter

    xint_trial = np.round(event['xcentroid_det']).astype(int)
    yint_trial = np.round(event['ycentroid_det']).astype(int)

    try:
        stacked_flux = np.zeros((3, 3), dtype=np.float32)
        for i in frames:  # or whatever your cap is
            cut = flux[i, yint_trial-1:yint_trial+2, xint_trial-1:xint_trial+2]
            stacked_flux += event['flux_sign'] * cut

        iy, ix = np.unravel_index(np.nanargmax(stacked_flux), stacked_flux.shape)
        brightesty = yint_trial + (iy - 1)  # shift from 3x3 center
        brightestx = xint_trial + (ix - 1)

        event['xint_brightest'] = brightestx
        event['yint_brightest'] = brightesty

        centred_flux = np.zeros((5, 5), dtype=np.float32)
        for i in frames:  # or whatever your cap is
            cut = flux[i, brightesty-2:brightesty+3, brightestx-2:brightestx+3]
            centred_flux += event['flux_sign'] * cut

        PSF_fitter = PSF_Fitter(5,prf)
        PSF_fitter.fit_psf(centred_flux,limx=0.5,limy=0.5)

        event['xcentroid_psf'] = PSF_fitter.source_x + brightestx
        event['ycentroid_psf'] = PSF_fitter.source_y + brightesty

        # event['e_xcentroid_psf'] = PSF_fitter.source_x_err
        # event['e_ycentroid_psf'] = PSF_fitter.source_y_err

        r = np.corrcoef(centred_flux.flatten(), PSF_fitter.psf.flatten())
        r = r[0,1]
        event['psf_like'] = r
        event['psf_diff'] = np.nansum(abs(centred_flux/np.nansum(centred_flux)-PSF_fitter.psf))
    except:
        event['xcentroid_psf'] = np.nan
        event['ycentroid_psf'] = np.nan
        event['psf_like'] = np.nan
        event['psf_diff'] = np.nan

    # plt.figure()
    # plt.imshow(centred_flux,origin='lower',cmap='gray',vmin=np.nanmin(centred_flux),vmax=np.nanmax(centred_flux))
    # plt.scatter(PSF_fitter.source_x+2,PSF_fitter.source_y+2)
    # plt.scatter(event['xcentroid_det']-(brightestx-2),event['ycentroid_det']-(brightesty-2))
    # plt.xlabel(event['flux_sign'])

    return event

        

def _Isolate_events(objid,time,flux,sources,sector,cam,ccd,cut,prf,frame_buffer=5,buffer=1,base_range=1,verbose=False):
    """_summary_

    Args:
        objid (int): ID for source to isolate individual events.
        frame_buffer: number of break frames permitted for continous events.
        buffer (float, optional): Space between event and baseline in days. Defaults to 0.5.
        duration (int, optional): Duration of time either side of event to create baseline in days. Defaults to 1.

    Returns:
        events
    """

    from .tools import pandas_weighted_avg
    from time import time as t

    # -- Select all sources grouped to this objid -- #
    source = sources[sources['objid']==objid]
    
    all_labelled_sources = []


    # -- For this objid, separate positive and negative detections -- #
    startingID=1
    for sign in source['flux_sign'].unique():
        signed_sources = source[source['flux_sign']==sign]

        weighted_signedsources = pandas_weighted_avg(signed_sources)

        # -- Look for any regular variability -- #
        peak_freq, peak_power = _Fit_period(time,flux,weighted_signedsources.iloc[0])

        # -- Run RFC Classification -- #
        cf_classification, cf_prob = _Check_classifind(time,flux,weighted_signedsources.iloc[0])

        labelled_sources = _Get_temporal_events(signed_sources,max_gap=frame_buffer,startingID=startingID)
        all_labelled_sources.append(labelled_sources)
        startingID = np.nanmax(labelled_sources['eventid'])+1
    
    all_labelled_sources = pd.concat(all_labelled_sources, ignore_index=True)
    
    n_events = np.nanmax(all_labelled_sources['eventid'])

    if verbose:
        print(f"Objid: {objid} , n_events: {n_events}")

    # goodevents=[]
    # for eventID,group in all_labelled_sources.groupby('eventid'):
    #     if len(group[group.sig>5]) != 0:
    #         goodevents.append(eventID)

    goodevents = all_labelled_sources['eventid'].unique()

    dfs = []
    for eventID in all_labelled_sources['eventid'].unique():
        if verbose:
            print(f'Event {eventID} of {n_events}')
        event = {}
        eventsources = deepcopy(all_labelled_sources[all_labelled_sources['eventid']==eventID])
        weighted_eventsources = pandas_weighted_avg(eventsources)

        xint = RoundToInt(weighted_eventsources.iloc[0]['xint_brightest'])
        yint = RoundToInt(weighted_eventsources.iloc[0]['yint_brightest'])

        # -- Calculate significance of detection above background and local light curve -- #
        _, _, sig_lc, _, _ = _Check_LC_significance(time,flux,eventsources['frame'].min(),eventsources['frame'].max(),
                                                                [xint,yint],sign,buffer=buffer,base_range=base_range)
        
        frame_start,frame_end,n_detections,frames = _Lightcurve_event_checker(sig_lc,eventsources['frame'].values,siglim=3,maxsep=5)

        event['objid'] = objid
        event['eventid'] = eventID
        event['sector'] = sector
        event['camera'] = cam
        event['ccd'] = ccd
        event['cut'] = cut
        event['classification'] = eventsources.iloc[0]['classification']
        event['frame_start'] = frame_start
        event['frame_end'] = frame_end
        event['frame_duration'] = event['frame_end']-event['frame_start']+1
        event['flux_sign'] = eventsources.iloc[0]['flux_sign']
        event['n_detections'] = n_detections
        event['bkg_level'] = weighted_eventsources.iloc[0]['bkg_level']
        event['bkg_std'] = weighted_eventsources.iloc[0]['bkgstd']

        event['xcentroid_det'] = weighted_eventsources.iloc[0]['xcentroid']
        event['ycentroid_det'] = weighted_eventsources.iloc[0]['ycentroid']

        if eventID in goodevents:
            event = _Fit_psf(flux,event,prf,frames)
            # event['xcentroid_psf'] = np.nan
            # event['ycentroid_psf'] = np.nan
            # event['psf_like'] = np.nan
            # event['xint_brightest'] = RoundToInt(weighted_eventsources.iloc[0]['xint_brightest'])
            # event['yint_brightest'] = RoundToInt(weighted_eventsources.iloc[0]['yint_brightest']) 
        else:
            event['xcentroid_psf'] = np.nan
            event['ycentroid_psf'] = np.nan
            event['psf_like'] = np.nan
            event['xint_brightest'] = RoundToInt(weighted_eventsources.iloc[0]['xint_brightest'])
            event['yint_brightest'] = RoundToInt(weighted_eventsources.iloc[0]['yint_brightest'])

        if event['psf_like']>0.6:
            event['xcentroid'] = event['xcentroid_psf']
            event['ycentroid'] = event['ycentroid_psf']
        else:
            event['xcentroid'] = event['xcentroid_det']
            event['ycentroid'] = event['ycentroid_det']

        event['xint'] = RoundToInt(event['xcentroid'])
        event['yint'] = RoundToInt(event['ycentroid'])

        sig_max, sig_med, _, max_flux, max_frame = _Check_LC_significance(time,flux,
                                                                event['frame_start'],event['frame_end'],
                                                                [event['xint'],event['yint']],
                                                                sign,buffer=buffer,base_range=base_range)

        event['frame_max'] = max_frame
        event['flux_max'] = max_flux
        event['image_sig_max'] = np.nanmax(eventsources['sig'].values)
        event['lc_sig_max'] = sig_max
        event['lc_sig_med'] = sig_med

        event['peak_freq'] = peak_freq[0]
        event['peak_power'] = peak_power[0]
        event['cf_class'] = cf_classification
        event['cf_prob'] = cf_prob
        event['GaiaID'] = eventsources.iloc[0]['GaiaID']
        event['source_mask'] = eventsources.iloc[0]['source_mask']

 
        df = pd.DataFrame([event])
        dfs.append(df)

    events_df = pd.concat(dfs, ignore_index=True)

    # -- Add to each dataframe row the number of events in the total object -- #
    events_df['total_events'] = len(events_df)
    
    return events_df 

def _Isolate_events_safe(id, time, flux_file, sources, sector,cam,ccd,cut,prf,frame_buffer,buffer,base_range):
    from joblib import load
    # log_worker_mem(f"start i={id}")
    flux = load(flux_file, mmap_mode='r')
    # result = _Isolate_events(id, time, flux, sources, sector,cam,ccd,cut,prf,frame_buffer,buffer,base_range)
    # log_worker_mem(f"end i={id}")
    return _Isolate_events(id, time, flux, sources, sector,cam,ccd,cut,prf,frame_buffer,buffer,base_range)


# def log_worker_mem(tag=""):
#     import os
#     import psutil
#     proc = psutil.Process(os.getpid())
#     mem_mb = proc.memory_info().rss / 1024**2
#     print(f"[PID {proc.pid} {tag}] Memory RSS: {mem_mb:.2f} MB")

    
def _Straight_line_asteroid_checker(time,flux,events):
    from astropy.stats import sigma_clipped_stats
    import cv2

    events = deepcopy(events)
    for i in range(len(events)):
        source = events.iloc[i]
        if source['classification'] != 'Asteroid':                
            frameStart = source['frame_start']
            frameEnd = source['frame_end']
            x = source['xint']; y = source['yint']
            xl = np.max((x - 5,0)); xu = np.min((x + 6,flux.shape[2]-1))
            yl = np.max((y - 5,0)); yu = np.min((y + 6,flux.shape[1]-1))
            fs = np.max((frameStart - 5, 0))
            fe = np.min((frameEnd + 5, len(time)-1))

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

            if (lines is not None) & (source['psf_like']>=0.8):
                events.loc[i, 'classification'] = 'Asteroid'
                events.loc[i, 'prob'] = 0.5
        
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

    from .tools import Gaussian
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

            t,f = Generate_LC(time,flux,x,y,frameStart,frameEnd,buffer=1)

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

    events = deepcopy(events)

    # -- Identify candidates to actually compute on -- #
    candidates = events[(events['frame_duration']>=2)&(events['frame_duration']<=50)]
    candidates = candidates[(candidates['lc_sig_max']>=5)&(candidates['flux_sign']==1)]
    # candidates = self.filter_events(self.cut, lower=2, upper=50, sig_lc=5, sign=1)

    candidate_indices = candidates.index

    # -- #Run motion and Gaussian score only on those candidates -- #
    candidates = _Calculate_xcom_motion(flux,candidates)
    candidates = _Gaussian_score(time,flux,candidates)

    # -- Add results back to full events -- #
    events['com_motion'] = np.nan
    events['gaussian_score'] = np.nan
    events.loc[candidate_indices, 'com_motion'] = candidates['com_motion'].values
    events.loc[candidate_indices, 'gaussian_score'] = candidates['gaussian_score'].values

    # -- Flagging loop -- #
    for com_thresh, gauss_thresh in zip(com_motion_thresholds, gaussian_score_thresholds):
        mask = (
            (events['com_motion'] >= com_thresh) &
            (events['gaussian_score'] >= gauss_thresh)
        )
        events.loc[mask, 'classification'] = 'Asteroid'
        events.loc[mask, 'prob'] = 0.8

    return events



class Detector():

    def __init__(self,sector,cam,ccd,data_path,n,
                 match_variables=True,mode='both',part=None,time_bin=None):

        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.data_path = data_path
        self.n = n
        self.match_variables = match_variables
        self.time_bin = time_bin

        self.flux = None
        self.ref = None
        self.time = None
        self.mask = None
        self.sources = None  #raw detection results
        self.events = None   #temporally located with same object id
        self.objects = None   #temporally and spatially combined
        self.cut = None
        self.bkg = None
        

        self.mode = mode

        if part is None:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}'
        elif part == 1:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Part1'
        elif part == 2:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Part2'
        else:
            e = 'Invalid Part Parameter!'
            raise AttributeError(e)

    def _wcs_time_info(self,result,cut):
        
        from .tools import CutWCS

        times = np.load(f'{self.path}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}_Times.npy')

        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=cut,n=self.n)
        coords = self.wcs.all_pix2world(result['xcentroid'],result['ycentroid'],0)
        result['ra'] = coords[0]
        result['dec'] = coords[1]
        result['mjd'] = times[result['frame']]
 
        return result
    
    def _gather_data(self,cut):

        from .tools import CutWCS
        
        self.cut = cut
        base = f'{self.path}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{self.cut}_of{self.n**2}'
        self.base_name = base
        self.flux = np.load(base + '_ReducedFlux.npy')
        self.ref = np.load(base + '_Ref.npy')
        try:
            self.bkg = np.load(base + '_Background.npy')
        except:
            pass
        self.mask = np.load(base + '_Mask.npy')
        self.time = np.load(base + '_Times.npy')
        if self.time_bin is not None:
            self._rebin_data()
        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=cut,n=self.n)

    def _rebin_data(self):
        points = np.arange(self.time[0]+time_bin*.5,self.time[-1],self.time_bin)
        flux = []
        bkg = []
        for i in range(len(points)):
            ind = abs(points[i] - self.time) <= time_bin/2
            flux += [np.nanmean(self.flux[ind])]
            bkg += [np.nanmean(self.bkg[ind])]
        flux = np.array(flux)
        bkg = np.array(bkg)
        
        self.time = points
        self.flux = flux
        self.bkg = bkg
    
    def _recheck_asteroid_lcs(self,events):

        asteroids = events[events['classification']=='Asteroid']
        if len (asteroids) > 0:
            for _, ast in asteroids.iterrows():
                objid = ast['objid']
                eventid = ast['eventid']
                xint = ast['xint']
                yint = ast['yint']
                frame_start = ast['frame_start']
                frame_end = ast['frame_end']

                _, _, lc_sig, _, _ = _Check_LC_significance(
                    self.time, self.flux, frame_start, frame_end, [xint, yint], 1, 0.5, 1
                )

                start, end, _, _ = _Lightcurve_event_checker(
                    lc_sig, np.arange(frame_start, frame_end+1)
                )

                n_detections = np.nansum([frame_start - start, end - frame_end])

                # Update d.events
                event_mask = (events['objid'] == objid) & (events['eventid'] == eventid)
                events.loc[event_mask, 'frame_start'] = start
                events.loc[event_mask, 'frame_end'] = end
                events.loc[event_mask, 'mjd_start'] = self.time[start]
                events.loc[event_mask, 'mjd_end'] = self.time[end]
                events.loc[event_mask, 'frame_duration'] = end - start
                events.loc[event_mask, 'mjd_duration'] = self.time[end] - self.time[start]
                events.loc[event_mask, 'n_detections'] += n_detections

        return events
            
    def _flag_asteroids(self):

        events = deepcopy(self.events)

        # -- Generally best, checks for centre of mass movement and light curve Gaussianity -- #
        events = _Threshold_asteroid_checker(self.time,self.flux,events)

        # -- Picks up events with weirdly long event boundaries -- # 
        events = _Straight_line_asteroid_checker(self.time,self.flux,events)

        events = self._recheck_asteroid_lcs(events)

        self.events = events
    

    def _get_all_independent_events(self,frame_buffer=10,buffer=0.5,base_range=1,cpu=1):
        from joblib import Parallel, delayed, dump 
        from tqdm import tqdm
        from .dataprocessor import DataProcessor
        from PRF import TESS_PRF
        import os
    

        dp = DataProcessor(self.sector,path=self.data_path)
        cutCornerPx, cutCentrePx, _, _ = dp.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        column = cutCentrePx[self.cut-1][0]
        row = cutCentrePx[self.cut-1][1]

        datadir='/fred/oz335/_local_TESS_PRFs/'
        prf = TESS_PRF(cam=self.cam,ccd=self.ccd,sector=self.sector,
                       colnum=column,rownum=row,localdatadir=datadir+'Sectors4+')

        ids = np.unique(self.sources['objid'].values).astype(int)
        if cpu > 1:
            length = np.arange(0,len(ids)).astype(int)
            events = Parallel(n_jobs=cpu)(delayed(_Isolate_events)(ids[i],self.time,self.flux,self.sources,
                                                                   self.sector,self.cam,self.ccd,self.cut,prf,
                                                                   frame_buffer,buffer,base_range) for i in tqdm(length))
        else:            
            events = []
            for id in ids:
                e = _Isolate_events(id,self.time,self.flux,self.sources,self.sector,self.cam,
                                    self.ccd,self.cut,prf,frame_buffer=frame_buffer,
                                    buffer=buffer,base_range=base_range)
                events += [e]

        events = pd.concat(events,ignore_index=True)

        events['xccd'] = RoundToInt(events['xint'] + cutCornerPx[self.cut-1][0])
        events['yccd'] = RoundToInt(events['yint'] + cutCornerPx[self.cut-1][1])

        self.events = events 

    def _events_physical_units(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        events = deepcopy(self.events)

        ra,dec = self.wcs.all_pix2world(events['xcentroid'],events['ycentroid'],0)
        events['ra'] = ra
        events['dec'] = dec

        coords = SkyCoord(ra=events['ra'].values*u.degree,dec=events['dec'].values*u.degree)
        events['gal_l'] = coords.galactic.l.value
        events['gal_b'] = coords.galactic.b.value

        events['mjd_start'] = self.time[events['frame_start']]
        events['mjd_end'] = self.time[events['frame_end']]
        events['mjd_duration'] = events['mjd_end'] - events['mjd_start']
        events['mjd_max'] = self.time[events['frame_max']]

        events['mag_min'] = -2.5*np.log10(events['flux_max'])

        self.events = events
        
    def _order_events_columns(self):

        ordered_cols = [
            # Primary Identification
            'objid', 'eventid', 'classification', 'TSS Catalogue',
            'sector', 'camera', 'ccd', 'cut',

            # Centroid Positions
            'xcentroid', 'ycentroid', 'xint', 'yint',
            'xccd', 'yccd',
            'xcentroid_det', 'ycentroid_det', 
            'xcentroid_psf', 'ycentroid_psf',

            # Astrometry
            'ra', 'dec', 'gal_l', 'gal_b',

            # Time and Frame Info
            'frame_start', 'frame_end', 'frame_max', 'frame_duration',
            'mjd_start', 'mjd_end', 'mjd_max', 'mjd_duration',

            # Photometry
            'flux_max', 'mag_min','flux_sign',
            'image_sig_max', 'lc_sig_max', 'lc_sig_med','bkg_level',

            # Morphology
            'psf_like', 'psf_diff','com_motion','gaussian_score',

            # Frequency Domain
            'peak_freq', 'peak_power',

            # Secondary Identification
            'prob','GaiaID', 'cf_class', 'cf_prob', 'source_mask',

            # Miscellaneous
            'n_detections','total_events'
        ]

        # Safely apply ordering
        self.events = self.events[[col for col in ordered_cols if col in self.events.columns]]
    

    def _TSS_catalogue_names(self):

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

    def _gather_results(self,cut,sources=True,events=True,objects=True):
        """
        Gather the results of the source detection for a given cut.
        """

        import multiprocessing
        from .tools import CutWCS

        self.objects = None
        # self.cut = cut
        path = f'{self.path}/Cut{cut}of{self.n**2}'

        if sources:
            try:
                self.sources = pd.read_csv(f'{path}/detected_sources.csv')    # raw detection results
            except:
                print('No detected events file found')
                self.sources = None

        if events:
            try:
                self.events = pd.read_csv(f'{path}/detected_events.csv')    # temporally located with same object id
            except:
                print('No detected events file found')
                self.events = None

        if objects: 
            try:
                self.objects = pd.read_csv(f'{path}/detected_objects.csv')    # temporally and spatially located with same object id
            except:
                print('No detected objects file found')
                self.objects = None
        
        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=cut,n=self.n)

    def _find_sources(self,mode,prf_path):

        from .dataprocessor import DataProcessor
        from .catalog_queries import match_result_to_cat #,find_variables, gaia_stars,
        from .tools import pandas_weighted_avg
        
        # -- Access information about the cut with respect to the original ccd -- #        
        processor = DataProcessor(sector=self.sector,path=self.data_path,verbose=2)
        cutCorners, cutCentrePx, _, _ = processor.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        column = cutCentrePx[self.cut-1][0]
        row = cutCentrePx[self.cut-1][1]

        # -- Run the detection algorithm, generates dataframe -- #
        results,single_isolated_detections = Detect(self.flux,cam=self.cam,ccd=self.ccd,sector=self.sector,column=column,
                         row=row,mask=self.mask,inputNums=None,mode=mode,datadir=prf_path)
        
        # -- Save out single detections which are isolated in space in time, probably noise, maybe cool -- #
        single_isolated_detections.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/single_isolated_detections.csv',index=False)
        
        # -- Add wcs, time, ccd info to the results dataframe -- #
        results = self._wcs_time_info(results,self.cut)
        results['xccd'] = deepcopy(results['xcentroid'] + cutCorners[self.cut-1][0])
        results['yccd'] = deepcopy(results['ycentroid'] + cutCorners[self.cut-1][1])
        
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
        
        # -- Cross matches location to Gaia -- #
        gaia = pd.read_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/local_gaia_cat.csv')
        results = match_result_to_cat(deepcopy(results),gaia,columns=['Source'])
        results = results.rename(columns={'Source': 'GaiaID'})

        # -- Cross matches location to variable catalog -- #
        variables = pd.read_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/variable_catalog.csv')
        results = match_result_to_cat(deepcopy(results),variables,columns=['Type','Prob'])

        # -- Calculates the background level for each source -- #
        results['bkg_level'] = 0
        if self.bkg is not None:
            f = results['frame'].values
            x = results['xint'].values; y = results['yint'].values
            bkg = self.bkg
            b = []
            for i in range(3):
                i-=1
                for j in range(3):
                    j-=1
                    b += [bkg[f,y-i,x+j]]
            b = np.array(b)
            b = np.nansum(b,axis=0)
            results['bkg_level'] = b

        results.loc[results['GaiaID'] == 0, 'GaiaID'] = '-'
        results.loc[results['Type'] == 0, 'Prob'] = '-'
        results.loc[results['Type'] == 0, 'Type'] = '-'
        results = results.rename(columns={'Type': 'classification'})
        results = results.rename(columns={'Prob': 'prob'})

        # Save detected sources out.
        if self.time_bin is None:
            results.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_sources.csv',index=False)
        else:
            results.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_sources_tbin{self.time_bin_name}d.csv',index=False)

        self.sources = results
    
    def _find_events(self):
        from time import time as t
        import multiprocessing

        # -- Group these sources into unique objects based on the objid -- #
        t2 = t()
        print('    Separating into individual events...',end='\r')
        self._get_all_independent_events(cpu = int(multiprocessing.cpu_count()))
        print(f'   Separating into individual events... Done in {(t()-t2):.1f} sec')

        t2 = t()
        print('    Getting time/coords/flux information...',end='\r')
        self._events_physical_units()
        print(f'   Getting time/coords/flux information... Done in {(t()-t2):.1f} sec')

        # -- Tag asteroids -- #
        t2 = t()
        print('    Checking for asteroids...',end='\r')
        self._flag_asteroids()
        print(f'   Checking for asteroids... Done in {(t()-t2):.1f} sec')

        # -- Get TSS Catalogues Names -- #        
        print('    Getting TSS Catalogue Names...',end='\r')
        self._TSS_catalogue_names()
        print(f'   Getting TSS Catalogue Names... Done!')

        self._order_events_columns()  

        # -- Save out results to csv files -- #
        if self.time_bin is None:
            self.events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_events.csv',index=False)
        else:
            self.events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_events_tbin{self.time_bin_name}d.csv',index=False)

    def _find_objects(self):

        objids = self.events['objid'].unique()

        columns = [
            'objid', 'sector', 'cam', 'ccd', 'cut', 'xcentroid', 'ycentroid', 
            'ra', 'dec', 'gal_l', 'gal_b', 'lc_sig_max', 'flux_maxsig', 'frame_maxsig',
            'mjd_maxsig','psf_maxsig','flux_sign', 'n_events',
            'min_eventlength_frame', 'max_eventlength_frame',
            'min_eventlength_mjd','max_eventlength_mjd','GaiaID','classification','TSS Catalogue'
        ]
        objects = pd.DataFrame(columns=columns)

        for objid in objids:
            obj = self.events[self.events['objid'] == objid]

            maxevent = obj.iloc[obj['lc_sig_max'].argmax()]

            if maxevent['classification'] == 'Asteroid' and len(obj) < 2:
                classification = 'Asteroid'
            else:
                classification = obj['classification'].mode()[0]

            if classification == 'RRLyrae':
                classification = 'VRRLyr'

            row_data = {
                'objid': objid,
                'xcentroid': maxevent['xcentroid'],
                'ycentroid': maxevent['ycentroid'],
                'ra': maxevent['ra'],
                'dec': maxevent['dec'],
                'gal_l': maxevent['gal_l'],
                'gal_b': maxevent['gal_b'],
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

        if self.time_bin is None:
            self.objects.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_objects.csv',index=False)
        else:
            self.objects.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_objects_tbin{self.time_bin_name}d.csv',index=False)

    def source_detect(self,cut,mode='starfind',prf_path='/fred/oz335/_local_TESS_PRFs/',time_bin=None):
        """
        Run the source detection algorithm on the data for a given cut.
        """

        # -- Check if using starfinder and/or sourcedetect for detection -- #
        if mode is None:
            mode = self.mode
        if (mode == 'both') | (mode == 'starfind') | (mode == 'sourcedetect'):
            pass
        else:
            m = 'Mode must be one of the following: both, starfind, sourcedetect.'
            raise ValueError(m)
        
        # -- Gather time/flux data for the cut -- #
        if cut != self.cut:
            self._gather_data(cut)
            self.cut = cut

        # -- Preload self.sources and self.events if they're already made, self.objects can't be made otherwise this function wouldn't be called -- #
        print('Preloading sources / events')
        self._gather_results(cut=cut,objects=False)  
        print('\n') 

        if time_bin is not None:
            self.time_bin = time_bin

        # -- self.sources contains all individual sources found in all frames -- #
        if self.sources is None:
            print('-------Source finding (see progress in errors log file)-------')
            self._find_sources(mode,prf_path)
            print('\n')

        # -- self.events contains all individual events, grouped by time and space -- #  
        if self.events is None:
            print('-------Event finding (see progress in errors log file)-------')
            self._find_events()
            print('\n')

        # -- self.objects contains all individual spatial objects -- #  
        print('-------Object finding-------')
        self._find_objects()


    def display_source_locations(self,cut):
        """
        Overlays the source locations on the first frame of the cut.
        """

        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        fig,ax = plt.subplots(figsize=(12,6),ncols=2)
        ax[0].scatter(self.sources['xcentroid'],self.sources['ycentroid'],c=self.sources['frame'],s=5)
        ax[0].imshow(self.flux[0],cmap='gray',origin='lower',vmin=-10,vmax=10)
        ax[0].set_xlabel(f'Frame 0')

        newmask = deepcopy(self.mask)

        c1 = newmask.shape[0]//2
        c2 = newmask.shape[1]//2

        newmask[c1-2:c1+3,c2-2:c2+3] -= 1

        ax[1].imshow(newmask,origin='lower')

        ax[1].scatter(self.sources['xcentroid'],self.sources['ycentroid'],c=self.sources['source_mask'],s=5,cmap='Reds')
        ax[1].set_xlabel('Source Mask')


    def filter_objects(self,cut,
                       ra=None,dec=None,distance=40,
                       min_events=None,max_events=None,
                       classification=None,flux_sign=None,
                       lc_sig_max=None,psf_like=None,
                       min_eventlength_frame=None,max_eventlength_frame=None,
                       min_eventlength_mjd=None,max_eventlength_mjd=None):
        
        """
        Filter self.objects based on these main things.
        """
        
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        # -- Gather results and data -- #
        # if (cut != self.cut) | :
            # self._gather_data(cut)
        self._gather_results(cut)
            # self.cut = cut

        objects = deepcopy(self.objects)

        if (ra is not None) & (dec is not None):
            if type(ra) == float:
                target_coord = SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
            elif 'd' in dec:
                SkyCoord(ra=ra,dec=dec)
            else:
                e = 'Please specify coordinates in (deg,deg) or (hms,dms)!'
                raise ValueError(e)
            
            source_coords = SkyCoord(ra=objects['ra'].values*u.degree, 
                                     dec=objects['dec'].values*u.degree)
            separations = target_coord.separation(source_coords)
            objects = objects[separations<distance*u.arcsec]

        if min_events is not None:
            objects = objects[objects['n_events']>=min_events]
        if max_events is not None:
            objects = objects[objects['n_events']<=max_events]

        if classification is not None:
            is_negation = classification.startswith(('!', '~'))
            classification_stripped = classification.lstrip('!~').lower()
            if classification_stripped in ['var', 'variable']:
                classification = classification = ['VCR', 'VRRLyr', 'VEB','VLPV','VST','VAGN','VRM','VMSO']  # Replace with variable classes
            else:
                classification = [classification_stripped]

            if is_negation:
                objects = objects[~objects['classification'].str.lower().isin([classification[i].lower() for i in range(len(classification))])]
            else:
                objects = objects[objects['classification'].str.lower().isin([classification[i].lower() for i in range(len(classification))])]

        if flux_sign is not None:
            objects = objects[objects['flux_sign']==flux_sign]
        
        if lc_sig_max is not None:
            objects = objects[objects['lc_sig_max']>=lc_sig_max]

        if psf_like is not None:
            objects = objects[objects['psf_maxsig']>=psf_like]

        if min_eventlength_frame is not None:
            objects = objects[objects['min_eventlength_frame']>=min_eventlength_frame]
        if max_eventlength_frame is not None:
            objects = objects[objects['max_eventlength_frame']<=max_eventlength_frame]
        
        if min_eventlength_mjd is not None:
            objects = objects[objects['min_eventlength_mjd']>=min_eventlength_mjd]
        if max_eventlength_mjd is not None:
            objects = objects[objects['max_eventlength_mjd']<=max_eventlength_mjd]

        return objects

    def filter_events(self,cut,starkiller=False,asteroidkiller=False,lower=None,upper=None,image_sig_max=None,
                      lc_sig_max=None,lc_sig_med=None,max_events=None,bkg_std=None,
                      flux_sign=None,classification=None,psf_like=None,galactic_latitude=None):
        
        """
        Returns a dataframe of the events in the cut, with options to filter by various parameters.
        """

        # -- Gather results and data -- #
            # if cut != self.cut:
            # self._gather_data(cut)
        self._gather_results(cut)
            # self.cut = cut

        # -- If true, remove events near sources in reduction source mask (ie. stars) -- #
        if starkiller:
            r = self.events.loc[self.events['source_mask']==0]
        else:
            r = self.events

        # # -- If true, remove asteroids from the results -- #
        if asteroidkiller:
            r = r.loc[~(r['classification'] == 'Asteroid')]

        if classification is not None:
            is_negation = classification.startswith(('!', '~'))
            classification_stripped = classification.lstrip('!~').lower()
            if classification_stripped in ['var', 'variable']:
                classification = classification = ['VCR', 'VRRLyr', 'VEB','VLPV','VST','VAGN','VRM','VMSO','RRLyrae']  
            else:
                classification = [classification_stripped]

            if is_negation:
                r = r[~r['classification'].str.lower().isin([classification[i].lower() for i in range(len(classification))])]
            else:
                r = r[r['classification'].str.lower().isin([classification[i].lower() for i in range(len(classification))])]

        # -- Filter by various parameters -- #
        if lc_sig_max is not None:
            r = r.loc[r['lc_sig_max']>=lc_sig_max]
        if lc_sig_med is not None:
            r = r.loc[r['lc_sig_med']>=lc_sig_med]
        if image_sig_max is not None:
            r = r.loc[r['image_sig_max'] >= image_sig_max]
        if max_events is not None:
            r = r.loc[r['total_events'] <= max_events]
        if bkg_std is not None:
            r = r.loc[r['bkg_std'] <= bkg_std]
        if flux_sign is not None:
            r = r.loc[r['flux_sign'] == flux_sign]
        if psf_like is not None:
            r = r.loc[r['psf_like']>=psf_like]

        # -- Filter by upper and lower limits on number of detections within each event -- #
        if upper is not None:
            if lower is not None:
                r = r.loc[(r['frame_duration'] <= upper) & (r['frame_duration'] >= lower)]
            else:
                r = r.loc[(r['frame_duration'] <= upper)]
        elif lower is not None:
            r = r.loc[(r['frame_duration'] >= lower)]

        return r



    def lc_ALL(self,cut,save_path=None,lower=2,max_events=None,starkiller=False,
                 image_sig_max=3,lc_sig_max=2.5,bkgstd_lim=100,flux_sign=None):
        """
        Generates all light curves for the detections in the cut to a zip file of csvs.
        """
        
        import multiprocessing
        from joblib import Parallel, delayed 
        import os
        from .tools import _Check_dirs

        # -- Gather detections to be considered for plotting -- #
        detections = self.filter_events(cut=cut,lower=lower,max_events=max_events,
                                        flux_sign=flux_sign,starkiller=starkiller,
                                        lc_sig_max=lc_sig_max,image_sig_max=image_sig_max)
        
        self._gather_data(cut=cut)
        
        # -- Generate save path and name -- #
        if save_path is None:
            save_path = self.path + f'/Cut{cut}of{self.n**2}/lcs/'
            print('LC path: ',save_path)
            _Check_dirs(save_path)
            save_name = save_path + f'Sec{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}'
        inds = detections['objid'].unique()

        # -- Run the saving in parallel -- #
        print('Total lcs to create: ', len(inds))
        events = Parallel(n_jobs=int(multiprocessing.cpu_count()))(delayed(Save_LC)(self.time,self.flux,detections,ind,save_name) for ind in inds)
        print('LCs complete!')

        # -- Now zips files and then deletes the directory to save inodes -- #
        cwd = os.getcwd()
        os.chdir(f'{save_path}/..')
        print('Zipping...')
        cmd = f"zip -r lcs.zip lcs > /dev/null 2>&1"
        os.system(cmd)
        print('Zip complete!')
        os.chdir(cwd)

        print('Deleting...')
        os.system(f'rm -r {save_path}')
        print('Delete complete!')

    
    def plot_ALL(self,cut,save_path=None,lower=3,max_events=30,starkiller=False,
                 image_sig_max=3,lc_sig_max=2.5,bkgstd_lim=100,flux_sign=1,time_bin=None):
        """
        Generates all source plots for the detections in the cut to a zip file of pngs
        """
        
        import multiprocessing
        from joblib import Parallel, delayed 
        import os
        from .tools import _Check_dirs

        # if time_bin is not None:
        #     self.time_bin = time_bin

        # -- Gather detections to be considered for plotting -- #
        detections = self.filter_events(cut=cut,lower=lower,max_events=max_events,
                                        flux_sign=flux_sign,starkiller=starkiller,lc_sig_max=lc_sig_max,image_sig_max=image_sig_max)
        
        self._gather_data(cut=cut)

        # -- Generate save path and name -- #
        if save_path is None:
            save_path = self.path + f'/Cut{cut}of{self.n**2}/figs/'
            print('Figure path: ',save_path)
            _Check_dirs(save_path)
            save_name = save_path + f'Sec{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}'

        # -- Count unique object IDs in detections -- #
        inds = detections['objid'].unique()
        print('Total events to plot: ', len(detections))

        # -- Run the plotting in parallel -- #
        events = Parallel(n_jobs=int(multiprocessing.cpu_count()))(delayed(Plot_Object)(self.time,self.flux,detections,
                                                                                        ind,event='seperate',latex=True,
                                                                                        save_path=save_name,zoo_mode=True) for ind in inds)
        print('Plot complete!')

        # -- Now zips files and then deletes the directory to save inodes -- #
        print('Zipping...')
        cmd = f"find {save_path} -type f -name '*.png' -exec zip {save_path}/../figs.zip -j {{}} + > /dev/null 2>&1"
        os.system(cmd)
        print('Zip complete!')
        print('Deleting...')
        os.system(f'rm -r {save_path}')
        print('Delete complete!')

    def save_lc(self,cut,id,save_path=None):
        from .tools import _Check_dirs

        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        if save_path is None:
            save_path = self.path + f'/Cut{cut}of{self.n**2}/lcs/'
            _Check_dirs(save_path)
            save_name = save_path + f'Sec{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}'

        Save_LC(self.time,self.flux,self.events,id,save_path=save_name)    


    def plot_object(self,cut,objid,event='seperate',save_name=None,save_path=None,
                    latex=True,zoo_mode=True,external_phot=False):
        """
        Plot a source from the cut data.
        """
            

        # -- Use Latex in the plots -- #
        if latex:
            plt.rc('text', usetex=latex)
            plt.rc('font', family='serif')
                                                    #else:
                                                        #plt.rc('text', usetex=False)
                                                        #plt.rc('font', family='sans-serif')

        # -- Gather data -- #
        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        # -- If saving is desired -- #
        if save_path is not None:
            from .tools import _Check_dirs

            if save_path[-1] != '/':
                save_path+='/'
            _Check_dirs(save_path)
            if save_name is None:
                save_name = f'Sec{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}'
            save_path = save_path + save_name

        obj = self.objects[self.objects['objid']==objid].iloc[0]
        obj.lc,obj.cutout = Plot_Object(self.time,self.flux,self.events,objid,event,save_path,latex,zoo_mode) 

        # -- If external photometry is requested, generate the WCS and cutout -- #
        if external_phot:
            from .external_photometry import event_cutout

            print('Getting Photometry...')

            xint = RoundToInt(obj['xcentroid'])
            yint = RoundToInt(obj['ycentroid'])

            RA,DEC = self.wcs.all_pix2world(xint,yint,0)
            # ra_obj,dec_obj = self.wcs.all_pix2world(obj['xcentroid'],obj['ycentroid'],0)
            ra_obj = obj['ra']
            dec_obj = obj['dec']

            #error = (source.e_xccd * 21 /60**2,source.e_yccd * 21/60**2) # convert to deg
            #error = np.nanmax([source.e_xccd,source.e_yccd])
            error = [10 / 60**2,10 / 60**2] # just set error to 10 arcsec. The calculated values are unrealistically small.
            
            fig, wcs, size, photometry,cat = event_cutout((RA,DEC),(ra_obj,dec_obj),error,100)
            axes = fig.get_axes()
            if len(axes) == 1:
                wcs = [wcs]


            theta = np.linspace(0,2*np.pi,10)
            raCircle = 5 * np.cos(theta) + ra_obj
            decCircle = 5 * np.sin(theta) + dec_obj

            xRange = np.arange(xint-3,xint+3)
            yRange = np.arange(yint-3,yint+3)

            lines = []
            for x in xRange:
                line = np.linspace((x,yRange[0]),(x,yRange[-1]),100)
                lines.append(line)

            for y in yRange:
                line = np.linspace((xRange[0],y),(xRange[-1],y),100)
                lines.append(line)

            # -- Plot the TESS pixel edges on the axes -- #
            for i,ax in enumerate(axes): 
                x,y = wcs[i].all_world2pix(raCircle,decCircle,0)
                ax.plot(x,y,'.',markersize=10,color='red',alpha=1,lw=1)

                ys = []
                for j,line in enumerate(lines):
                    if j in [0,6]:
                        color = 'red'
                        lw = 5
                        alpha = 0.7
                    elif j in [5,11]:
                        color = 'cyan'
                        lw = 5
                        alpha = 0.7
                    else:
                        color='white'
                        lw = 2
                        alpha = 0.3

                    ra,dec = self.wcs.all_pix2world(line[:,0]+0.5,line[:,1]+0.5,0)
                    if wcs[i].naxis == 3:
                        x,y,_ = wcs[i].all_world2pix(ra,dec,0,0)
                    else:
                        x,y = wcs[i].all_world2pix(ra,dec,0)
                    if j in [0,5]:
                        ys.append(np.mean(y))
                    good = (x>0)&(y>0)&(x<size)&(y<size)
                    x = x[good]
                    y = y[good]
                    if len(x) > 0:
                        ax.plot(x,y,color=color,alpha=alpha,lw=lw)

            obj.photometry = fig
            obj.cat = cat
            obj.coord = (ra_obj,dec_obj)
        
        return obj
    
    def event_lc(self,cut,objid,eventid=None,frame_buffer=10):

        # -- Gather data -- #
        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut
        
        events = self.events[self.events['objid']==objid]
        if eventid is not None:
            eventid = [eventid]
        else:
            eventid = range(1,len(events)+1)
        
        lcs = []
        for id in eventid:
            e = events[events['eventid']==id].iloc[0]
            x = int(e['xint'])     # x coordinate of the source
            y = int(e['yint'])      # y coordinate of the source
            frameStart = int(e['frame_start'])        # Start frame of the event
            frameEnd = int(e['frame_end'])            # End frame of the event

            frameStart = np.max([frameStart-frame_buffer,0])
            frameEnd = np.min([frameEnd+frame_buffer+1,len(self.time)-1])

            t,f = Generate_LC(self.time,self.flux,x,y,frameStart,frameEnd,buffer=1)
            # t = self.time[frameStart:frameEnd]
            # f = np.nansum(self.flux[frameStart:frameEnd,y-1:y+2,x-1:x+2],axis=(2,1))
            lcs.append((t,f))

        return lcs
    
    def event_frames(self,cut,objid,eventid=None,frame_buffer=5,image_size=11):

        # -- Gather data -- #
        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        events = self.events[self.events['objid']==objid]
        if eventid is None:
            print('No event specificed, using brightest one!')
            eventid = events['lc_sig_max'].argmax()+1
        
        event = events[events.eventid==eventid]
        brightestframe = event['frame_max']
        frames = np.array([brightestframe+frame_buffer*n for n in range(-2,3)])
        frames[frames<0] = 0
        frames[frames>len(self.time)]=len(self.time)-1
        frames = np.unique(frames)

        x = int(event['xint']) 
        y = int(event['yint']) 

        return self.flux[frames,y-image_size//2:y+image_size//2+1,x-image_size//2:x+image_size//2+1]
        


    def locate_transient(self,cut,xcentroid,ycentroid,threshold=3):

        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        return self.events[(self.events['ycentroid'].values < ycentroid+threshold) & (self.events['ycentroid'].values > ycentroid-threshold) & (self.events['xcentroid'].values < xcentroid+threshold) & (self.events['xcentroid'].values > xcentroid-threshold)]

    def full_ccd(self,psflike_lim=0,psfdiff_lim=1,savename=None):

        import matplotlib.patches as patches
        from .dataprocessor import DataProcessor

        p = DataProcessor(sector=self.sector,path=self.data_path)
        lb,_,_,_ = p.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)

        cutCorners, cutCentrePx, cutCentreCoords, cutSize = p.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)

        fig,ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(44,2076)
        ax.set_ylim(0,2048)

        # -- Adds cuts -- #
        #colours = iter(plt.cm.rainbow(np.linspace(0, 1, self.n**2)))
        for corner in cutCorners:
            #c = next(colours)
            c='black'
            rectangle = patches.Rectangle(corner,2*cutSize,2*cutSize,edgecolor=c,
                                            facecolor='none',alpha=1)
            ax.add_patch(rectangle)

        for cut in range(1,17):
            try:
                path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Cut{cut}of{self.n**2}'
                r = pd.read_csv(f'{path}/detected_sources.csv')
                r = r.loc[(r.psfdiff < psfdiff_lim) & (r.psflike > psflike_lim)]
                r['xcentroid'] += lb[cut-1][0]
                r['ycentroid'] += lb[cut-1][1]
                ax.scatter(r['xcentroid'],r['ycentroid'],c=r['frame'],s=5)
                if savename is not None:
                    plt.savefig(savename)
            except:
                pass


def Plot_Object(times,flux,events,id,event,save_path=None,latex=True,zoo_mode=True):
    """
    Plot a source's light curve and image cutout.
    """
    
    import matplotlib.patches as patches
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    
    # -- Use Latex in the plots -- #
    if latex:
        plt.rc('text', usetex=latex)
        plt.rc('font', family='serif')
                                                #else:
                                                    #plt.rc('text', usetex=False)
                                                    #plt.rc('font', family='sans-serif')

    # -- Select sources associated with the object id -- #
    events =  events[events['objid']==id]      # not sources, events, but 
    total_events = int(np.nanmean(events['total_events'].values))   #  Number of events associated with the object id
    
                                                # if type(objectid_bin) == str:
                                                #     if total_events > 5:
                                                #         objectid_bin = True
                                                #     else:
                                                #         objectid_bin = False

    # -- Compile source list based on if plotted source contains all in one -- #
    if type(event) == str:
        if event.lower() == 'seperate':
            pass
        elif event.lower() == 'all':

            # Sets this one "event" to include all the times between first and last detection #
            e = deepcopy(events.iloc[0])
            e['frame_end'] = events['frame_end'].iloc[-1]
            e['mjd_end'] = events['mjd_end'].iloc[-1]
            e['mjd_duration'] = e['mjd_end'] - e['mjd_start']
            e['frame'] = (e['frame_end'] + e['frame_start']) / 2 
            e['mjd'] = (e['mjd_end'] + e['mjd_start']) / 2 

            # Sets the x and y coordinates to the brightest source in the event #
            brightest = np.where(events['lc_sig_max']==np.nanmax(events['lc_sig_max']))[0][0]
            brightest = deepcopy(events.iloc[brightest])
            e['xccd'] = brightest['xccd']
            e['yccd'] = brightest['yccd']
            e['xint'] = brightest['xint']
            e['yint'] = brightest['yint']
            e['xcentroid'] = brightest['xcentroid']
            e['ycentroid'] = brightest['ycentroid']

            events = e.to_frame().T       # "events" in now a single event
            
    elif type(event) == int:
        events = deepcopy(events.iloc[events['eventid'].values == event])
    elif type(event) == list:
        events = deepcopy(events.iloc[events['eventid'].isin(event).values])
    else:
        m = "No valid option selected, input either 'all', 'seperate', an integer event id, or list of integers."
        raise ValueError(m)

    # -- Generates time for plotting and finds breaks in the time series based on the median and standard deviation - #
    time = times - times[0]             
    med = np.nanmedian(np.diff(time))           
    std = np.nanstd(np.diff(time))              
    break_ind = np.where(np.diff(time) > med+1*std)[0]
    break_ind = np.append(break_ind,len(time)) 
    break_ind += 1
    break_ind = np.insert(break_ind,0,0)

    eventtype = event       # just for use down the line as we redefine what event is

    # -- Iterates over each source in the events dataframe and generates plot -- #
    for i in range(len(events)):
        event_id = events['eventid'].iloc[i]          # Select event ID
        event = deepcopy(events.iloc[i])             # Select source 
        x = RoundToInt(event['xcentroid'])      # x coordinate of the source
        y = RoundToInt(event['ycentroid'])      # y coordinate of the source
        frameStart = int(event['frame_start'])        # Start frame of the event
        frameEnd = int(event['frame_end'])            # End frame of the event

        _,f = Generate_LC(times,flux,x,y,buffer=1)
        # f = np.nansum(flux[:,y-1:y+2,x-1:x+2],axis=(2,1))    # Sum the flux in a 3x3 pixel box around the source

        # Find brightest frame in the event #
        if frameEnd - frameStart >= 2:
            brightestframe = frameStart + np.where(abs(f[frameStart:frameEnd]) == np.nanmax(abs(f[frameStart:frameEnd])))[0][0]
        else:
            brightestframe = frameStart
        try:
            brightestframe = int(brightestframe)
        except:
            brightestframe = int(brightestframe[0])
        if brightestframe >= len(flux):   # If the brightest frame is out of bounds, set it to the last frame
            brightestframe -= 1
        if frameEnd >= len(flux):         # If the frame end is out of bounds, set it to the last frame
            frameEnd -= 1

        # Generate light curve around event #
        fstart = frameStart-20
        if fstart < 0:
            fstart = 0
        zoom = f[fstart:frameEnd+20]

        
                                        # if include_periodogram:
                                        #     fig,ax = plt.subplot_mosaic([[1,1,1,2,2],[1,1,1,3,3],[4,4,4,4,4]],figsize=(7,9),constrained_layout=True)
                                        #     fig,ax = plt.subplot_mosaic([[1,1,1,2,2],[1,1,1,3,3],[4,4,4,4,4]],figsize=(7,9),constrained_layout=True)
                                        # else:

        # Create the figure and axes for the plot #
        fig,ax = plt.subplot_mosaic([[1,1,1,2,2],[1,1,1,3,3]],figsize=(7*1.1,5.5*1.1),constrained_layout=True)

        # Invisibly plot event into main panel to extract ylims for zoom inset plot # 
        ax[1].plot(time[fstart:frameEnd+20],zoom,'k',alpha=0)          
        insert_ylims = ax[1].get_ylim()

        # Plot each segment of the light curve in black, with breaks in the time series #
        for i in range(len(break_ind)-1):
            ax[1].plot(time[break_ind[i]:break_ind[i+1]],f[break_ind[i]:break_ind[i+1]],'k',alpha=0.8)
        ylims = ax[1].get_ylim()
        ax[1].set_ylim(ylims[0],ylims[1]+(abs(ylims[0]-ylims[1])))
        ax[1].set_xlim(np.min(time),np.max(time))

        # Differences between Zooniverse mode and normal mode #
        if zoo_mode:
            ax[1].set_title('Is there a transient in the orange region?',fontsize=15)   
            ax[1].set_ylabel('Brightness',fontsize=15,labelpad=10)
            ax[1].set_xlabel('Time (days)',fontsize=15)
            
            axins = ax[1].inset_axes([0.02, 0.55, 0.96, 0.43])      # add inset axes for zoomed in view of the event
            axins.yaxis.set_tick_params(labelleft=False,left=False)
            axins.xaxis.set_tick_params(labelbottom=False,bottom=False)
            ax[1].yaxis.set_tick_params(labelleft=False,left=False)

        else:
            ax[1].set_title(f"{event['TSS Catalogue']}   |   ObjID: {event['objid']}",fontsize=15)   
            ax[1].set_ylabel('Counts (e/s)',fontsize=15,labelpad=10)
            ax[1].set_xlabel(f'Time (MJD - {np.round(times[0],3)})',fontsize=15)

            axins = ax[1].inset_axes([0.1, 0.55, 0.86, 0.43])       # add inset axes for zoomed in view of the event
    

        # Generate a coloured span during the event #
        cadence = np.median(np.diff(time))
        axins.axvspan(time[frameStart]-cadence/2,time[frameEnd]+cadence/2,color='C1',alpha=0.4)

        # Plot full light curve in inset axes #
        for i in range(len(break_ind)-1):
            axins.plot(time[break_ind[i]:break_ind[i+1]],f[break_ind[i]:break_ind[i+1]],'k',alpha=0.8)

        # Change the x and y limits of the inset axes to focus on the event #
        if (frameEnd - frameStart) > 2:
            duration = time[frameEnd] - time[frameStart]
        else:
            duration = 2
        fe = frameEnd + 20
        if fe >= len(time):
            fe = len(time) - 1
        duration = int(event['frame_duration'])
        if duration < 4:
            duration = 4
        xmin = frameStart - 3*duration
        xmax = frameEnd + 3*duration
        if xmin < 0:
            xmin = 0
        if xmax >= len(time):
            xmax = len(time) - 1
        xmin = time[frameStart] - (3*duration * cadence)
        xmax = time[frameEnd] + (3*duration * cadence)
        if xmin <= 0:
            xmin = 0
        if xmax >= np.nanmax(time):
            xmax = np.nanmax(time)
        axins.set_xlim(xmin,xmax)
        axins.set_ylim(insert_ylims[0],insert_ylims[1])

        # Colour the inset axes spines #
        mark_inset(ax[1], axins, loc1=3, loc2=4, fc="none", ec="r",lw=2)
        plt.setp(axins.spines.values(), color='r',lw=2)
        plt.setp([axins.get_xticklines(), axins.get_yticklines()], color='C3')


        # Define max and min brightness for frame plot based on closer 5x5 cutout of brightest frame #
        bright_frame = flux[brightestframe,y-2:y+3,x-2:x+3]   
        vmin = np.percentile(flux[brightestframe],16)
        try:
            vmax = np.percentile(bright_frame,80)
        except:
            vmax = vmin + 20
        if vmin >= vmax:
            vmin = vmax - 5

        # Define and imshow the cutout image (19x19) #
        ymin = y - 9
        if ymin < 0:
            ymin = 0 
        xmin = x -9
        if xmin < 0:
            xmin = 0
        cutout_image = flux[:,ymin:y+10,xmin:x+10]
        ax[2].imshow(cutout_image[brightestframe],cmap='gray',origin='lower',vmin=vmin,vmax=vmax)

        # Add 3x3 rectangle around the centre of the cutout image #
        rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin),5,5, linewidth=3, edgecolor='r', facecolor='none')
        ax[2].add_patch(rect)

        # Add labels, remove axes #
        ax[2].set_title('Brightest image',fontsize=15)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        ax[3].get_xaxis().set_visible(False)
        ax[3].get_yaxis().set_visible(False)
        
        # Find the first frame after the brightest frame that is at least 1 hour later #
        try:
            tdiff = np.where(time-time[brightestframe] >= 1/24)[0][0]
        except:
            tdiff = np.where(time[brightestframe] - time >= 1/24)[0][-1]
        after = tdiff
        if after >= len(cutout_image):
            after = len(cutout_image) - 1 

        # Plot the cutout image 1 hour later #
        ax[3].imshow(cutout_image[after],
                    cmap='gray',origin='lower',vmin=vmin,vmax=vmax)
        rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin),5,5, linewidth=3, edgecolor='r', facecolor='none')
        ax[3].add_patch(rect)
        ax[3].set_title('1 hour later',fontsize=15)
        ax[3].annotate('', xy=(0.2, 1.15), xycoords='axes fraction', xytext=(0.2, 1.), 
                            arrowprops=dict(arrowstyle="<|-", color='r',lw=3))
        ax[3].annotate('', xy=(0.8, 1.15), xycoords='axes fraction', xytext=(0.8, 1.), 
                            arrowprops=dict(arrowstyle="<|-", color='r',lw=3))
        
                                            # if include_periodogram:
                                            #     frequencies = periodogram(period,axis=ax[4])
                                            #     unit = u.electron / u.s
                                            #     light = lk.LightCurve(time=Time(self.time, format='mjd'),flux=(f - np.nanmedian(f))*unit)
                                            #     period = light.to_periodogram()
            
            
        # Save the figure if a save path is provided #
        if save_path is not None:
            save_name = f'{save_path}_object{id}'

                                            # if star_bin: 
                                            #     if source['GaiaID'] > 0:
                                            #         extension = 'star'
                                            #     else:
                                            #         extension = 'no_star'
                                            #     sp += '/' + extension
                                            #splc += '/' + extension

                                            # if period_bin:
                                            #     if type_bin:
                                            #         if source['Prob'] > 0:
                                            #             extension = source['Type']
                                            #         else:
                                            #             extension = self.period_bin(source['peak_freq'],source['peak_power'])
                                            #     if type(extension) != str:
                                            #         extension = 'none'
                                            #     sp += '/' + extension
                                            #     _Check_dirs(sp)
                                            #     #splc += '/' + extension
                                            #     #_Check_dirs(splc)
                                                
                                            # if objectid_bin:
                                            #     extension = f'{self.sector}_{self.cam}_{self.ccd}_{self.cut}_{id}'
                                            #     sp += '/' + extension
                                            #     _Check_dirs(sp)
                                                #splc += '/' + extension
                                                #_Check_dirs(splc)
                                                                        
            if eventtype == 'all':
                plt.savefig(f'{save_name}_all_events.png', bbox_inches = "tight")
            else:
                plt.savefig(f'{save_name}_event{event_id}of{total_events}.png', 
                            bbox_inches = "tight")
                
                                            # if save_lc:
                                            #     headers = ['mjd','counts','event']
                                            #     lc = pd.DataFrame(data=lc,columns=headers)
                                            #     if event == 'all':
                                            #         if self.time_bin is None:
                                            #             lc.to_csv(splc+'/'+savename+'_all_events.csv', index=False)
                                            #         else:
                                            #             lc.to_csv(splc+'/'+savename+f'_all_events_tbin{self.time_bin_name}d.csv', index=False)
                                            #     else:
                                            #         if self.time_bin is None:
                                            #             lc.to_csv(splc+'/'+savename+f'_event{event_id}of{total_events}.csv', index=False)
                                            #         else:
                                            #             lc.to_csv(splc+'/'+savename+f'_event{event_id}of{total_events}_tbin{str(self.time_bin)}d.csv', index=False)
                                            #np.save(save_path+'/'+savename+'_lc.npy',[time,f])
                                            #np.save(save_path+'/'+savename+'_cutout.npy',cutout_image)

    return [times,f], cutout_image

def Save_LC(times,flux,events,id,save_path):
    """
    Save the light curve for a given object id to a csv.
    """

    # -- Get source list for the object id -- #
    sources =  events[events['objid']==id]
    total_events = int(np.nanmean(sources['total_events'].values))

    # -- Initialize arrays for frames, positive and negative flux -- #
    frames = np.zeros_like(times,dtype=int)
    frame_counter = np.arange(len(times))
    pe = np.zeros_like(times,dtype=int)
    ne = np.zeros_like(times,dtype=int)
    for i in range(total_events):
        s = sources.loc[sources['eventid'] == i+1]
        if len(s) > 0:
            frameStart = int(np.nanmean(s['frame_start'].values)) #min(source['frame'].values)
            frameEnd = int(np.nanmean(s['frame_end'].values)) #max(source['frame'].values)
            frames[(frame_counter >= frameStart) & (frame_counter <= frameEnd)] = int(i + 1)
            if s['flux_sign'].values > 0:
                pe[(frame_counter >= frameStart) & (frame_counter <= frameEnd)] = int(s['flux_sign'].values)
            else:
                ne[(frame_counter >= frameStart) & (frame_counter <= frameEnd)] = int(s['flux_sign'].values)
    
    x = int(np.round(np.nanmedian(sources['xcentroid'])))
    y = int(np.round(np.nanmedian(sources['ycentroid'])))

    times,f = Generate_LC(times,flux,x,y,buffer=1)

    # f = np.nansum(flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
    
    lc = np.array([times,f,frames,pe,ne]).T
    headers = ['mjd','counts','event','positive','negative']
    lc = pd.DataFrame(data=lc,columns=headers)
    
    lc.to_csv(f'{save_path}_object{id}_lc.csv',index = False)