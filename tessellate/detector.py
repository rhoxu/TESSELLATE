# -- A good number of functions are imported only in the functions they get utilised -- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# from sourcedetect import SourceDetect, PrfModel
# Now importing this only in the source_detect function

# -- Primary Detection Functions -- #

def _Correlation_check(res,data,prf,corlim=0.8,psfdifflim=0.5,position=True):
    """
    Iterates over sources picked up by StarFinder in parent function.
    Cuts around the coordinates (currently size is 5x5).
    Finds CoM of cut to generate PSF.
    Compares cut with generated PSF, uses np.corrcoef (pearsonr) to judge similarity.
    """

    from scipy.ndimage import center_of_mass

    ind = []
    cors = []
    diff = []
    xcentroids = []
    ycentroids = []
    for _,source in res.iterrows():
        try:
            x = np.round(source['xcentroid']+0.5).astype(int)
            y = np.round(source['ycentroid']+0.5).astype(int)
            cut = deepcopy(data)[y-2:y+3,x-2:x+3]
            cut[cut<0] = 0
            
            if np.nansum(cut) != 0.0:
                cut /= np.nansum(cut)
                cm = center_of_mass(cut)

                if (cm[0]>0) & (cm[0]<5) & (cm[1]>0) & (cm[1]<5):
                    localpsf = prf.locate(cm[1],cm[0],(5,5))
                    if np.nansum(localpsf) != 0:
                        localpsf /= np.nansum(localpsf)

                        if cut.shape == localpsf.shape:
                            r = np.corrcoef(cut.flatten(), localpsf.flatten())
                            r = r[0,1]
                            cors += [r]
                            diff += [np.nansum(abs(cut-localpsf))]
                            xcentroids += [x+cm[1]-2]
                            ycentroids += [y+cm[0]-2]
                        else:
                            cors += [0]
                            diff += [2]
                            xcentroids += [x]
                            ycentroids += [y]
                    else:
                        cors += [0]
                        diff += [2]
                        xcentroids += [x]
                        ycentroids += [y]
                else:
                    cors += [0]
                    diff += [2]
                    xcentroids += [x]
                    ycentroids += [y]
            else:
                cors += [0]
                diff += [2]
                xcentroids += [x]
                ycentroids += [y]
        except:
            cors += [0]
            diff += [2]
            xcentroids += [x]
            ycentroids += [y]

    cors = np.array(cors)
    cors = np.round(cors,2)
    diff = np.array(diff)
    ind = (cors >= corlim) & (diff < psfdifflim)

    if position:
        return ind,cors,diff, ycentroids, xcentroids
    else:
        return ind,cors,diff


def _Spatial_group(result,distance=0.5,njobs=-1):
    """
    Groups events based on proximity.
    """

    from sklearn.cluster import DBSCAN

    pos = np.array([result.xcentroid,result.ycentroid]).T
    cluster = DBSCAN(eps=distance,min_samples=1,n_jobs=njobs).fit(pos)
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

# def _process_detection(star,parallel=False):
#     from scipy.signal import fftconvolve
#     from photutils.aperture import RectangularAnnulus, CircularAperture
#     from astropy.stats import sigma_clip
#     from photutils.aperture import ApertureStats, aperture_photometry

#     pos_ind = ((star.xcentroid.values >=3) & (star.xcentroid.values < data.shape[1]-3) & 
#                 (star.ycentroid.values >=3) & (star.ycentroid.values < data.shape[0]-3))
#     star = star.iloc[ind & pos_ind]

#     x = (star.xcentroid.values + 0.5).astype(int); y = (star.ycentroid.values + 0.5).astype(int)
#     #x = star.xcentroid.values; y = star.ycentroid.values
#     pos = list(zip(x, y))
#     #aperture = RectangularAperture(pos, 3.0, 3.0)
#     aperture = CircularAperture(pos, 1.5)
#     annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
#     m = sigma_clip(data,masked=True,sigma=5).mask
#     mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
#     aperstats_sky = ApertureStats(data, annulus_aperture,mask = mask)
#     annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=40,h_out=40)
#     aperstats_sky_no_mask = ApertureStats(data, annulus_aperture)
#     aperstats_source = ApertureStats(data, aperture)
#     phot_table = aperture_photometry(data, aperture)
#     phot_table = phot_table.to_pandas()
#     bkg_std = aperstats_sky.std
#     bkg_std[bkg_std==0] = aperstats_sky_no_mask.std[bkg_std==0] # assign a value without mask using a larger area of sky
#     negative_ind = aperstats_source.min >= aperstats_sky.median - aperture.area * aperstats_sky.std
#     star['sig'] = phot_table['aperture_sum'].values / (aperture.area * aperstats_sky.std)
#     star['flux'] = phot_table['aperture_sum'].values
#     star['mag'] = -2.5*np.log10(phot_table['aperture_sum'].values)
#     star['bkgstd'] = 9 * aperstats_sky.std
#     star = star.iloc[negative_ind]
#     star = star.loc[(star['sig'] > siglim) & (star['bkgstd'] < bkgstd_lim)]
#     ind, psfcor, psfdiff, ypos ,xpos = _Correlation_check(star,data,prf,corlim=0,psfdifflim=1,position=True)
#     star['ycentroid_com'] = ypos; star['xcentroid_com'] = xpos
#     star['psflike'] = psfcor
#     star['psfdiff'] = psfdiff
#     return star

def _Find_stars(data,prf,fwhmlim=7,siglim=2.5,bkgstd_lim=50,negative=False):

    from scipy.signal import fftconvolve
    from photutils.aperture import RectangularAperture, RectangularAnnulus, ApertureStats, aperture_photometry
    from astropy.stats import sigma_clip

    if negative:
        data = data * -1
    star = _Star_finding_procedure(data,prf,sig_limit=siglim)
    if star is None:
        return None
    ind = (star['fwhm'].values < fwhmlim) & (star['fwhm'].values > 0.8)
    pos_ind = ((star.xcentroid.values >=3) & (star.xcentroid.values < data.shape[1]-3) & 
                (star.ycentroid.values >=3) & (star.ycentroid.values < data.shape[0]-3))
    star = star.iloc[ind & pos_ind]

    x = (star.xcentroid.values + 0.5).astype(int); y = (star.ycentroid.values + 0.5).astype(int)
    pos = list(zip(x, y))
    aperture = RectangularAperture(pos, 3.0, 3.0)
    #aperture = CircularAperture(pos, 1.5)
    annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
    m = sigma_clip(data,masked=True,sigma=5).mask
    mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
    aperstats_sky = ApertureStats(data, annulus_aperture,mask = mask)
    annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=40,h_out=40)
    aperstats_sky_no_mask = ApertureStats(data, annulus_aperture)
    aperstats_source = ApertureStats(data, aperture)
    phot_table = aperture_photometry(data, aperture)
    phot_table = phot_table.to_pandas()

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
    ind, psfcor, psfdiff, ypos ,xpos = _Correlation_check(star,data,prf,corlim=0,psfdifflim=1,position=True)
    star['ycentroid_com'] = ypos; star['xcentroid_com'] = xpos
    star['psflike'] = psfcor
    star['psfdiff'] = psfdiff
    if negative:
        star['flux_sign'] = -1
        star['flux'] = star['flux'].values * -1
        star['max_value'] = star['max_value'].values * -1
    else:
        star['flux_sign'] = 1
    #star['method'] = 'starfind'
    return star


def _Frame_detection(data,prf,corlim,psfdifflim,frameNum):
    """
    Acts on a frame of data. Uses StarFinder to find bright sources, then on each source peform correlation check.
    """
    star = None
    if np.nansum(data) != 0.0:
        #t1 = t()
        #star = _Star_finding_procedure(data,prf)
        p = _Find_stars(data,prf)
        n = _Find_stars(deepcopy(data),prf,negative=True)
        if (p is not None) | (n is not None):
            star = pd.concat([p,n])
            star['frame'] = frameNum
            #t2 = t()   
            #ind, cors,diff = _Correlation_check(star,data,prf,corlim=corlim,psfdifflim=psfdifflim)
            #print(f'Correlation Check: {(t()-t2):.1f} sec - {len(res)} events')
            #star['psflike'] = cors
            #star['psfdiff'] = diff
            #star['flux_sign'] = 1
            #star = star[ind]
        else:
            star = None
    return star

def _Source_mask(res,mask):

    xInts = res['xint'].values
    yInts = res['yint'].values

    #newmask = deepcopy(mask)

    #c1 = newmask.shape[0]//2
    #c2 = newmask.shape[1]//2

    #newmask[c1-2:c1+3,c2-2:c2+3] -= 1
    #newmask[(newmask>3)&(newmask<7)] -= 4
    #maskResult = np.array([newmask[yInts[i],xInts[i]] for i in range(len(xInts))])

    #for id in res['objid'].values:
    #   index = (res['objid'] == id).values
    #   subset = maskResult[index]
    #   maxMask = np.nanmax(subset)
    #   maskResult[index] = maxMask

    res['source_mask'] = mask[yInts,xInts]#maskResult

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

def _Parallel_correlation_check(source,data,prf,corlim,psfdifflim):
    ind, cors,diff, ypos, xpos = _Correlation_check(source,data,prf,corlim=corlim,psfdifflim=psfdifflim)
    return ind,cors,diff,ypos,xpos


def _Do_photometry(star,data,siglim=3,bkgstd_lim=50):

    from scipy.signal import fftconvolve
    from photutils.aperture import RectangularAnnulus, CircularAperture, ApertureStats, aperture_photometry
    from astropy.stats import sigma_clip, SigmaClip

    x = (star.xcentroid.values + 0.5).astype(int); y = (star.ycentroid.values + 0.5).astype(int)
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
    bkg_std = aperstats_sky.std
    bkg_std[bkg_std==0] = aperstats_sky_no_mask.std[bkg_std==0] # assign a value without mask using a larger area of sky
    bkg_std[(~np.isfinite(bkg_std)) | (bkg_std == 0)] = 100
    negative_ind = aperstats_source.min >= aperstats_sky.median - aperture.area * aperstats_sky.std
    star['sig'] = phot_table['aperture_sum'].values / (aperture.area * aperstats_sky.std)
    star['flux'] = phot_table['aperture_sum'].values
    star['mag'] = -2.5*np.log10(phot_table['aperture_sum'].values)
    star['bkgstd'] = aperture.area * bkg_std #* aperstats_sky.std
    
    star = star.loc[(star['sig'] >= siglim) & (star['bkgstd'] <= bkgstd_lim)]
    #star = star.iloc[negative_ind]

    return star 


def _Source_detect(flux,inputNum,prf,corlim,psfdifflim,cpu,siglim=2,bkgstd=50):
    
    from sourcedetect import SourceDetect, PrfModel
    from joblib import Parallel, delayed 

    #model = PrfModel(save_model=False)
    #res = SourceDetect(flux,run=True,train=False,model=model).result
    res = SourceDetect(flux,run=True,train=False).result
    #res = _Spatial_group(res,2)
    frames = res['frame'].unique()
    stars = Parallel(n_jobs=cpu)(delayed(_Do_photometry)(res.loc[res['frame'] == frame],flux[frame],siglim,bkgstd) for frame in frames)
    res = pd.concat(stars)

    ind = (res['xint'].values > 3) & (res['xint'].values < flux.shape[2]-3) & (res['yint'].values >3) & (res['yint'].values < flux.shape[1]-3)
    res = res[ind]
    ind,cors,diff,ypos, xpos = zip(*Parallel(n_jobs=cpu)(delayed(_Parallel_correlation_check)(res.loc[res['frame'] == frame],flux[frame],prf,corlim,psfdifflim) for frame in frames))

    res['psfdiff'] = 0; res['psflike'] = 0
    res['ycentroid_com'] = 0; res['xcentroid_com'] = 0
    for i in range(len(frames)):
        index = np.where(res['frame'].values == frames[i])[0]
        res['psfdiff'].iloc[index] = diff[i]; res['psflike'].iloc[index] = cors[i]
        res['ycentroid_com'].iloc[index] = ypos[i]; res['xcentroid_com'].iloc[index] = xpos[i]
    #res['method'] = 'sourcedetect'
    return res

def _Make_dataframe(results,data):
    frame = None
    for result in results:
        if frame is None:
            frame = result
        else:
            frame = pd.concat([frame,result])
    
    frame['xint'] = deepcopy(np.round(frame['xcentroid'].values)).astype(int)
    frame['yint'] = deepcopy(np.round(frame['ycentroid'].values)).astype(int)
    ind = ((frame['xint'].values >3) & (frame['xint'].values < data.shape[1]-3) & 
           (frame['yint'].values >3) & (frame['yint'].values < data.shape[0]-3))
    frame = frame[ind]
    return frame
    

def _Main_detection(flux,prf,corlim,psfdifflim,inputNum,mode='both'):

    from time import time as t
    import multiprocessing
    from joblib import Parallel, delayed 
    from tqdm import tqdm

    
    print('Starting source detection')
    length = np.linspace(0,flux.shape[0]-1,flux.shape[0]).astype(int)
    if mode == 'starfind':
        results = Parallel(n_jobs=int(multiprocessing.cpu_count()*3/4))(delayed(_Frame_detection)(flux[i],prf,corlim,psfdifflim,inputNum+i) for i in tqdm(length))
        print('found sources')
        results = _Make_dataframe(results,flux[0])
        results['method'] = 'starfind'
    elif mode == 'sourcedetect':
        results = _Source_detect(flux,inputNum,prf,corlim,psfdifflim,int(multiprocessing.cpu_count()*3/4))
        results['method'] = 'sourcedetect'
        results = results[~pd.isna(results['xcentroid'])]
    elif mode == 'both':
        
        results = Parallel(n_jobs=int(multiprocessing.cpu_count()*2/3))(delayed(_Frame_detection)(flux[i],prf,corlim,psfdifflim,inputNum+i) for i in tqdm(length))
        t1 = t()
        star = _Make_dataframe(results,flux[0])
        star['method'] = 'starfind'
        print(f'Done Starfind: {(t()-t1):.1f} sec')
        t1 = t()
        machine = _Source_detect(flux,inputNum,prf,corlim,psfdifflim,int(multiprocessing.cpu_count()*3/4))
        machine = machine[~pd.isna(machine['xcentroid'])]
        machine['method'] = 'sourcedetect'
        print(f'Done Sourcedetect: {(t()-t1):.1f} sec')
        total = [star,machine]
        results = pd.concat(total)


    return results

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
    frame = _Main_detection(flux,prf,corlim,psfdifflim,inputNum,mode=mode)
    print(f'Main Search: {(t()-t1):.1f} sec')

        
    t1 = t()
    print(len(frame))
    frame = _Spatial_group(frame,distance=1)
    frame = frame[~frame.duplicated(subset=['objid', 'frame'], keep='first')]
    print(len(frame))
    print(f'Spatial Group: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _Source_mask(frame,mask)
    print(f'Source Mask: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _Count_detections(frame)
    #frame = frame[frame['n_detections'] > 1]
    print(f'Count Detections: {(t()-t1):.1f} sec')

    return frame

# -- Secondary Functions for looking at periods -- #

def Exp_func(x,a,b,c):
   e = np.exp(a)*np.exp(-x/np.exp(b)) + np.exp(c)
   return e

# -- Secondary Functions (Just for functionality in testing) -- #

# def periodogram(period,plot=True,axis=None):
#     from scipy.signal import find_peaks

#     p = deepcopy(period)

#     norm_p = p.power / np.nanmean(p.power)
#     norm_p[p.frequency.value < 0.05] = np.nan
#     a = find_peaks(norm_p,prominence=3,distance=20,wlen=300)
#     peak_power = p.power[a[0]].value
#     peak_freq = p.frequency[a[0]].value

#     ind = np.argsort(-a[1]['prominences'])
#     peak_power = peak_power[ind]
#     peak_freq = peak_freq[ind]

#     freq_err = np.nanmedian(np.diff(p.frequency.value)) * 3

#     signal_num = np.zeros_like(peak_freq,dtype=int)
#     harmonic = np.zeros_like(peak_freq,dtype=int)
#     counter = 1
#     while (signal_num == 0).any():
#         inds = np.where(signal_num ==0)[0]
#         remaining = peak_freq[inds]
#         r = (np.round(remaining / remaining[0],1)) - remaining // remaining[0]
#         harmonics = r <= freq_err
#         signal_num[inds[harmonics]] = counter
#         harmonic[inds[harmonics]] = (remaining[harmonics] // remaining[0])
#         counter += 1

#     frequencies = {'peak_freq':peak_freq,'peak_power':peak_power,'signal_num':signal_num,'harmonic':harmonic}

#     if plot:
#         if axis is None:
#             fig,ax = plt.subplots()
#         else:
#             ax = axis

#         plt.loglog(p.frequency,p.power,'-')
#         #plt.scatter(peak_freq,peak_power,color='C1')
#         if len(signal_num) > 0:
#             s = max(signal_num)
#             if s > 5:
#                 s = 5
#             for i in range(s):
#                 i += 1
#                 color = f'C{i}'
#                 sig_ind = signal_num == i
#                 for ii in range(max(harmonic[sig_ind])):
#                     ii += 1
#                     hind = harmonic == ii
#                     ind = sig_ind & hind
#                     if ind[0]:
#                         ind
#                     if ii == 1 :
#                         #plt.axvline(peak_freq[ind],label=f'{np.round(1/peak_freq[ind],2)[0]} days',ls='--',color=color)
#                         ax.plot(peak_freq[ind],peak_power[ind],'*',label=f'{np.round(1/peak_freq[ind],2)[0]} days',color=color,ms=10)
#                         #plt.text(peak_freq[ind[hind]],peak_power[ind[hind]],f'{np.round(1/peak_freq[i],2)} days',)
#                     elif ii == 2:
#                         ax.plot(peak_freq[ind],peak_power[ind],'+',color=color,label='harmonics',ms=10)
#                         #plt.axvline(peak_freq[ind],label=f'harmonics',ls='-.',color=color)
#                     else:
#                         ax.plot(peak_freq[ind],peak_power[ind],'+',color=color,ms=10)
#             ax.legend(loc='upper left')
#             ax.set_title('Periodogram')
#             #ax.set_title(f'Peak frequency {np.round(peak_freq[0],2)}'+
#             #            r'$\;$days$^{-1}$' +f' ({np.round(1/peak_freq[0],2)} days)')
#         else:
#             ax.set_title(f'Peak frequency None')
#         ax.set_xlabel(r'Frequency (days$^{-1}$)')
#         ax.set_ylabel(r'Power $(e^-/s)$')

#     return frequencies


def _Check_dirs(save_path):
    """
    Check that all reduction directories are constructed.

    Parameters:
    -----------
    dirlist : list
        List of directories to check for, if they don't exist, they will be created.
    """
    import os
    #for d in dirlist:
    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except:
            pass
    

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
    
    def _check_lc_significance(self,start,end,pos,flux_sign,buffer = 0.5,base_range=1):
        time_per_frame = self.time[1] - self.time[0]
        buffer = int(buffer/time_per_frame)
        base_range = int(base_range/time_per_frame)
        ap = np.zeros_like(self.flux[0])
        y = pos[1]#event.yint.values.astype(int)[0]
        x = pos[0]#event.xint.values.astype(int)[0]
        lc = np.nansum(self.flux[:,y-1:y+2,x-1:x+2],axis=(1,2))
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
        lcevent = lc[start:end]
        lc_sig = (lcevent - med) / std

        if flux_sign >= 0:
            sig_max = np.nanmax(lc_sig)
            sig_med = np.nanmean(lc_sig)
            
        else:
            sig_max = abs(np.nanmin(lc_sig))
            sig_med = abs(np.nanmean(lc_sig))
        
        lc_sig = (lc - med) / std
        return sig_max, sig_med, lc_sig * flux_sign
    
    def _asteroid_checker(self):#,asteroid_distance=3,asteroid_correlation=0.9,asteroid_duration=1):
        from astropy.stats import sigma_clipped_stats
        import cv2

        events = deepcopy(self.events)
        #time = self.time - self.time[0]
        for i in range(len(events)):
            source = events.iloc[i]
            
            frameStart = source['frame_start']
            frameEnd = source['frame_end']
            x = source['xint']; y = source['yint']
            xl = np.max((x - 5,0)); xu = np.min((x + 6,self.flux.shape[2]-1))
            yl = np.max((y - 5,0)); yu = np.min((y + 6,self.flux.shape[1]-1))
            fs = np.max((frameStart - 5, 0))
            fe = np.min((frameEnd + 5, len(self.time)-1))
            
            image = self.flux[fs:fe,yl:yu,xl:xu]
            image = np.nanmax(image,axis=0)
            image = (image / image[(yu-yl)//2,(xu-xl)//2]) * 255
            image[image > 255] = 255
            mean, med, std = sigma_clipped_stats(image,maxiters=10,sigma_upper=2)
            edges = cv2.Canny(image.astype('uint8'), med + 5*std, med + 10*std)
            lines = probabilistic_hough_line(edges, threshold=5, line_length=5, line_gap=1)
            
            # if (frameEnd - frameStart) > 2:
            #   ax[1].axvspan(time[frameStart],time[frameEnd],color='C1',alpha=0.4)
            #     duration = time[frameEnd] - time[frameStart]
            # else:
            # #   ax[1].axvline(time[(frameEnd + frameStart)//2],color='C1')
            #     duration = 0

            # s = self.sources.iloc[self.sources.objid.values == source['objid']]
            # e = s.iloc[(s.frame.values >= frameStart) & (s.frame.values <= frameEnd)]
            # x = e.xcentroid.values
            # y = e.ycentroid.values
            # dist = np.sqrt((x[:,np.newaxis]-x[np.newaxis,:])**2 + (y[:,np.newaxis]-y[np.newaxis,:])**2)
            # dist = np.nanmax(dist,axis=1)
            # dist = np.nanmean(dist)
            # if len(x)>= 2:
            #     cor = np.round(abs(pearsonr(x,y)[0]),1)
            # else:
            #     cor = 0
            # dpass = dist - asteroid_distance
            # cpass = cor - asteroid_correlation
            # asteroid = dpass + cpass > 0
            # asteroid_check = asteroid & (duration < asteroid_duration)
        
            if len(lines) > 0:
                #idx = events['objid']==source['objid']
                events.loc[i, 'Type'] = 'Asteroid'
                events.loc[i, 'Prob'] = 0.5
    
        self.events = events

    def check_classifind(self,source):
        import joblib
        from .temp_classifind import classifind as cf 
        import os

        package_directory = os.path.dirname(os.path.abspath(__file__))

        x = (source['xint']+0.5).astype(int)
        y = (source['yint']+0.5).astype(int)
        f = np.nansum(self.flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
        t = self.time
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
        
    def fit_period(self,source,significance=3):

        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks
        import lightkurve as lk
        from astropy.stats import sigma_clipped_stats
        import astropy.units as u
        from astropy.time import Time

        x = (source['xint']+0.5).astype(int)
        y = (source['yint']+0.5).astype(int)
        
        f = np.nansum(self.flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
        t = self.time
        finite = np.isfinite(f) & np.isfinite(t)
        
        #ap = CircularAperture([source.xcentroid,source.ycentroid],1.5)
        #phot_table = aperture_photometry(data, aperture)
        #phot_table = phot_table.to_pandas()
        unit = u.electron / u.s
        t = self.time
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
    
    def isolate_events(self,objid,frame_buffer=5,duration=2,buffer=1,base_range=1):
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


        # -- Select all sources grouped to this objid -- #
        obj_ind = self.sources['objid'].values == objid
        source = self.sources.iloc[obj_ind]

                        # variable = abs(np.nanmean(source['flux_sign'].values)) <= 0.3

        # -- Initialise -- #
        counter = 1
        events = []
        times = []

        # -- Finds nearest pixels using first source entry then extracts reference image counts 3x3 -- #
        xx = (source['x_source'].values + .5); yy = (source['y_source'].values + 0.5)
        if len(xx) > 1:
            xx = int(xx[0]); yy = int(yy[0])
        else:
            xx = int(xx); yy = int(yy)
        ref_counts = np.nansum(self.ref[yy-1:yy+2,xx-1:xx+2])

        # -- For this objid, separate positive and negative detections -- #
        for sign in source['flux_sign'].unique():
            obj = source.loc[source['flux_sign'] == sign]
            frames = obj.frame.values
            obj['eventID'] = 0

            # -- This whole thing basically finds detections near in time and groups them -- #
            if len(frames) > 1:
                triggers = np.zeros_like(self.time)
                triggers[frames] = 1
                tarr = triggers>0
                temp = np.insert(tarr,0,False)
                temp = np.insert(temp,-1,False)
                detections = np.where(temp)[0]
                if len(detections) > 1:
                    testf = np.diff(np.where(temp)[0])
                    testf = np.insert(testf,-1,0)
                else:
                    testf = np.array([0,0])
                indf = np.where(temp)[0]
                
                min_length = frame_buffer
                testind = (testf<=min_length) & (testf>1)
                if sum(testind) > 0:
                    for j in range(len(indf[testind])):
                        start = indf[testind][j]
                        end = (indf[testind][j] + testf[testind][j])
                        temp[start:end] = True
                temp[0] = False
                temp[-1] = False
                testf = np.diff(np.where(~temp)[0] -1 )
                indf = np.where(~temp)[0] - 1
                testf[testf == 1] = 0
                testf = np.append(testf,0)

                event_time = []
                min_length = duration
                if len(indf[testf>=min_length]) > 0:
                    for j in range(len(indf[testf>min_length])):
                        start = int(indf[testf>min_length][j])
                        if start <0:
                                start = 0
                        end = int(indf[testf>min_length][j] + testf[testf>min_length][j])
                        if end >= len(self.time):
                            end = len(self.time) -1 
                        event_time += [[start,end]]
                event_time = np.array(event_time)
            else:
                try:
                    start = frames-1 
                    if start < 0:
                        start = 0
                    end = frames +1 
                    if end >= len(self.time):
                        end = len(self.time) - 1 
                    event_time = np.array([[start,end]])
                except:
                    start = frames[0]-1 
                    if start < 0:
                        start = 0
                    end = frames[-1] +1 
                    if end >= len(self.time):
                        end = len(self.time) - 1 
                    event_time = np.array([[start,end]])

            # -- Look for any regular variability -- #
            peak_freq, peak_power = self.fit_period(obj.iloc[0])

            # -- Run RFC Classification -- #
            classification, prob = self.check_classifind(obj.iloc[0])
            
            # -- For each event in this objid, create an event dataframe row -- #
            for e in event_time:  # event_time is actually in frames

                # -- Isolate all detections within the frames of this event -- #
                ind = (obj['frame'].values >= e[0]) & (obj['frame'].values <= e[1])
                obj.loc[ind,'eventID'] = counter
                detections = deepcopy(obj.iloc[ind])
                detections = detections.drop(columns='Type')
                triggers = detections['frame'].values       
                
                                    #triggers[triggers>=len(self.time)] = len(self.time) -1 
                                    #av = np.average(detections.values,axis=0,weights=detections['sig'].values)
                
                # -- Find the flux weighted average for things like RA, DEC... of all detections in this event -- #
                event = pandas_weighted_avg(detections)
                event['objid'] = detections['objid'].values[0]
                event['flux_sign'] = sign
                pos = [int(event.xint.values),int(event.yint.values)]

                # -- Calculate significance of detection above background and local light curve -- #
                sig_max, sig_med, sig_lc = self._check_lc_significance(int(e[0]),int(e[1]),pos,event['flux_sign'].values,
                                                                       buffer=buffer,base_range=base_range)
                
                # -- Determine the rest of the parameters to go into the event dataframe row -- #
                min_ind,max_ind,n_detections = self._lightcurve_event_checker(int(e[0]),int(e[1]),sig_lc,triggers,siglim=3)
                indo = [min_ind,max_ind]
                if indo not in times:
                    times += [indo]
                    e[0] = min_ind; e[1] = max_ind
                    prfs = detections['psflike'].values
                    psfdiff = detections['psfdiff'].values
                    event['max_psflike'] = np.nanmax(prfs)
                    event['min_psfdiff'] = np.nanmin(psfdiff)
                    event['flux'] = np.nanmax(detections['flux'].values)
                    event['mag'] = np.nanmin(detections['mag'].values)
                    event['max_sig'] = np.nanmax(detections['sig'].values)
                    event['eventID'] = counter
                    event['frame_start'] = int(e[0])
                    event['frame_end'] = int(e[1])
                    event['duration'] = e[1]-e[0]
                    event['n_detections'] = n_detections#len(detections)
                    event['mjd_start'] = self.time[e[0]]
                    event['mjd_end'] = self.time[e[1]]
                    event['yint'] = event['yint'].values.astype(int)
                    event['xint'] = event['xint'].values.astype(int)
                    event['sector'] = self.sector 
                    event['camera'] = self.cam
                    event['ccd'] = self.ccd
                    event['cut'] = self.cut
                    
                    event['Type'] = obj['Type'].iloc[0]
                    event['peak_freq'] = peak_freq[0]
                    event['peak_power'] = peak_power[0]
                    event['cf_class'] = classification
                    event['cf_prob'] = prob
                    
                    event['lc_sig'] = sig_max
                    event['lc_sig_med'] = sig_med
                    event['ref_counts'] = ref_counts

                    events += [event]
                    counter += 1
                    
        try:
            events = pd.concat(events,ignore_index=True)
        except:
            print(len(events))

        # -- Add to each dataframe row the number of events in the total object -- #
        events['total_events'] = len(events)
        
        return events 

    def _get_all_independent_events(self,frame_buffer=20,duration=1,buffer=0.5,base_range=1,cpu=1):
        from joblib import Parallel, delayed 
        from tqdm import tqdm

        ids = np.unique(self.sources['objid'].values).astype(int)
        if cpu > 1:
            length = np.arange(0,len(ids)).astype(int)
            events = Parallel(n_jobs=cpu)(delayed(self.isolate_events)(ids[i],frame_buffer,duration,buffer,base_range) for i in tqdm(length))
        else:            
            events = []
            for id in ids:
                e = self.isolate_events(id,buffer=buffer,base_range=base_range)
                events += [e]
        events = pd.concat(events,ignore_index=True)
        self.events = events 
    

    
    def _lightcurve_event_checker(self,start,stop,lc_sig,im_triggers,siglim=3):
        #lc_sig = self._check_lc_significance(event,sig_lc=True)
        from .tools import consecutive_points

        sig_ind = np.where(lc_sig>= siglim)[0]
        segments = consecutive_points(sig_ind)
        triggers = np.zeros_like(lc_sig)

        min_ind = int(start)
        max_ind = int(stop)
        triggers[im_triggers] = 1
        triggers[:start] = 0; triggers[end:] = 0
        detections = 0
        for segment in segments:
            if np.sum(triggers[segment]) > 0:
                triggers[segment] = 1
                if np.min(segment) < min_ind:
                    min_ind = np.min(segment)
                if np.max(segment) > max_ind:
                    max_ind = np.max(segment)
                detections += len(segment)
        if max_ind > len(lc_sig):
            max_ind = len(lc_sig)
        if (detections > 0) & (max_ind > min_ind):
            #if detections > event['n_detections'].values:
            #    print('Found more points!!')
            #    print(event['n_detections'].values)
            detections = np.sum(triggers).astype(int)
            #event['frame_start'] = min_ind
            #event['frame_end'] = max_ind
            #event['duration'] = max_ind - min_ind
            #event['mjd_start'] = self.time[min_ind]
            #event['mjd_end'] = self.time[max_ind]
            #event['n_detections'] = detections
            #event['lc_sig'] = np.nanmax(lc_sig[min_ind:max_ind])
            #event['lc_sig_med'] = np.nanmedian(lc_sig[min_ind:max_ind])
            #if detections > event['n_detections'].values:
            #    print('Found more points!!')
            #    print(event['n_detections'])
        else:
            max_ind = stop; min_ind = start
            detections = np.sum(im_triggers).astype(int)
        return min_ind,max_ind,detections#event

        
    
                            # tarr = sig_lc >= siglim
                            # temp = np.insert(tarr,0,False)
                            # temp = np.insert(temp,-1,False)
                            # detections = np.where(temp)[0]
                            # if len(detections) > 1:
                            #     testf = np.diff(np.where(temp)[0])
                            #     testf = np.insert(testf,-1,0)
                            # else:
                            #     testf = np.array([0,0])
                            # indf = np.where(temp)[0]
                            
                            # testf = np.diff(np.where(~temp)[0] -1 )
                            # indf = np.where(~temp)[0] - 1
                            # testf[testf == 1] = 0
                            # testf = np.append(testf,0)


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
            RAs = f'{ra_hms.s:.2f}'

            sign = '+' if dec_dms.d >= 0 else '-'
            DECd = abs(int(dec_dms.d))
            DECm = f'{abs(int(dec_dms.m)):02d}'
            DECs = f'{abs(dec_dms.s):.2f}'

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
        self.cut = cut
        path = f'{self.path}/Cut{self.cut}of{self.n**2}'

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
        
        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=self.cut,n=self.n)

        # if self.events is None:
        #     self.sources['Prob'] = 0; self.sources['Type'] = 0
        #     self.sources['GaiaID'] = 0
        #     self._get_all_independent_events(cpu=int(multiprocessing.cpu_count()))
        # if self.objects is None:    
        #     self._get_objects_df()

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
        results = Detect(self.flux,cam=self.cam,ccd=self.ccd,sector=self.sector,column=column,
                         row=row,mask=self.mask,inputNums=None,mode=mode,datadir=prf_path)
        
        # -- Add wcs, time, ccd info to the results dataframe -- #
        results = self._wcs_time_info(results,self.cut)
        results['xccd'] = deepcopy(results['xcentroid'] + cutCorners[self.cut-1][0])#.astype(int)
        results['yccd'] = deepcopy(results['ycentroid'] + cutCorners[self.cut-1][1])#.astype(int)
        
        # -- For each source, finds the average position based on the weighted average of the flux -- #
        av_var = pandas_weighted_avg(results[['objid','sig','xcentroid','ycentroid','ra','dec','xccd','yccd']])
        av_var = av_var.rename(columns={'xcentroid':'x_source','ycentroid':'y_source','e_xcentroid':'e_x_source',
                                        'e_ycentroid':'e_y_source','ra':'ra_source','dec':'dec_source',
                                        'e_ra':'e_ra_source','e_dec':'e_dec_source','xccd':'xccd_source',
                                        'yccd':'yccd_source','e_xccd':'e_xccd_source',
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
        self._get_all_independent_events(cpu = int(multiprocessing.cpu_count()))
        print(f'Isolate Events: {(t()-t2):.1f} sec')

        # -- Tag asteroids -- #
        self._asteroid_checker()
        
        self.events['objid'] = self.events['objid'].astype(int)

        self._TSS_catalogue_names()

        # -- Save out results to csv files -- #
        if self.time_bin is None:
            self.events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_events.csv',index=False)
        else:
            self.events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/detected_events_tbin{self.time_bin_name}d.csv',index=False)

    def _find_objects(self):

        objids = self.events['objid'].unique()

        columns = [
            'objid', 'sector', 'cam', 'ccd', 'cut', 'xcentroid', 'ycentroid', 
            'ra', 'dec', 'max_lcsig', 'flux_maxsig', 'frame_maxsig',
            'mjd_maxsig','psf_maxsig','flux_sign', 'n_events',
             'min_eventlength', 'max_eventlength','classification',
        ]
        objects = pd.DataFrame(columns=columns)

        for objid in objids:
            obj = self.events[self.events['objid'] == objid]

            maxevent = obj.iloc[obj['lc_sig'].argmax()]

            row_data = {
                'objid': objid,
                'xcentroid': maxevent['xcentroid'],
                'ycentroid': maxevent['ycentroid'],
                'ra': maxevent['ra'],
                'dec': maxevent['dec'],
                'max_lcsig': maxevent['lc_sig'],
                'flux_maxsig': maxevent['flux'],
                'frame_maxsig': maxevent['frame'],
                'mjd_maxsig': maxevent['mjd_start'],
                'psf_maxsig': maxevent['max_psflike'],
                'flux_sign': np.sum(obj['flux_sign'].unique()).astype(int),
                'sector': maxevent['sector'],
                'cam': maxevent['camera'],
                'ccd': maxevent['ccd'],
                'cut': maxevent['cut'],
                'classification': maxevent['Type'],              #['cf_class'],
                                                            # 'classification_prob': maxevent['cf_prob'],
                'n_events': len(obj),
                'min_eventlength': (obj['mjd_end'] - obj['mjd_start']).min(),
                'max_eventlength': (obj['mjd_end'] - obj['mjd_start']).max(),
                'TSS Catalogue' : maxevent['TSS Catalogue']
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

        print('Preloading sources / events')
        # -- Preload self.sources and self.events if they're already made, self.objects can't be made otherwise this function wouldn't be called -- #
        self._gather_results(cut=cut,objects=False)  
        print('\n')

        if time_bin is not None:
            self.time_bin = time_bin

        # -- self.sources contains all individual sources found in all frames -- #
        if self.sources is None:
            print('Source finding (see progress in errors log file)...')
            self._find_sources(mode,prf_path)
            print('\n')

        # -- self.events contains all individual events, grouped by time and space -- #  
        if self.events is None:
            print('Event finding (see progress in errors log file)...')
            self._find_events()
            print('\n')

        print('Object finding...')
        # -- self.objects contains all individual spatial objects -- #  
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
                       sig_lc=None,psf_like=None,
                       min_eventlength=None,max_eventlength=None):
        
        """
        Filter self.objects based on these main things.
        """
        
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        # -- Gather results and data -- #
        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

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
                classification = ['class1', 'class2', 'class3']  # Replace with variable classes
            else:
                classification = [classification_stripped]

            if is_negation:
                objects = objects[~objects['classification'].str.lower().isin(classification)]
            else:
                objects = objects[objects['classification'].str.lower().isin(classification)]

        if flux_sign is not None:
            objects = objects[objects['flux_sign']==flux_sign]
        
        if sig_lc is not None:
            objects = objects[objects['max_lcsig']>=sig_lc]

        if psf_like is not None:
            objects = objects[objects['psf_maxsig']>=psf_like]

        if min_eventlength is not None:
            objects = objects[objects['min_eventlength']>=min_eventlength]
        if max_eventlength is not None:
            objects = objects[objects['max_eventlength']<=max_eventlength]

        return objects

    def filter_events(self,cut,starkiller=False,asteroidkiller=False,
                            lower=None,upper=None,sig_image=None,sig_lc=None,sig_lc_average=None,
                            max_events=None,bkgstd_lim=None,sign=None):
        
        """
        Returns a dataframe of the events in the cut, with options to filter by various parameters.
        """

        # -- Gather results and data -- #
        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        # -- If true, remove events near sources in reduction source mask (ie. stars) -- #
        if starkiller:
            r = self.events.loc[self.events['source_mask']==0]
        else:
            r = self.events

        # -- If true, remove asteroids from the results -- #
        if asteroidkiller:
            r = r.loc[~(r['Type'] == 'Asteroid')]

        # -- Filter by various parameters -- #
        if sig_lc is not None:
            r = r.loc[r['lc_sig']>=sig_lc]
        if sig_lc_average is not None:
            r = r.loc[r['lc_sig_med']>=sig_lc_average]
        if sig_image is not None:
            r = r.loc[r['max_sig'] >= sig_image]
        if max_events is not None:
            r = r.loc[r['total_events'] <= max_events]
        if bkgstd_lim is not None:
            r = r.loc[r['bkgstd'] <= bkgstd_lim]
        if sign is not None:
            r = r.loc[r['flux_sign'] == sign]

        # -- Filter by upper and lower limits on number of detections within each event -- #
        if upper is not None:
            if lower is not None:
                r = r.loc[(r['n_detections'] < upper) & (r['n_detections'] > lower)]
            else:
                r = r.loc[(r['n_detections'] < upper)]
        elif lower is not None:
            r = r.loc[(r['n_detections'] > lower)]

        return r

    # def period_bin(self,frequency,power,power_limit=1):
    #     if np.isfinite(frequency):
    #         p = 1/frequency
    #         if power > power_limit:
    #             if p <= 1/24:
    #                 extension = '1hr_below'
    #             elif (p > 1/24) & (p <= 10/24):
    #                 extension = '1to10hr'
    #             elif (p > 10/24) & (p <= 1):
    #                 extension = '10hrto1day'
    #             elif (p > 1) & (p <= 10):
    #                 extension = '1to10day'
    #             elif (p >10):
    #                 extension = '10day_greater'
    #         else:
    #             extension = 'none'
    #     else:
    #         extension = 'none'
    #     return extension


    def lc_ALL(self,cut,save_path=None,lower=2,max_events=None,starkiller=False,
                 sig_image=3,sig_lc=2.5,bkgstd_lim=100,sign=None):
        """
        Generates all light curves for the detections in the cut to a zip file of csvs.
        """
        
        import multiprocessing
        from joblib import Parallel, delayed 
        import os

        # -- Gather detections to be considered for plotting -- #
        detections = self.filter_events(cut=cut,lower=lower,max_events=max_events,
                                        sign=sign,starkiller=starkiller,sig_lc=sig_lc,sig_image=sig_image)
        
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
                 sig_image=3,sig_lc=2.5,bkgstd_lim=100,sign=1,time_bin=None):
        """
        Generates all source plots for the detections in the cut to a zip file of pngs
        """
        
        import multiprocessing
        from joblib import Parallel, delayed 
        import os

        # if time_bin is not None:
        #     self.time_bin = time_bin

        # -- Gather detections to be considered for plotting -- #
        detections = self.filter_events(cut=cut,lower=lower,max_events=max_events,
                                        sign=sign,starkiller=starkiller,sig_lc=sig_lc,sig_image=sig_image)
        
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
                
        from .external_photometry import event_cutout

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

        if save_path is None:
            save_path = f'{self.path}/Cut{cut}of{self.n**2}/figs/'
            _Check_dirs(save_path)
        
        if save_name is None:
            save_name = f'Sec{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}'

        save_path = save_path + save_name

        obj = self.objects[self.objects['objid']==objid].iloc[0]
        obj.lc,obj.cutout = Plot_Object(self.time,self.flux,self.events,objid,event,save_path,latex,zoo_mode) 

        # -- If external photometry is requested, generate the WCS and cutout -- #
        if external_phot:
            print('Getting Photometry...')

            xint = obj['xcentroid'].astype(int)
            yint = obj['ycentroid'].astype(int)

            RA,DEC = self.wcs.all_pix2world(xint,yint,0)
            ra_obj,dec_obj = self.wcs.all_pix2world(obj['xcentroid'],obj['ycentroid'],0)

            #error = (source.e_xccd * 21 /60**2,source.e_yccd * 21/60**2) # convert to deg
            #error = np.nanmax([source.e_xccd,source.e_yccd])
            error = [10 / 60**2,10 / 60**2] # just set error to 10 arcsec. The calculated values are unrealistically small.
            
            fig, wcs, size, photometry,cat = event_cutout((RA,DEC),(ra_obj,dec_obj),error,100)
            axes = fig.get_axes()
            if len(axes) == 1:
                wcs = [wcs]

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
    sources =  events[events['objid']==id]
    total_events = int(np.nanmean(sources['total_events'].values))   #  Number of events associated with the object id
    
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
            e = deepcopy(sources.iloc[0])
            e['frame_end'] = sources['frame_end'].iloc[-1]
            e['mjd_end'] = sources['mjd_end'].iloc[-1]
            e['mjd_duration'] = e['mjd_end'] - e['mjd_start']
            e['frame'] = (e['frame_end'] + e['frame_start']) / 2 
            e['mjd'] = (e['mjd_end'] + e['mjd_start']) / 2 

            # Sets the x and y coordinates to the brightest source in the event #
            brightest = np.where(sources['lc_sig']==np.nanmax(sources['lc_sig']))[0][0]
            brightest = deepcopy(sources.iloc[brightest])
            e['xccd'] = brightest['xccd']
            e['yccd'] = brightest['yccd']
            e['xint'] = brightest['xint']
            e['yint'] = brightest['yint']
            e['xcentroid'] = brightest['xcentroid']
            e['ycentroid'] = brightest['ycentroid']

            sources = e.to_frame().T       # "sources" in now a single event
            
    elif type(event) == int:
        sources = deepcopy(sources.iloc[sources['eventID'].values == event])
    elif type(event) == list:
        sources = deepcopy(sources.iloc[sources['eventID'].isin(event).values])
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

    # -- Iterates over each source in the sources dataframe and generates plot -- #
    for i in range(len(sources)):
        event_id = sources['eventID'].iloc[i]          # Select event ID
        source = deepcopy(sources.iloc[i])             # Select source 
        x = (source['xcentroid']+0.5).astype(int)      # x coordinate of the source
        y = (source['ycentroid']+0.5).astype(int)      # y coordinate of the source
        frameStart = int(source['frame_start'])        # Start frame of the event
        frameEnd = int(source['frame_end'])            # End frame of the event

        f = np.nansum(flux[:,y-1:y+2,x-1:x+2],axis=(2,1))    # Sum the flux in a 3x3 pixel box around the source

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
            ax[1].set_title('Lightcurve',fontsize=15)   
            ax[1].set_ylabel('Counts (e/s)',fontsize=15,labelpad=10)
            ax[1].set_xlabel(f'Time (MJD - {np.round(times[0],3)})',fontsize=15)

            axins = ax[1].inset_axes([0.1, 0.55, 0.86, 0.43])       # add inset axes for zoomed in view of the event
    

        # Generate a coloured span during the event #
        axins.axvspan(time[frameStart],time[frameEnd],color='C1',alpha=0.4)

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
        duration = int(source['duration'])
        if duration < 4:
            duration = 4
        xmin = frameStart - 3*duration
        xmax = frameEnd + 3*duration
        if xmin < 0:
            xmin = 0
        if xmax >= len(time):
            xmax = len(time) - 1
        cadence = np.mean(np.diff(time))
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
                                                                        
            if event == 'all':
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
        s = sources.loc[sources['eventID'] == i+1]
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

    f = np.nansum(flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
    
    lc = np.array([times,f,frames,pe,ne]).T
    headers = ['mjd','counts','event','positive','negative']
    lc = pd.DataFrame(data=lc,columns=headers)
    
    lc.to_csv(f'{save_path}_object{id}_lc.csv',index = False)