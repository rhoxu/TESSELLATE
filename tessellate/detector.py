import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
from PRF import TESS_PRF
from copy import deepcopy
from photutils.detection import StarFinder
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import center_of_mass
from sklearn.cluster import DBSCAN


import multiprocessing
from joblib import Parallel, delayed 
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from tqdm import tqdm
from time import time as t
import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.ndimage import convolve

from sourcedetect import SourceDetect

from .dataprocessor import DataProcessor
from .catalog_queries import find_variables, gaia_stars, match_result_to_cat

# -- Primary Detection Functions -- #

def _correlation_check(res,data,prf,corlim=0.8,psfdifflim=0.5):
    """
    Iterates over sources picked up by StarFinder in parent function.
    Cuts around the coordinates (currently size is 5x5).
    Finds CoM of cut to generate PSF.
    Compares cut with generated PSF, uses np.corrcoef (pearsonr) to judge similarity.
    """

    cors = []
    diff = []
    xcentroids = []
    ycentroids = []
    for _,source in res.iterrows():
        try:
            x = np.round(source['xcentroid']).astype(int)
            y = np.round(source['ycentroid']).astype(int)
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

    return ind,cors,diff

def _spatial_group(result,distance=0.5,njobs=-1):
    """
    Groups events based on proximity.
    """

    pos = np.array([result.xcentroid,result.ycentroid]).T
    cluster = DBSCAN(eps=distance,min_samples=1,n_jobs=njobs).fit(pos)
    labels = cluster.labels_
    unique_labels = set(labels)
    for label in unique_labels:
        result.loc[label == labels,'objid'] = label + 1
    result['objid'] = result['objid'].astype(int)
    return result

def _star_finding_procedure(data,prf,sig_limit = 2):

    mean, med, std = sigma_clipped_stats(data, sigma=5.0)

    psfCentre = prf.locate(5,5,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfCentre)
    res1 = finder.find_stars(deepcopy(data))

    psfUR = prf.locate(5.25,5.25,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfUR)
    res2 = None#finder.find_stars(deepcopy(data))

    psfUL = prf.locate(4.75,5.25,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfUL)
    res3 = None#finder.find_stars(deepcopy(data))

    psfDR = prf.locate(5.25,4.75,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfDR)
    res4 = None#finder.find_stars(deepcopy(data))

    psfDL = prf.locate(4.75,4.75,(11,11))
    finder = StarFinder(med + sig_limit*std,kernel=psfDL)
    res5 = None#finder.find_stars(deepcopy(data))

    tables = [res1, res2, res3, res4, res5]
    good_tables = [table.to_pandas() for table in tables if table is not None]
    if len(good_tables)>0:
        total = pd.concat(good_tables)
        total = total[~pd.isna(total['xcentroid'])]
        if len(total) > 0:
            grouped = _spatial_group(total,distance=2)
            res = grouped.groupby('objid').head(1)
            res = res.reset_index(drop=True)
            res = res.drop(['id','objid'],axis=1)
        else:
            res=None
    else:
        res = None

    return res

def _frame_detection(data,prf,corlim,psfdifflim,frameNum):
    """
    Acts on a frame of data. Uses StarFinder to find bright sources, then on each source peform correlation check.
    """
    star = None
    if np.nansum(data) != 0.0:
        t1 = t()
        star = _star_finding_procedure(data,prf)

        if star is not None:
            star['frame'] = frameNum
            t2 = t()    
            ind, cors,diff = _correlation_check(star,data,prf,corlim=corlim,psfdifflim=psfdifflim)
            #print(f'Correlation Check: {(t()-t2):.1f} sec - {len(res)} events')
            star['psflike'] = cors
            star['psfdiff'] = diff
            star['flux_sign'] = 1
            star = star[ind]
    return star

def _source_mask(res,mask):

    xInts = res['xint'].values
    yInts = res['yint'].values

    newmask = deepcopy(mask)

    c1 = newmask.shape[0]//2
    c2 = newmask.shape[1]//2

    newmask[c1-2:c1+3,c2-2:c2+3] -= 1
    newmask[(newmask>3)&(newmask<7)] -= 4
    maskResult = np.array([newmask[yInts[i],xInts[i]] for i in range(len(xInts))])

    for id in res['objid'].values:
        index = (res['objid'] == id).values
        subset = maskResult[index]
        maxMask = np.nanmax(subset)
        maskResult[index] = maxMask

    res['source_mask'] = maskResult

    return res

def _count_detections(result):

    ids = result['objid'].values
    unique = np.unique(ids, return_counts=True)
    unique = list(zip(unique[0],unique[1]))

    array = np.zeros_like(ids)

    for id,count in unique:
        index = (result['objid'] == id).values
        array[index] = count

    result['n_detections'] = array

    return result

def _parallel_correlation_check(source,data,prf,corlim,psfdifflim):
    ind, cors,diff = _correlation_check(source,data,prf,corlim=corlim,psfdifflim=psfdifflim)
    return ind,cors,diff

def _source_detect(flux,inputNum,prf,corlim,psfdifflim,cpu):
    res = SourceDetect(flux,run=True,train=False).result
    res['psfdiff'] = 0
    res['xint'] = deepcopy(np.round(res['xcentroid'].values)).astype(int)
    res['yint'] = deepcopy(np.round(res['ycentroid'].values)).astype(int)
    ind = (res['xint'].values >3) & (res['xint'].values < flux.shape[2]-3) & (res['yint'].values >3) & (res['yint'].values < flux.shape[1]-3)
    res = res[ind]
    frames = res['frame'].unique()
    ind,cors,diff = zip(*Parallel(n_jobs=cpu)(delayed(_parallel_correlation_check)(res.loc[res['frame'] == frame],flux[frame],prf,corlim,psfdifflim) for frame in frames))
    res['good'] = False
    for i in range(len(frames)):
        index = np.where(res['frame'].values == frames[i])[0]
        res['good'].iloc[index[ind[i]]] = True
        res['psfdiff'].iloc[index] = diff[i]
        res['psflike'].iloc[index] = cors[i]
    res = res.loc[res['good']]
    res = res.drop(columns='good')
    #length = np.linspace(0,flux.shape[0]-1,flux.shape[0]).astype(int)
    #for i in tqdm(length):
    #    if np.nansum(flux[i]) != 0.0:
    #        obj = SourceDetect(flux[i],run=True,train=False).result
    #        obj['frame'] = inputNum + i
    #        obj['psfdiff'] = 0
    #        res += [obj]
    #res = _make_dataframe(res,flux[0])
    return res

def _make_dataframe(results,data):
    frame = None
    for result in results:
        if frame is None:
            frame = result
        else:
            frame = pd.concat([frame,result])
    
    frame['xint'] = deepcopy(np.round(frame['xcentroid'].values)).astype(int)
    frame['yint'] = deepcopy(np.round(frame['ycentroid'].values)).astype(int)
    ind = (frame['xint'].values >3) & (frame['xint'].values < data.shape[1]-3) & (frame['yint'].values >3) & (frame['yint'].values < data.shape[0]-3)
    frame = frame[ind]
    return frame
    

def _main_detection(flux,prf,corlim,psfdifflim,inputNum,mode='both'):

    length = np.linspace(0,flux.shape[0]-1,flux.shape[0]).astype(int)
    if mode == 'starfind':
        results = Parallel(n_jobs=int(multiprocessing.cpu_count()/2))(delayed(_frame_detection)(flux[i],prf,corlim,psfdifflim,inputNum+i) for i in tqdm(length))
        results = _make_dataframe(results,flux[0])
    elif mode == 'sourcedetect':
        results = _source_detect(flux,inputNum,prf,corlim,psfdifflim,int(multiprocessing.cpu_count()/2))
    elif mode == 'both':
        results = Parallel(n_jobs=int(multiprocessing.cpu_count()/2))(delayed(_frame_detection)(flux[i],prf,corlim,psfdifflim,inputNum+i) for i in tqdm(length))
        star = _make_dataframe(results,flux[0])
        machine = _source_detect(flux,inputNum,prf,corlim,psfdifflim,int(multiprocessing.cpu_count()/2))
        total = [star,machine]
        total = pd.concat(total)
        total = total[~pd.isna(total['xcentroid'])]
        grouped = _spatial_group(total,distance=2)
        drop = []
        for obj in grouped['objid'].unique():
            sub = grouped[grouped['objid'] == obj]
            for frame in sub['frame'].unique():
                s2 = sub[sub['frame'] == frame]
                if len(s2) == 2:
                    drop += [s2.index[1]]
        results = grouped.drop(drop)

    return results

def detect(flux,cam,ccd,sector,column,row,mask,inputNums=None,corlim=0.6,psfdifflim=0.7,mode='both'):
    """
    Main Function.
    """

    if inputNums is not None:
        flux = flux[inputNums]
        inputNum = inputNums[-1]
    else:
        inputNum = 0
            
    if sector < 4:
        prf = TESS_PRF(cam,ccd,sector,column,row,localdatadir='/fred/oz335/_local_TESS_PRFs/Sectors1_2_3')
    else:
        prf = TESS_PRF(cam,ccd,sector,column,row,localdatadir='/fred/oz335/_local_TESS_PRFs/Sectors4+')

    t1 = t()
    frame = _main_detection(flux,prf,corlim,psfdifflim,inputNum,mode=mode)
    print(f'Main Correlation: {(t()-t1):.1f} sec')

    # if len(frame) > 25_000:
    #     print(len(frame))
    #     print('Increasing Correlation Limit to 0.9')
    #     del(frame)
    #     frame = _main_correlation(flux,prf,0.9,psfdifflim,inputNum)
    #     print(f'Main Correlation: {(t()-t1):.1f} sec')
    #     print(len(frame))
    #     if len(frame) > 25_000:
    #         print('Reducing PSF Difference Limit to 0.4')
    #         del(frame)
    #         frame = _main_correlation(flux,prf,0.9,0.4,inputNum)
    #         print(f'Main Correlation: {(t()-t1):.1f} sec')
    #         print(len(frame))
        
    t1 = t()
    frame = _spatial_group(frame,distance=1.5)
    print(f'Spatial Group: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _source_mask(frame,mask)
    print(f'Source Mask: {(t()-t1):.1f} sec')

    t1 = t()
    frame = _count_detections(frame)
    print(f'Count Detections: {(t()-t1):.1f} sec')

    return frame

# -- Secondary Functions for looking at periods -- #

def exp_func(x,a,b,c):
   e = np.exp(a)*np.exp(-x/np.exp(b)) + np.exp(c)
   return e

# -- Secondary Functions (Just for functionality in testing) -- #

def periodogram(period,plot=True,axis=None):
    p = deepcopy(period)

    norm_p = p.power / np.nanmean(p.power)
    norm_p[p.frequency.value < 0.05] = np.nan
    a = find_peaks(norm_p,prominence=3,distance=20,wlen=300)
    peak_power = p.power[a[0]].value
    peak_freq = p.frequency[a[0]].value

    ind = np.argsort(-a[1]['prominences'])
    peak_power = peak_power[ind]
    peak_freq = peak_freq[ind]

    freq_err = np.nanmedian(np.diff(p.frequency.value)) * 3

    signal_num = np.zeros_like(peak_freq,dtype=int)
    harmonic = np.zeros_like(peak_freq,dtype=int)
    counter = 1
    while (signal_num == 0).any():
        inds = np.where(signal_num ==0)[0]
        remaining = peak_freq[inds]
        r = (np.round(remaining / remaining[0],1)) - remaining // remaining[0]
        harmonics = r <= freq_err
        signal_num[inds[harmonics]] = counter
        harmonic[inds[harmonics]] = (remaining[harmonics] // remaining[0])
        counter += 1

    frequencies = {'peak_freq':peak_freq,'peak_power':peak_power,'signal_num':signal_num,'harmonic':harmonic}

    if plot:
        if axis is None:
            fig,ax = plt.subplots()
        else:
            ax = axis

        plt.loglog(p.frequency,p.power,'-')
        #plt.scatter(peak_freq,peak_power,color='C1')
        if len(signal_num) > 0:
            s = max(signal_num)
            if s > 5:
                s = 5
            for i in range(s):
                i += 1
                color = f'C{i}'
                sig_ind = signal_num == i
                for ii in range(max(harmonic[sig_ind])):
                    ii += 1
                    hind = harmonic == ii
                    ind = sig_ind & hind
                    if ind[0]:
                        ind
                    if ii == 1 :
                        #plt.axvline(peak_freq[ind],label=f'{np.round(1/peak_freq[ind],2)[0]} days',ls='--',color=color)
                        ax.plot(peak_freq[ind],peak_power[ind],'*',label=f'{np.round(1/peak_freq[ind],2)[0]} days',color=color,ms=10)
                        #plt.text(peak_freq[ind[hind]],peak_power[ind[hind]],f'{np.round(1/peak_freq[i],2)} days',)
                    elif ii == 2:
                        ax.plot(peak_freq[ind],peak_power[ind],'+',color=color,label='harmonics',ms=10)
                        #plt.axvline(peak_freq[ind],label=f'harmonics',ls='-.',color=color)
                    else:
                        ax.plot(peak_freq[ind],peak_power[ind],'+',color=color,ms=10)
            ax.legend(loc='upper left')
            ax.set_title('Periodogram')
            #ax.set_title(f'Peak frequency {np.round(peak_freq[0],2)}'+
            #                r'$\;$days$^{-1}$' +f' ({np.round(1/peak_freq[0],2)} days)')
        else:
            ax.set_title(f'Peak frequency None')
        ax.set_xlabel(r'Frequency (days$^{-1}$)')
        ax.set_ylabel(r'Power $(e^-/s)$')

    return frequencies



class Detector():

    def __init__(self,sector,cam,ccd,data_path,n,match_variables=True,mode='both'):

        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.data_path = data_path
        self.n = n
        self.match_variables = match_variables

        self.flux = None
        self.time = None
        self.mask = None
        self.sources = None  #raw detection results
        self.events = None   #temporally located with same object id
        self.cut = None
        self.bkg = None

        self.mode = mode

        self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}'

    def _wcs_time_info(self,result,cut):
        
        cut_path = f'{self.path}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}.fits'
        times = np.load(f'{self.path}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}_Times.npy')

        tpf = lk.TessTargetPixelFile(cut_path)
        self.wcs = tpf.wcs
        coords = tpf.wcs.all_pix2world(result['xcentroid'],result['ycentroid'],0)
        result['ra'] = coords[0]
        result['dec'] = coords[1]
        result['mjd'] = times[result['frame']]

        return result
    
    def _gather_data(self,cut):
        base = f'{self.path}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}'
        self.base_name = base
        self.flux = np.load(base + '_ReducedFlux.npy')
        try:
            self.bkg = np.load(base + '_Background.npy')
        except:
            pass
        self.mask = np.load(base + '_Mask.npy')
        self.time = np.load(base + '_Times.npy')
        try:
            self.wcs = WCS(f'{self.path}/Cut{cut}of{self.n**2}/wcs.fits')
        except:
            print('Could not find a wcs file')
    
    def _check_lc_significance(self,event,buffer = 10,base_range=20):
        ap = np.zeros_like(self.flux[0])
        y = event.yint.values.astype(int)[0]
        x = event.xint.values.astype(int)[0]
        lc = np.nansum(self.flux[:,y-1:y+2,x-1:x+2],axis=(1,2))
        fs = event['frame_start'].values[0] - buffer
        fe = event['frame_end'].values[0] + buffer
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
        mean,med, std = sigma_clipped_stats(lc[ind])
        if event['flux_sign'].values >= 0:
            lc_max = np.nanmax(lc[event['frame_start'].values[0]:event['frame_end'].values[0]])
        else:
            lc_max = np.nanmin(lc[event['frame_start'].values[0]:event['frame_end'].values[0]])
        significance = (lc_max - med) / std
        return significance 
    
    def _asteroid_checker(self):

        events = deepcopy(self.events)

        time = self.time - self.time[0]

        for source in self.events.iterrows():
            source = source[1]
            
            frameStart = source['frame_start']
            frameEnd = source['frame_end']

            if (frameEnd - frameStart) > 2:
            #    ax[1].axvspan(time[frameStart],time[frameEnd],color='C1',alpha=0.4)
                duration = time[frameEnd] - time[frameStart]
            else:
            #    ax[1].axvline(time[(frameEnd + frameStart)//2],color='C1')
                duration = 0

            s = self.sources.iloc[self.sources.objid.values == source['objid']]
            e = s.iloc[(s.frame.values >= frameStart) & (s.frame.values <= frameEnd)]
            x = e.xcentroid.values
            y = e.ycentroid.values
            dist = np.sqrt((x[:,np.newaxis]-x[np.newaxis,:])**2 + (y[:,np.newaxis]-y[np.newaxis,:])**2)
            dist = np.nanmax(dist,axis=1)
            dist = np.nanmean(dist)
            if len(x)>= 2:
                cor = np.round(abs(pearsonr(x,y)[0]),1)
            else:
                cor = 0
            dpass = dist - 0.8
            cpass = cor - 0.8
            asteroid = dpass + cpass > 0 
            asteroid_check = asteroid & (duration < 1)
        
            if asteroid_check:
                idx = events['objid']==source['objid']
                events.loc[idx, 'Type'] = 'Asteroid'
                events.loc[idx, 'Prob'] = 1.0
    
        self.events = events
        
    def fit_period(self,source,significance=3):
        x = (source['xint']+0.5).astype(int)
        y = (source['yint']+0.5).astype(int)

        f = np.nansum(self.flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
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
                popt, pcov = curve_fit(exp_func, x[ind], y[ind])
                fit = exp_func(x, *popt)
                m,med,std = sigma_clipped_stats(y - fit)
                ind = (y - fit) < (5 * std + med)

            norm = y/exp_func(x, *popt)
            a = find_peaks(norm,prominence=3,distance=50,wlen=300,height=significance)
            peak_power = y[a[0]]
            peak_freq = x[a[0]]
            peak_freq = peak_freq[peak_power>1] 
            peak_power = peak_power[peak_power>1] 
            if peak_power is None:
                peak_power = [np.nan]
                peak_freq = [np.nan]
            elif len(peak_power) < 1:
                peak_power = [np.nan]
                peak_freq = [np.nan]
        except:
            peak_power = [np.nan]
            peak_freq = [np.nan]
        return peak_freq, peak_power
    
    def isolate_events(self,objid,frame_buffer=20,duration=1,
                       asteroid_distance=2,asteroid_correlation=0.8,asteroid_duration=1):
        obj_ind = self.sources['objid'].values == objid
        obj = self.sources.iloc[obj_ind]
        variable = abs(np.nanmean(obj['flux_sign'].values)) <= 0.7
        frames = obj.frame.values
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
                    start = indf[testf>min_length][j]
                    if start <0:
                            start = 0
                    end = indf[testf>min_length][j] + testf[testf>min_length][j]
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
        events = []
        peak_freq, peak_power = self.fit_period(obj.iloc[0])
        counter = 1
        obj['eventID'] = 0
        for e in event_time:
            ind = (obj['frame'].values >= e[0]) & (obj['frame'].values <= e[1])
            obj.loc[ind,'eventID'] = counter
            detections = deepcopy(obj.iloc[ind])
            detections = detections.drop(columns='Type')
            # check for asteroid
            x = detections.xcentroid.values; y = detections.ycentroid.values
            #dist = np.sqrt((x[:,np.newaxis]-x[np.newaxis,:])**2 + (y[:,np.newaxis]-y[np.newaxis,:])**2)
            #dist = np.nanmax(dist,axis=1)
            #av_dist = np.nanmean(dist)
            av_dist = np.sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2)
            if len(x)>= 2:
                cor = np.round(abs(pearsonr(x,y)[0]),1)
            else:
                cor = 0
            asteroid = (av_dist - asteroid_distance) + (cor - asteroid_correlation) > 0 #(cor >= 0.8) & (dist >= 0.85)
            
            event = deepcopy(detections.mean().to_frame().T)
            event['eventID'] = counter
            event['frame_start'] = e[0]
            event['frame_end'] = e[1]
            event['n_detections'] = e[1]-e[0]
            event['mjd_start'] = self.time[e[0]]
            event['mjd_end'] = self.time[e[1]]
            event['yint'] = event['yint'].values.astype(int)
            event['xint'] = event['xint'].values.astype(int)
            event['yccd'] = event['yccd'].values.astype(int)
            event['xccd'] = event['xccd'].values.astype(int)
            duration = self.time[e[1]] - self.time[e[0]]
            event['mjd_duration'] = duration
            if asteroid & (duration < asteroid_duration):
                obj.loc[ind,'Type'] = 'Asteroid'
                event['Prob'] = 1
            event['Type'] = obj['Type'].iloc[0]
            event['peak_freq'] = peak_freq[0]
            event['peak_power'] = peak_power[0]
            event['variable'] = variable | np.isfinite(peak_power[0])
            sig = self._check_lc_significance(event)
            event['lc_sig'] = sig
            
            events += [event]
            counter += 1
            
        events = pd.concat(events,ignore_index=True)
        events['total_events'] = len(events)
        
        try:
            events = events.drop('Unnamed: 0',axis=1)
        except:
            pass
        return events 

    def _get_all_independent_events(self,frame_buffer=20):
        ids = np.unique(self.sources['objid'].values).astype(int)
        events = []
        for id in ids:
            e = self.isolate_events(id,frame_buffer=frame_buffer)
            events += [e]
        events = pd.concat(events,ignore_index=True)
        self.events = events 

    def _gather_results(self,cut):
        path = f'{self.path}/Cut{cut}of{self.n**2}'
        self.sources = pd.read_csv(f'{path}/detected_sources.csv')
        try:
            self.events = pd.read_csv(f'{path}/detected_events.csv')
        except:
            print('No detected events file found')
        try:
            self.wcs = f'{path}/wcs.fits'
        except:
            print('No wcs found')
            self.wcs = None
        '''try:
            self.gaia = pd.read_csv(f'{path}/local_gaia_cat.csv')
            query_gaia = False
        except:
            print('No local Gaia cat, will try to query')

        try:
            self.variables = pd.read_csv(f'{path}/variable_catalog.csv')
        except:
            print('No local variable catalog, will try to query.')
            query_var = False
        self.result['Prob'] = 0; self.result['Type'] = 0
        self.result['GaiaID'] = 0
        if ~query_gaia:
            self.result = match_result_to_cat(deepcopy(self.result),self.gaia,columns=['Source'])
            self.result = self.result.rename(columns={'Source': 'GaiaID'})
        if ~query_var:
            self.result = match_result_to_cat(deepcopy(self.result),self.variables,columns=['Type','Prob'])

        if query_gaia | query_var:
            try:
                ids = np.unique(self.result['objid'].values)
                ra = []; dec = []
                Id = []
                for id in ids:
                    if len(self.result.iloc[self.result['objid'].values == id]) >=2:
                        Id += [id]
                        ra += [self.result.loc[self.result['objid'] == id, 'ra'].mean()]
                        dec += [self.result.loc[self.result['objid'] == id, 'dec'].mean()]
                pos = {'objid':Id,'ra':ra,'dec':dec}
                pos = pd.DataFrame(pos)
                center = [pos.loc[:,'ra'].mean(),pos.loc[:,'dec'].mean()]
                rad = np.max(np.sqrt((pos['ra'].values-center[0])**2 +(pos['dec'].values-center[1])**2)) + 1/60
                var_cat = find_variables(center,pos,rad,rad)
                ind = np.where(var_cat['Prob'].values > 0)[0]
                for i in ind:
                    self.result.loc[self.result['objid'] == var_cat['objid'].iloc[i], 'Type'] = var_cat['Type'].iloc[i]
                    self.result.loc[self.result['objid'] == var_cat['objid'].iloc[i], 'Prob'] = var_cat['Prob'].iloc[i]
                
                stars = gaia_stars(pos)
                ind = np.where(stars['GaiaID'].values > 0)[0]
                for i in ind:
                    self.result.loc[self.result['objid'] == stars['objid'].iloc[i], 'GaiaID'] = var_cat['GaiaID'].iloc[i]

            except:
                print('Could not query variable catalogs')
        '''
        if self.events is None:
            self.sources['Prob'] = 0; self.sources['Type'] = 0
            self.sources['GaiaID'] = 0
            self._get_all_independent_events()


    def event_coords(self,objid):
        self.obj_ra = self.events.loc[self.events['objid'] == objid, 'ra'].mean()
        self.obj_dec = self.events.loc[self.events['objid'] == objid, 'dec'].mean()

    def source_detect(self,cut,mode=None):
        from glob import glob 
        from astropy.io import fits 
        if mode is None:
            mode = self.mode
        if (mode == 'both') | (mode == 'starfind') | (mode == 'sourcedetect'):
            pass
        else:
            m = 'Mode must be one of the following: both, starfind, sourcedetect.'
            raise ValueError(m)
        
        if cut != self.cut:
            self._gather_data(cut)
            self.cut = cut

        processor = DataProcessor(sector=self.sector,path=self.data_path,verbose=2)
        cutCorners, cutCentrePx, _, _ = processor.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)

        column = cutCentrePx[cut-1][0]
        row = cutCentrePx[cut-1][1]

        results = detect(self.flux,cam=self.cam,ccd=self.ccd,sector=self.sector,column=column,row=row,mask=self.mask,inputNums=None,mode=mode)
        results = self._wcs_time_info(results,cut)
        #try:
        gaia = pd.read_csv(f'{self.path}/Cut{cut}of{self.n**2}/local_gaia_cat.csv')
        results = match_result_to_cat(deepcopy(results),gaia,columns=['Source'])
        results = results.rename(columns={'Source': 'GaiaID'})
        # except:
        #     print('No local Gaia catalog, can not cross match.')
        # try:
        variables = pd.read_csv(f'{self.path}/Cut{cut}of{self.n**2}/variable_catalog.csv')
        results = match_result_to_cat(deepcopy(results),variables,columns=['Type','Prob'])
        # except:
        #     print('No local variable catalog, can not cross match.')

        results['xccd'] = deepcopy(results['xint'] + cutCorners[cut-1][0]).astype(int)
        results['yccd'] = deepcopy(results['yint'] + cutCorners[cut-1][1]).astype(int)

        wcs_save = self.wcs.to_fits()
        wcs_save[0].header['NAXIS'] = self.wcs.naxis
        wcs_save[0].header['NAXIS1'] = self.wcs._naxis[0]
        wcs_save[0].header['NAXIS2'] = self.wcs._naxis[1]
        wcs_save.writeto(f'{self.path}/Cut{cut}of{self.n**2}/wcs.fits',overwrite=True)
        results.to_csv(f'{self.path}/Cut{cut}of{self.n**2}/detected_sources.csv',index=False)
        results['bkg_level'] = 0
        if self.bkg is not None:
            f = results['frame'].values
            x = results['xint'].values; y = results['yint'].values
            bkg = self.bkg
            mask = np.zeros_like(bkg)
            mask[f,y,x] = 1
            mask = convolve(mask,np.ones((1,3,3)))
            b = np.nansum(bkg*mask,axis=(1,2))
            results['bkg_level'].iloc[f] = b[f]
        else:
            results['bkg_level'] = 0
        self.sources = results
        self._get_all_independent_events()
        self._asteroid_checker()
        self.events['objid'] = self.events['objid'].astype(int)
        self.events.to_csv(f'{self.path}/Cut{cut}of{self.n**2}/detected_events.csv',index=False)

    def plot_results(self,cut):

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

    def count_detections(self,cut,starkiller=False,asteroidkiller=False,lower=None,upper=None,sig_thresh=None):

        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        if starkiller:
            r = self.events[self.events['source_mask']==0]
        else:
            r = self.events

        if asteroidkiller:
            r = r[~(r['Type']=='Asteroid')]

        if sig_thresh is not None:
            r = r[r['lc_sig']>sig_thresh]

        array = r['objid'].values
        counts = []
        ids = np.unique(array)
        for id in ids:
            counts.append(np.nansum(self.events[self.events['objid']==id]['n_detections'].values))
        dictionary = dict(zip(ids, counts))

        if upper is not None:
            if lower is not None:
                dictionary = dict((k, v) for k, v in dictionary.items() if lower < v < upper)
            else:
                dictionary = dict((k, v) for k, v in dictionary.items() if v < upper)
        elif lower is not None:
            dictionary = dict((k, v) for k, v in dictionary.items() if lower < v)

        return dictionary

    def _check_dirs(self,save_path):
        """
        Check that all reduction directories are constructed.

        Parameters:
        -----------
        dirlist : list
            List of directories to check for, if they don't exist, they will be created.
        """
        #for d in dirlist:
        if not os.path.isdir(save_path):
            try:
                os.mkdir(save_path)
            except:
                pass

    def period_bin(self,frequency,power,power_limit=1):
        if np.isfinite(frequency):
            p = 1/frequency
            if power > power_limit:
                if p <= 1/24:
                    extension = '1hr_below'
                elif (p > 1/24) & (p <= 10/24):
                    extension = '1to10hr'
                elif (p > 10/24) & (p <= 1):
                    extension = '10hrto1day'
                elif (p > 1) & (p <= 10):
                    extension = '1to10day'
                elif (p >10):
                    extension = '10day_greater'
            else:
                extension = 'none'
        else:
            extension = 'none'
        return extension


    def plot_ALL(self,cut,save_path=None,lower=3,starkiller=False,save_lc=True):
        detections = self.count_detections(cut=cut,lower=lower,starkiller=starkiller)
        if save_path is None:
            save_path = self.path + f'/Cut{cut}of{self.n**2}/figs/'
            print('Figure path: ',save_path)
            self._check_dirs(save_path)
        inds = list(detections.keys())
        
        events = Parallel(n_jobs=int(multiprocessing.cpu_count()/2))(delayed(self.plot_source)(cut,ind,event='seperate',savename='auto',save_path=save_path) for ind in inds)
        



    def plot_source(self,cut,id,event='seperate',savename=None,save_path='.',
                    star_bin=True,period_bin=True,type_bin=True,objectid_bin='auto',
                    include_periodogram=False,latex=True,period_power_limit=10,
                    asteroid_check=True,zoo_mode=True,save_lc=True):
        if latex:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
        #else:
            #plt.rc('text', usetex=False)
            #plt.rc('font', family='sans-serif')
        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut
        sources =  self.events[self.events['objid']==id]
        total_events = len(sources)
        if type(objectid_bin) == str:
            if total_events > 5:
                objectid_bin = True
            else:
                objectid_bin = False
        if type(event) == str:
            if event.lower() == 'seperate':
                pass
            elif event.lower() == 'all':
                e = deepcopy(sources.iloc[0])
                e['frame_end'] = sources['frame_end'].iloc[-1]
                e['mjd_end'] = sources['mjd_end'].iloc[-1]
                e['mjd_duration'] = e['mjd_end'] - e['mjd_start']
                e['frame'] = (e['frame_end'] + e['frame_start']) / 2 
                e['mjd'] = (e['mjd_end'] + e['mjd_start']) / 2 
                sources = e.to_frame().T
        elif type(event) == int:
            sources = deepcopy(sources.iloc[sources['eventID'].values == event])
        elif type(event) == list:
            sources = deepcopy(sources.iloc[sources['eventID'].isin(event).values])
        else:
            m = "No valid option selected, input either 'all', 'seperate', an integer event id, or list of inegers."
            raise ValueError(m)
        #source = self.result[self.result['objid']==id]
        time = self.time - self.time[0]
        for i in range(len(sources)):
            #if type(sources) == list:
            #   source = sources[0]
            #else:
            source = deepcopy(sources.iloc[i])
            #x = source.iloc[0]['xint'].astype(int)
            #y = source.iloc[0]['yint'].astype(int)
            x = (source['xint']+0.5).astype(int)
            y = (source['yint']+0.5).astype(int)

            #frames = source['frame'].values
            if include_periodogram:
                fig,ax = plt.subplot_mosaic([[1,1,1,2,2],[1,1,1,3,3],[4,4,4,4,4]],figsize=(7,9),constrained_layout=True)
                fig,ax = plt.subplot_mosaic([[1,1,1,2,2],[1,1,1,3,3],[4,4,4,4,4]],figsize=(7,9),constrained_layout=True)
            else:
                fig,ax = plt.subplot_mosaic([[1,1,1,2,2],[1,1,1,3,3]],figsize=(7*1.1,5.5*1.1),constrained_layout=True)

            frameStart = int(source['frame_start']) #min(source['frame'].values)
            frameEnd = int(source['frame_end']) #max(source['frame'].values)

            f = np.nansum(self.flux[:,y-1:y+2,x-1:x+2],axis=(2,1))
            if frameEnd - frameStart >= 2:
                #brightestframe = source['frame'].values[np.where(f[source['frame'].values] == np.nanmax(f[source['frame'].values]))[0][0]]
                brightestframe = frameStart + np.where(abs(f[frameStart:frameEnd]) == np.nanmax(abs(f[frameStart:frameEnd])))[0][0]
            else:
                brightestframe = frameStart
            try:
                brightestframe = int(brightestframe)
            except:
                brightestframe = int(brightestframe[0])
            if brightestframe >= len(self.flux):
                brightestframe -= 1
            if frameEnd >= len(self.flux):
                frameEnd -= 1
            frames = np.arange(0,len(self.time))
            frames = (frames >= frameStart) & (frames <= frameEnd)
            lc = np.array([time,f,frames]).T
            fstart = frameStart-20
            if fstart < 0:
                fstart = 0
            zoom = f[fstart:frameEnd+20]
            
            if (frameEnd - frameStart) > 2:
            #    ax[1].axvspan(time[frameStart],time[frameEnd],color='C1',alpha=0.4)
                duration = time[frameEnd] - time[frameStart]
            else:
            #    ax[1].axvline(time[(frameEnd + frameStart)//2],color='C1')
                duration = 0
            if zoo_mode:
                ax[1].set_title('Is there a transient in the orange region?',fontsize=15)    
            else:
                ax[1].set_title('Lightcurve',fontsize=15)    
            
            ax[1].plot(time[fstart:frameEnd+20],zoom,'k',alpha=0)
            insert_ylims = ax[1].get_ylim()
            tn = deepcopy(time)
            fn = deepcopy(f)
            b = np.where(np.diff(tn) > 0.5)[0]
            timen = np.insert(tn,b,np.nan)
            fn = np.insert(fn,b,np.nan)
            ax[1].plot(timen,fn,'k',alpha=0.8)
            ax[1].set_ylabel('Brightness',fontsize=15,labelpad=10)
            ax[1].set_xlabel('Time (days)',fontsize=15)
            ylims = ax[1].get_ylim()
            ax[1].set_ylim(ylims[0],ylims[1]+(abs(ylims[0]-ylims[1])))
            ax[1].set_xlim(np.min(time),np.max(time))
            if zoo_mode:
                axins = ax[1].inset_axes([0.02, 0.55, 0.96, 0.43])
            else:
                axins = ax[1].inset_axes([0.1, 0.55, 0.86, 0.43])
                
            axins.axvspan(time[frameStart],time[frameEnd],color='C1',alpha=0.4)
            axins.plot(timen,fn,'k',alpha=0.8)
            fe = frameEnd + 20
            if fe >= len(time):
                fe = len(time) - 1
            axins.set_xlim(time[fstart],time[fe])
            axins.set_ylim(insert_ylims[0],insert_ylims[1])
            mark_inset(ax[1], axins, loc1=3, loc2=4, fc="none", ec="r",lw=2)
            plt.setp(axins.spines.values(), color='r',lw=2)
            plt.setp([axins.get_xticklines(), axins.get_yticklines()], color='C3')
            
            if zoo_mode:
                axins.yaxis.set_tick_params(labelleft=False,left=False)
                axins.xaxis.set_tick_params(labelbottom=False,bottom=False)
                ax[1].yaxis.set_tick_params(labelleft=False,left=False)

            
            ymin = y - 9
            if ymin < 0:
                ymin = 0 
            xmin = x -9
            if xmin < 0:
                xmin = 0
            bright_frame = self.flux[brightestframe,y-1:y+2,x-1:x+2]
            vmin = np.percentile(self.flux[brightestframe],16)
            #if vmin > -5:
            #   vmin =-5
            vmax = np.percentile(bright_frame,80)
            if vmin >= vmax:
                vmin = vmax - 5
            #   vmax = 10
            cutout_image = self.flux[:,ymin:y+10,xmin:x+10]
            ax[2].imshow(cutout_image[brightestframe],cmap='gray',origin='lower',vmin=vmin,vmax=vmax)
            #ax[2].plot(source['xcentroid'] - xmin,source['ycentroid'] - ymin,'C1*',alpha=0.8)
            rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin),5,5, linewidth=3, edgecolor='r', facecolor='none')
            ax[2].add_patch(rect)

            #ax[2].set_xlabel(f'Time {np.round(time[brightestframe],2)}')
            ax[2].set_title('Brightest image',fontsize=15)
            #ax[2].set_title('Does the spot in red move?')
            ax[2].get_xaxis().set_visible(False)
            ax[2].get_yaxis().set_visible(False)
            ax[3].get_xaxis().set_visible(False)
            ax[3].get_yaxis().set_visible(False)
            
            try:
                tdiff = np.where(time-time[brightestframe] >= 1/24)[0][0]
            except:
                tdiff = np.where(time[brightestframe] - time >= 1/24)[0][-1]
            after = tdiff#brightestframe + 1


            
            if after >= len(cutout_image):
                after = len(cutout_image) - 1 
            #before = brightestframe - 5
            #if before < 0:
            #    before = 0
            ax[3].imshow(cutout_image[after],
                        cmap='gray',origin='lower',vmin=vmin,vmax=vmax)
            #ax[3].plot(source['xcentroid'] - xmin,source['ycentroid'] - ymin,'C1*',alpha=0.8)
            rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin),5,5, linewidth=3, edgecolor='r', facecolor='none')
            ax[3].add_patch(rect)
            ax[3].set_title('1 hour later',fontsize=15)
            ax[3].annotate('', xy=(0.2, 1.15), xycoords='axes fraction', xytext=(0.2, 1.), 
                                arrowprops=dict(arrowstyle="<|-", color='r',lw=3))
            ax[3].annotate('', xy=(0.8, 1.15), xycoords='axes fraction', xytext=(0.8, 1.), 
                                arrowprops=dict(arrowstyle="<|-", color='r',lw=3))
            
            if include_periodogram:
                frequencies = periodogram(period,axis=ax[4])
                unit = u.electron / u.s
                light = lk.LightCurve(time=Time(self.time, format='mjd'),flux=(f - np.nanmedian(f))*unit)
                period = light.to_periodogram()
            #else:
            #    frequencies = periodogram(period,axis=None,plot=False)
            #    signal_num = frequencies['signal_num']
            #    harmonic = frequencies['harmonic']
            #    textstr = r'$\bf{Period}$' + '\n'
                #for i in range(1):
            #    ind = (signal_num == 1) & (harmonic == 1)
                '''
                try:
                    #print('power: ',frequencies['peak_power'][ind][0])
                    if frequencies['peak_power'][ind][0] > period_power_limit:
                        period = str(np.round(1/frequencies['peak_freq'][ind][0],2)) +' days\n'
                    
                        textstr += period
                    

                    # these are matplotlib.patch.Patch properties
                        props = dict(boxstyle='round', facecolor='white', alpha=0.7)

                        # place a text box in upper left in axes coords
                        ax[1].text(0.05, 0.95, textstr[:-1], transform=ax[1].transAxes, fontsize=10,
                                verticalalignment='top', bbox=props)
                        ylims = ax[1].get_ylim()
                        start_flux = np.nanmax(f[:len(f)//2])
                        if ylims[1] < start_flux * 1.4:
                            ax[1].set_ylim(ylims[0],start_flux * 1.4)
                except:
                    pass
                '''

            #plt.tight_layout()
            #plt.subplots_adjust(hspace=0.1)
            #plt.subplots_adjust(wspace=0.1)
            if asteroid_check:
                s = self.sources.iloc[self.sources.objid.values == id]
                e = s.iloc[(s.frame.values >= frameStart) & (s.frame.values <= frameEnd)]
                x = e.xcentroid.values
                y = e.ycentroid.values
                dist = np.sqrt((x[:,np.newaxis]-x[np.newaxis,:])**2 + (y[:,np.newaxis]-y[np.newaxis,:])**2)
                dist = np.nanmax(dist,axis=1)
                dist = np.nanmean(dist)
                if len(x)>= 2:
                    cor = np.round(abs(pearsonr(x,y)[0]),1)
                else:
                    cor = 0
                dpass = dist - 0.8
                cpass = cor - 0.8
                asteroid = dpass + cpass > 0 
                if asteroid & (duration < 1):
                    source['Type'] = 'Asteroid'
                    source['Prob'] = 1
                
                
            if savename is not None:
                sp = deepcopy(save_path)
                splc = deepcopy(save_path).replace('fig','lc')
                self._check_dirs(splc)
                if savename.lower() == 'auto':
                    savename = f'Sec{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{self.cut}_object{id}'
                if star_bin:
                    if source['GaiaID'] > 0:
                        extension = 'star'
                    else:
                        extension = 'no_star'
                sp += '/' + extension
                #splc += '/' + extension
                self._check_dirs(sp)
                #self._check_dirs(splc)

                if period_bin:
                    if type_bin:
                        if source['Prob'] > 0:
                            extension = source['Type']
                        else:
                            extension = self.period_bin(source['peak_freq'],source['peak_power'])
                    sp += '/' + extension
                    self._check_dirs(sp)
                    #splc += '/' + extension
                    #self._check_dirs(splc)
                    
                if objectid_bin:
                    extension = f'{self.sector}_{self.cam}_{self.ccd}_{self.cut}_{id}'
                    sp += '/' + extension
                    self._check_dirs(sp)
                    #splc += '/' + extension
                    #self._check_dirs(splc)
                if event == 'all':
                    plt.savefig(sp+'/'+savename+'_all_events.png', bbox_inches = "tight")
                else:
                    plt.savefig(sp+'/'+savename+f'_event{i+1}of{total_events}.png', 
                                bbox_inches = "tight")
                if save_lc:
                    headers = ['mjd','counts','event']
                    lc = pd.DataFrame(data=lc,columns=headers)
                    if event == 'all':
                        lc.to_csv(splc+'/'+savename+'_all_events.csv', index=False)
                    else:
                        lc.to_csv(splc+'/'+savename+f'_event{i+1}of{total_events}.csv', index=False)
                #np.save(save_path+'/'+savename+'_lc.npy',[time,f])
                #np.save(save_path+'/'+savename+'_cutout.npy',cutout_image)
                self.save_base = sp+'/'+savename
        self.lc = [time,f]
        self.cutout = cutout_image
        
        #self.periodogram = period
        #self.frequencies = frequencies
        return source



    def locate_transient(self,cut,xcentroid,ycentroid,threshold=3):

        if cut != self.cut:
            self._gather_data(cut)
            self._gather_results(cut)
            self.cut = cut

        return self.events[(self.events['ycentroid'].values < ycentroid+threshold) & (self.events['ycentroid'].values > ycentroid-threshold) & (self.events['xcentroid'].values < xcentroid+threshold) & (self.events['xcentroid'].values > xcentroid-threshold)]

    def full_ccd(self,psflike_lim=0,psfdiff_lim=1,savename=None):

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

