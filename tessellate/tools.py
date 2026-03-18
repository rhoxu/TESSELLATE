import numpy as np
import os

fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27			   # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

def _Save_space(Save,delete=False):
    """
    Creates a path if it doesn't already exist.
    """
    try:
        os.makedirs(Save)
    except FileExistsError:
        if delete:
            os.system(f'rm -r {Save}/')
            os.makedirs(Save)
        else:
            pass

def _Remove_emptys(files):
    """
    Deletes corrupt fits files before creating cube
    """

    deleted = 0
    for file in files:
        size = os.stat(file)[6]
        if size < 35500000:
            os.system('rm ' + file)
            deleted += 1
    return deleted

def _Extract_fits(pixelfile):
    """
    Quickly extract fits
    """
    from astropy.io import fits

    try:
        hdu = fits.open(pixelfile)
        return hdu
    except OSError:
        print('OSError ',pixelfile)
        return
    
def _Print_buff(length,string):

    strLength = len(string)
    buff = '-' * int((length-strLength)/2)
    return f"{buff}{string}{buff}"

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

def _remove_ffis(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            os.system(f'rm -r -f {data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/image_files')

    os.chdir(home_path)

def _remove_cubes(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            if part:
                for i in range(1,3):
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}')
                        os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_cube.fits')
                        os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_wcs.fits')
                        os.system(f'rm -f cubed.txt')
                    except:
                        pass
            else:
                try:
                    os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}')
                    os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_cube.fits')
                    os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_wcs.fits')
                    os.system(f'rm -f cubed.txt')
                except:
                    pass

    os.chdir(home_path)

def _remove_cuts(data_path,sector,n,cams,ccds,cuts,part):

    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        os.system(f'rm -r -f {data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                else:
                    os.system(f'rm -r -f {data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')

def _remove_reductions(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system(f'rm -f *.npy')
                            os.system(f'rm -f reduced.txt')
                            os.system(f'rm -f detected_events.csv')
                            os.system(f'rm -f detected_sources.csv')
                            os.system(f'rm -f detected_objects.csv')
                            os.system('rm -f figs.zip')
                            os.system('rm -f lcs.zip')  
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f *.npy')
                        os.system(f'rm -f reduced.txt')
                        os.system(f'rm -f detected_events.csv')
                        os.system(f'rm -f detected_sources.csv')
                        os.system(f'rm -f detected_objects.csv')
                        os.system('rm -f figs.zip')
                        os.system('rm -f lcs.zip')  
                    except:
                        pass   

    os.chdir(home_path)

def _remove_search(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system(f'rm -f detected_events.csv')
                            os.system(f'rm -f detected_sources.csv')
                            os.system(f'rm -f detected_objects.csv')
                            os.system('rm -f figs.zip')
                            os.system('rm -f lcs.zip') 
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f detected_events.csv')
                        os.system(f'rm -f detected_sources.csv')
                        os.system(f'rm -f detected_objects.csv')
                        os.system('rm -f figs.zip')
                        os.system('rm -f lcs.zip')  
                    except:
                        pass 
    os.chdir(home_path)
    
def _remove_plots(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system('rm -r -f figs.zip')
                            os.system('rm -r -f lcs.zip')
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system('rm -r -f figs.zip')
                        os.system('rm -r -f lcs.zip')
                    except:
                        pass 
    os.chdir(home_path)

def delete_files(filetype,data_path,sector,n=4,cams='all',ccds='all',cuts='all',part=False):

    if cams == 'all':
        cams = [1,2,3,4]
    elif type(cams) == int:
        cams = [cams]
    if ccds == 'all':
        ccds = [1,2,3,4]
    elif type(ccds) == int:
        ccds = [ccds]
    if cuts == 'all':
        cuts = np.linspace(1,n**2,n**2).astype(int)
    elif type(cuts) == int:
        cuts = [cuts]
        
    possibleFiles = {'ffis':_remove_ffis,
                        'cubes':_remove_cubes,
                        'cuts':_remove_cuts,
                        'reductions':_remove_reductions,
                        'search':_remove_search,
                        'plot':_remove_plots}
    
    if filetype.lower() in possibleFiles.keys():
        function = possibleFiles[filetype.lower()]
        function(data_path,sector,n,cams,ccds,cuts,part)
    else:
        e = 'Invalid filetype! Valid types: "ffis" , "cubes" , "cuts" , "reductions" , "search", "plot". '
        raise AttributeError(e)
    
    
def weighted_avg_var(group, weight_col):
    import pandas as pd

    weighted_stats = {}
    numeric_cols = group.select_dtypes(include='number').columns  # Select only numeric columns
    miss = ['Prob','n_detections','GaiaID','flux_sign',
            'ra_source','x_source','y_source','dec_source',
            'e_ra_source','e_x_source','e_y_source','e_dec_source',
            'source_mask','eventID','xccd_source','yccd_source',
            'e_xccd_source','e_yccd_source']
    if len(group) == 1:  # If group has only one entry, return the row but with the same column structure
        original_row = group.iloc[0].copy()
        result = {}
        for col in numeric_cols:
            if (col != 'objid') & (col not in miss):
                result[col] = original_row[col]  # Original value
                result[f'e_{col}'] = np.nan  # No variance
            else:
                result[col] = group[col].iloc[0]
        return pd.Series(result)
    
    for col in numeric_cols:
        if (col != 'objid') & (col not in miss):  # Exclude the weight column itself
            # Compute the weighted average using nansum
            weighted_avg = np.ma.masked_invalid(group[col] * group[weight_col]).sum() / np.ma.masked_invalid(group[weight_col]).sum()
            # Compute the weighted variance using nansum
            variance = np.ma.masked_invalid(group[weight_col] * (group[col] - weighted_avg) ** 2).sum() / np.ma.masked_invalid(group[weight_col]).sum()
            # Store both weighted average and variance
            weighted_stats[col] = weighted_avg
            weighted_stats[f'e_{col}'] = variance
        else:
            weighted_stats[col] = group[col].iloc[0]
    return pd.Series(weighted_stats)


def pandas_weighted_avg(df,weight_col='sig'):
    # df = df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col,include_groups=False).reset_index()
    # print(df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col).head())
    df = df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col).reset_index(drop=True)
    return df

def consecutive_points(data, stepsize=2):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)

def Gaussian(t, A, t0, sigma, offset):
    return A * np.exp(-0.5 * ((t - t0) / sigma)**2) + offset

def Distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def RoundToInt(num):
    return np.floor(num+0.5).astype(int)

def Exp_func(x,a,b,c):
   e = np.exp(a)*np.exp(-x/np.exp(b)) + np.exp(c)
   return e






def Generate_LC(time,flux,x,y,frame_start=None,frame_end=None,method='sum',radius=1.5):
    
    from photutils.aperture import CircularAperture, RectangularAnnulus, ApertureStats, aperture_photometry
    from scipy.signal import fftconvolve
    
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

    if method.lower() == 'aperture':
        aperture = CircularAperture([x, y], radius)
        annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
        flux = []
        flux_err = []
        for i in range(len(f)):
            m = sigma_clip(data,masked=True,sigma=5).mask
            mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
            aperstats_sky = ApertureStats(f[i], annulus_aperture,mask = mask)
            phot_table = aperture_photometry(f[i], aperture)
            bkg_std = aperstats_sky.std
            flux_err += [aperture.area * bkg_std]
            flux += [phot_table['aperture_sum'].value[0]]
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        return t, flux, flux_err
    elif method.lower() == 'sum':
        xint = int(np.round(x,0))
        yint = int(np.round(y,0))
        buffer = np.floor(radius).astype(int)
        f = np.nansum(f[:,yint-buffer:yint+buffer+1,xint-buffer:xint+buffer+1],axis=(1,2))

        return t,f



def Frame_Bin(time, flux=None, frame_bin=1):
    break_idx = np.argmax(np.diff(time)) + 1

    def _bin_segment(t, f=None):
        points = np.arange(0, len(t), frame_bin)
        binned_time = np.array([np.nanmean(t[i:i+frame_bin]) for i in points])
        if f is None:
            return binned_time
        binned_flux = np.array([np.nanmean(f[i:i+frame_bin], axis=0) for i in points])
        return binned_time, binned_flux

    if flux is None:
        return np.concatenate([_bin_segment(time[:break_idx]), _bin_segment(time[break_idx:])])
    
    time_a, flux_a = _bin_segment(time[:break_idx], flux[:break_idx])
    time_b, flux_b = _bin_segment(time[break_idx:], flux[break_idx:])
    return np.concatenate([time_a, time_b]), np.concatenate([flux_a, flux_b])