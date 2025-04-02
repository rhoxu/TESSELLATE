import numpy as np
import os
from astropy.io import fits
import pandas as pd


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
                            os.system('rm -r -f figs')
                            os.system('rm -r -f lcs')   
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f *.npy')
                        os.system(f'rm -f reduced.txt')
                        os.system(f'rm -f detected_events.csv')
                        os.system(f'rm -f detected_sources.csv')
                        os.system('rm -r -f figs')
                        os.system('rm -r -f lcs') 
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
                            os.system('rm -r -f figs')
                            os.system('rm -r -f lcs') 
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f detected_events.csv')
                        os.system(f'rm -f detected_sources.csv')
                        os.system('rm -r -f figs')
                        os.system('rm -r -f lcs')  
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
                            os.system('rm -r -f figs')
                            os.system('rm -r -f lcs')
                            os.system('rm -r -f object_lcs') 
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system('rm -r -f figs')
                        os.system('rm -r -f lcs')  
                        os.system('rm -r -f object_lcs') 
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
    df = df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col).reset_index()
    return df

def consecutive_points(data, stepsize=2):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)