import numpy as np
import os
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
import requests
from PIL import Image
from io import BytesIO
from astropy.wcs import WCS

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
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system('rm -r -f figs')
                        os.system('rm -r -f lcs')  
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
    df = df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col,include_groups=False).reset_index()
    return df

def consecutive_points(data, stepsize=2):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)


def _Get_images(ra,dec,filters):

    """Query ps1filenames.py service to get a list of images"""

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table

def _Get_url(ra, dec, size, filters, color=False):

    """Get URL for images in the table"""

    table = _Get_images(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={ra}&dec={dec}&size={size}&format=jpg")

    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url

def _Get_im(ra, dec, size,color):

    """Get color image at a sky position"""

    if color:
        url = _Get_url(ra,dec,size=size,filters='grz',color=True)
        r = requests.get(url)
    else:
        url = _Get_url(ra,dec,size=size,filters='i')
        r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im

def _Panstarrs_phot(ra,dec,size):

    grey_im = _Get_im(ra,dec,size=size*6,color=False)
    colour_im = _Get_im(ra,dec,size=size*6,color=True)

    plt.rcParams.update({'font.size':12})
    fig,ax = plt.subplots(ncols=2,figsize=(3*fig_width,1*fig_width))

    ax[0].imshow(grey_im,origin="lower",cmap="gray")
    ax[0].set_title('PS1 i')
    ax[0].set_xlabel('px (0.25")')
    ax[0].set_ylabel('px (0.25")')
    ax[1].set_title('PS1 grz')
    ax[1].imshow(colour_im,origin="lower")
    ax[1].set_xlabel('px (0.25")')
    ax[1].set_ylabel('px (0.25")')

    for i in range(-1,3):
        ax[0].axvline(size*3+i*21/0.25-21/0.5,color='white',alpha=0.5)
        ax[0].axhline(size*3+i*21/0.25-21/0.5,color='white',alpha=0.5)
        ax[1].axvline(size*3+i*21/0.25-21/0.5,color='white',alpha=0.5)
        ax[1].axhline(size*3+i*21/0.25-21/0.5,color='white',alpha=0.5)

    return fig


def _Skymapper_phot(ra,dec,size):
    """
    Gets g,r,i from skymapper.
    """
    og_size = size
    size /= 3600

    url = f"https://api.skymapper.nci.org.au/public/siap/dr2/query?POS={ra},{dec}&SIZE={size}&BAND=g,r,i&FORMAT=GRAPHIC&VERB=3"
    table = Table.read(url, format='ascii')

    # sort filters from red to blue
    flist = ["irg".find(x) for x in table['col3']]
    table = table[np.argsort(flist)]

    if len(table) > 3:
        # pick 3 filters
        table = table[[0,len(table)//2,len(table)-1]]

    wcsList = []
    for i in range(len(table)):
        crpix = np.array(table['col23'][i].split(' ')).astype(float)
        crval = np.array(table['col24'][i].split(' ')).astype(float)
        cdmatrix = np.array(table['col25'][i].split(' ')).astype(float).reshape(2,2)
        
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = crpix
        wcs.wcs.crval = crval
        wcs.wcs.cd = cdmatrix
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Common projection type for celestial coordinates

        wcsList.append(wcs)

    plt.rcParams.update({'font.size':12})
    fig,ax = plt.subplots(ncols=3,figsize=(3*fig_width,1*fig_width))
    url = table[2][3]
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    ax[0].imshow(im,origin="upper",cmap="gray")
    ax[0].set_title('SkyMapper g')
    ax[0].set_xlabel('px (1.1")')

    url = table[1][3]
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    ax[1].set_title('SkyMapper r')
    ax[1].imshow(im,origin="upper",cmap="gray")
    ax[1].set_xlabel('px (1.1")')

    url = table[0][3]
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    ax[2].set_title('SkyMapper i')
    ax[2].imshow(im,origin="upper",cmap="gray")
    ax[2].set_xlabel('px (1.1")')

    # for i in range(-2,4):
    #     ax[0].axvline(og_size+i*21/1.1-21/2.2,color='white',alpha=0.5)
    #     ax[0].axhline(og_size+i*21/1.1-21/2.2,color='white',alpha=0.5)
    #     ax[1].axvline(og_size+i*21/1.1-21/2.2,color='white',alpha=0.5)
    #     ax[1].axhline(og_size+i*21/1.1-21/2.2,color='white',alpha=0.5)
    #     ax[2].axvline(og_size+i*21/1.1-21/2.2,color='white',alpha=0.5)
    #     ax[2].axhline(og_size+i*21/1.1-21/2.2,color='white',alpha=0.5)

    return fig,wcsList

def event_cutout(coords,size=50,phot=None):

    if phot is None:
        if coords[1] > -10:
            phot = 'PS1'
        else:
            phot = 'SkyMapper'
        
    if phot == 'PS1':
        fig = _Panstarrs_phot(coords[0],coords[1],size)

    elif phot.lower() == 'skymapper':
        fig,wcs = _Skymapper_phot(coords[0],coords[1],size)

    else:
        print('Photometry name invalid.')
        fig = None
        wcs = None

    plt.close()

    return fig,wcs