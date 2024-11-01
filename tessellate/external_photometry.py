import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from astropy.io import fits 
from astropy.wcs import WCS
from astropy.visualization import PercentileInterval, AsinhStretch

import requests
from PIL import Image
from io import BytesIO


fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27			   # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

def _Get_images(ra,dec,filters):

    """Query ps1filenames.py service to get a list of images"""

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table

def _Get_url_wcs(ra, dec, size, filters, color=False):

    """Get URL for images in the table"""

    table = _Get_images(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={ra}&dec={dec}&size={size}&format=fits")

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
        url = _Get_url_wcs(ra,dec,size=size,filters='grz',color=True)
        fh = fits.open(url)
        wcs = WCS(fh[0])
        # r = requests.get(url)
        # im = Image.open(BytesIO(r.content))

        fim = fh[0].data

        fim[np.isnan(fim)] = 0.0
        # set contrast to something reasonable
        transform = AsinhStretch() + PercentileInterval(99.5)
        bfim = transform(fim)

        # transform = AsinhStretch() + PercentileInterval(99.5)
        # bfim = transform(im)
        im = bfim
        
    else:
        url = _Get_url_wcs(ra,dec,size=size,filters='i')
        fh = fits.open(url[0])
        wcs = WCS(fh[0])

        fim = fh[0].data
        # replace NaN values with zero for display
        fim[np.isnan(fim)] = 0.0
        # set contrast to something reasonable
        transform = AsinhStretch() + PercentileInterval(99.5)
        bfim = transform(fim)
        im = bfim

    return im,wcs

def _Panstarrs_phot(ra,dec,size):

    grey_im,wcsI = _Get_im(ra,dec,size=size*6,color=False)
    colour_im,wcsGRZ = _Get_im(ra,dec,size=size*6,color=True)

    wcsList = [wcsGRZ,wcsGRZ,wcsGRZ,wcsI]

    plt.rcParams.update({'font.size':12})
    fig,ax = plt.subplots(ncols=4,figsize=(3*fig_width,1*fig_width))

    ax[2].imshow(grey_im,origin="lower",cmap="gray")
    ax[2].set_title('PS1 i')
    ax[2].set_xlabel('px (0.25")')
    ax[3].imshow(colour_im[0],origin="lower",cmap="gray")
    ax[3].set_title('PS1 z')
    ax[3].set_xlabel('px (0.25")')
    ax[1].imshow(colour_im[1],origin="lower",cmap="gray")
    ax[1].set_title('PS1 r')
    ax[1].set_xlabel('px (0.25")')
   
    ax[0].set_title('PS1 g')
    ax[0].imshow(colour_im[2],origin="lower",cmap="gray")
    ax[0].set_xlabel('px (0.25")')

    return fig,wcsList,grey_im.shape[0]


def _Skymapper_phot(ra,dec,size):
    """
    Gets g,r,i from skymapper.
    """
    size*=1.5
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

    return fig,wcsList,og_size*2

def _DESI_phot(ra,dec,size):

    size = size *4
    urlFITS = f"http://legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&size={size}"
    urlIM = f"http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&size={size}"

    response = requests.get(urlIM)
    if response.status_code == 200:
        image = Image.open(BytesIO(requests.get(urlIM).content))
        
        hdulist = fits.open(BytesIO(requests.get(urlFITS).content))
        hdu = hdulist[0]
        wcs = WCS(hdu.header)

        plt.rcParams.update({'font.size':12})
        fig,ax = plt.subplots(ncols=1,figsize=(3*fig_width,1*fig_width))
        ax.imshow(image,origin="upper",cmap="gray")
        ax.set_title('DESI grz')
        
        return fig,wcs,size
    else:
        return None,None,None


def event_cutout(coords,size=50,phot=None):

    if phot is None:

        fig,wcs,size = _DESI_phot(coords[0],coords[1],size)
        if fig is None:
            if coords[1] > -10:
                phot = 'PS1'
            else:
                phot = 'SkyMapper'
        
    if phot == 'PS1':
        fig,wcs,size = _Panstarrs_phot(coords[0],coords[1],size)

    elif phot.lower() == 'skymapper':
        fig,wcs,size = _Skymapper_phot(coords[0],coords[1],size)

    else:
        print('Photometry name invalid.')
        fig = None
        wcs = None

    plt.close()

    return fig,wcs,size