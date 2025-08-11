import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from astropy.io import fits 
from astropy.wcs import WCS
from astropy.visualization import PercentileInterval, AsinhStretch

import requests
from PIL import Image
from io import BytesIO
import time as Time 

import pandas as pd
from matplotlib.patches import Ellipse

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


def _skymapper_objects(ra,dec,rad=30/60**2):
    """
    radius in degrees
    """
    query = f'https://skymapper.anu.edu.au/sm-cone/public/query?RA={ra}&DEC={dec}&SR={np.round(rad,3)}&RESPONSEFORMAT=CSV'
    sm = pd.read_csv(query)
    if len(sm) > 0:
        keep = ['object_id','raj2000','dej2000','u_psf', 'e_u_psf',
                'v_psf', 'e_v_psf','g_psf', 'e_g_psf','r_psf', 
                'e_r_psf','i_psf', 'e_i_psf','z_psf', 'e_z_psf','class_star']
        sm = sm[keep]
        sm = sm.rename(columns={'raj2000':'ra','dej2000':'dec','u_psf':'umag',
                                'v_psf':'vmag','g_psf':'gmag','r_psf':'rmag',
                                'i_psf':'imag','z_psf':'zmag',
                                'e_u_psf':'e_umag','e_v_psf':'e_vmag','e_g_psf':'e_gmag',
                                'e_r_psf':'e_rmag','e_i_psf':'e_imag','e_z_psf':'e_zmag'})
        sm['star'] = 0
        sm['star'].loc[sm['class_star'] >= 0.9] = 1
        sm['star'].loc[sm['class_star'] <= 0.7] = 0
        sm['star'].loc[(sm['class_star'] > 0.7) & (sm['class_star'] < 0.9)] = 2
    else:
        sm = None
    return sm
    

def _Skymapper_phot(ra,dec,size):
    """
    Gets g,r,i from skymapper.
    """
    size*=1.1
    og_size = size
    size /= 3600

    url = f"https://api.skymapper.nci.org.au/public/siap/dr4/query?POS={ra},{dec}&SIZE={size}&BAND=g,r,i&FORMAT=GRAPHIC&VERB=3&INTERSET=COVERS"
    max_attempts = 5

    complete = False
    attempt = 0
    while (not complete) & (attempt < max_attempts):
        try:
            table = Table.read(url, format='ascii').to_pandas()
            complete = True
        except:
            attempt += 1
            print('failed')
            Time.sleep(1)

    t = table[(table['col16'] == 'main')]
    t_unique = t[~t.duplicated(subset='col3', keep='first')]

    wcsList = []        
    images = []
    filts = ['g','r','i']
    max_attempts = 5
    for i in range(len(filts)):
        complete = False
        attempt = 0
        while (not complete) & (attempt < max_attempts):
            try:
                f = t_unique.loc[t_unique['col3'] == filts[i]].iloc[0]
                url = f['col4']
                r = requests.get(url)

                im = (Image.open(BytesIO(r.content)))* 10**((f['col22'] - 25)/-2.5)
                im[im==0] = np.nan
                im -= np.nanmedian(im)
                im[np.isnan(im)] = 0
                images += [im]
                
                crpix = np.array(f['col23'].split(' ')).astype(float)
                crval = np.array(f['col24'].split(' ')).astype(float)
                cdmatrix = np.array(f['col25'].split(' ')).astype(float).reshape(2,2)

                wcs = WCS(naxis=2)
                wcs.wcs.crpix = crpix
                wcs.wcs.crval = crval
                wcs.wcs.cd = cdmatrix
                wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Common projection type for celestial coordinates

                wcsList.append(wcs)
                
                complete = True
            except:
                attempt += 1
                Time.sleep(1)
    #images = np.array(images)

    fig = plt.figure(figsize=(3*fig_width,1*fig_width))

    plt.rcParams.update({'font.size':11})
    titles = ['SkyMapper g','SkyMapper r','SkyMapper i']
    for i in range(3):
        ax = plt.subplot(1,3,i+1,projection=wcsList[i])
        ax.imshow(images[i],cmap="gray_r")
        ax.grid(alpha=0.5,color='w')
        ax.set_title(titles[i])
        if i == 1:
            ax.set_xlabel('Right Ascension')
        else:
            ax.set_xlabel(' ')
        if i == 0:
            ax.set_ylabel('Declination')
        else:
            #
            ax.set_ylabel(' ')
            ax.coords['dec'].set_ticklabel_visible(False)
    plt.tight_layout()

    return fig,wcsList,og_size*2

def _delve_objects(ra,dec,size=60/60**2):
    from dl import queryClient as qc
    query = f"""
        SELECT o.quick_object_id,o.ra, o.dec,
        o.mag_psf_g,o.mag_psf_r,o.mag_psf_i,o.mag_psf_z,
        o.mag_auto_g,o.mag_auto_r,o.mag_auto_i,o.mag_auto_z,
        o.magerr_psf_g,o.magerr_psf_r,o.magerr_psf_i,o.magerr_psf_z,
        o.magerr_auto_g,o.magerr_auto_r,o.magerr_auto_i,o.magerr_auto_z,
        o.extended_class_g, o.extended_class_r,o.extended_class_i,o.extended_class_z
        FROM delve_dr2.objects AS o
        WHERE q3c_radial_query(ra,dec,{ra},{dec},{size})
        """
    try:
        result = qc.query(sql=query,fmt='pandas')
        result.replace(99.0,np.nan,inplace=True)
        result['star'] = 0
        ty = []
        val = result[['extended_class_g','extended_class_r','extended_class_i','extended_class_z']].values
        for v in val:
            if 3 in v:
                ty += [0]
            elif 1 in v:
                ty += [1]
            else:
                ty += [2]
        result['star'] = ty
        return result
    except:
        return

def _DESI_phot(ra,dec,size):

    size = size * 5
    urlFITS = f"http://legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&size={size}&layer=ls-dr10"
    urlIM = f"http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&size={size}&layer=ls-dr10"

    response = requests.get(urlIM)
    if response.status_code == 200:
        image = Image.open(BytesIO(requests.get(urlIM).content))
        #if np.nansum(image) == 0:
        try:
            hdulist = fits.open(BytesIO(requests.get(urlFITS).content),ignore_missing_simple=True)
            hdu = hdulist[0]
            wcs = WCS(hdu.header)
            wcs = wcs.dropaxis(2)

            plt.rcParams.update({'font.size':12})
            fig = plt.figure(figsize=(8,8))#3*fig_width,1*fig_width))
            ax = plt.subplot(111,projection=wcs)
            ax.imshow(image,cmap="gray")
            ax.set_title('DES gri')
            ax.grid(alpha=0.2)
            ax.set_xlabel('Right Ascension')
            ax.set_ylabel('Declination')
            ax.invert_xaxis()
            return fig,wcs,size
        except Exception as error:
            print("DES Photometry failed: ", error)
            print(urlFITS)
            print(urlIM)
            return None,None,None
        #else:
          #  return None,None,None
    else:
        return None,None,None

def simbad_sources(ra,dec,size):
    from astroquery.simbad import Simbad

    simbad = Simbad()
    simbad.ROW_LIMIT = -1
    gal_query = f"""SELECT ra, dec, main_id, otype

                    FROM basic

                    WHERE otype != 'star..'

                    AND CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', {ra}, {dec} , {size})) = 1
                """
    gal = simbad.query_tap(gal_query)
    gal = gal.to_pandas()

    star_query = f"""SELECT ra, dec, main_id, otype

                    FROM basic

                    WHERE otype = 'star..'

                    AND CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', {ra}, {dec} , {size})) = 1
                 """
    stars = simbad.query_tap(star_query)
    stars = stars.to_pandas()

    gal['star'] = 0
    stars['star'] = 1
    sbad = pd.concat([stars,gal])
    
    return sbad

def check_simbad(cat,sbad):
    dist = np.sqrt((sbad['ra'].values[:,np.newaxis] - cat['ra'].values[np.newaxis,:])**2 + (sbad['dec'].values[:,np.newaxis] - cat['dec'].values[np.newaxis,:])**2)*60**2
    sind, catind = np.where(dist<5)
    cat['otype'] = 'none'
    if len(catind) > 0:
        cat.loc[catind,'star'] = sbad.loc[sind,'star'].values
        cat.loc[catind,'otype'] = sbad.loc[sind,'otype'].values
    return cat

def get_gaia(ra,dec,size):
    from astroquery.gaia import Gaia
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    Gaia.ROW_LIMIT = -1  # Ensure the default row limit.

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

    j = Gaia.cone_search_async(coord, radius=u.Quantity(100, u.arcsec))
    j = j.get_results().to_pandas()
    j['star'] = 1
    j.loc[(j['classprob_dsc_combmod_quasar'] > 0.7) | (j['classprob_dsc_combmod_galaxy'] > 0.7),'star'] = 2
    j.loc[(j['classprob_dsc_combmod_quasar'] > 0.9) | (j['classprob_dsc_combmod_galaxy'] > 0.9),'star'] = 0
    return j

def check_gaia(cat,gaia):
    dist = np.sqrt((gaia['ra'].values[:,np.newaxis] - cat['ra'].values[np.newaxis,:])**2 + (gaia['dec'].values[:,np.newaxis] - cat['dec'].values[np.newaxis,:])**2)*60**2
    gind, catind = np.where(dist<5)
    cat['dist'] = np.nan
    cat['dist_l'] = np.nan
    cat['dist_u'] = np.nan
    if len(catind) > 0:
        cat.loc[catind,'star'] = gaia.loc[gind,'star'].values
        cat.loc[catind,'dist'] = gaia.loc[gind,'distance_gspphot'].values
        cat.loc[catind,'dist_l'] = gaia.loc[gind,'distance_gspphot_lower'].values
        cat.loc[catind,'dist_u'] = gaia.loc[gind,'distance_gspphot_upper'].values
        
    return cat
    

def _add_sources(fig,target_coords,cat):
    axs = fig.get_axes()
    count = 0
    for ax in axs:
        ax.scatter(target_coords[0],target_coords[1], transform=ax.get_transform('fk5'),
                    edgecolors='red',marker='x',s=50,facecolors="red",linewidths=2,label='Target')
        

        # if error is not None:
        #     if len(error) > 1:
        #         xerr,yerr = error
        #     else:
        #         xerr = yerr = error
        #     ellipse = Ellipse(xy=(coords[0],coords[1]),  
        #                       width=error[0],height=error[1],     
        #                       edgecolor='red',facecolor='none',
        #                       linestyle=':', linewidth=3,
        #                       transform=ax.get_transform('fk5'))
        #     ax.add_patch(ellipse)

        
        
        # scatter stars 
        stars = cat.loc[cat['star'] ==1]
        ax.scatter(stars.ra,stars.dec, transform=ax.get_transform('fk5'),
                    edgecolors='w',marker='*',s=80,facecolors="None",linewidths=1,label='Star')

        # scatter galaxies
        gal = cat.loc[cat['star'] == 0]
        ax.scatter(gal.ra,gal.dec, transform=ax.get_transform('fk5'),
                    edgecolors='w',marker='o',s=80,facecolors="None",label='Galaxy')
        
        # scatter other
        gal = cat.loc[cat['star'] == 2]
        ax.scatter(gal.ra,gal.dec, transform=ax.get_transform('fk5'),
                    edgecolors='w',marker='D',s=80,facecolors="None",label='Possible galaxy')
        if count == 0:
            legend = ax.legend(loc=2,facecolor="black",fontsize=10)
            for text in legend.get_texts():
                text.set_color("white")
        count += 1
    return fig


def event_cutout(coords,real_loc=None,error=10,size=50,phot=None,check='gaia'):
    """
    Make an image using ground catalogs for the region of interest.

    parameters
    ----------
    coords : array
        array with elements of (ra,dec) in decimal degrees
    real_loc : array
        real on-sky position of target, if known.
    error : float
        Positional error in arcseconds, currently only 1 value is accepted.
    size : int
        size of the cutout image, likely in arcseconds 
    phot : str
        String defining the catalog to use. If None, then the catalog is automatically determined.
    check : str
        Check the photometry catalog against another catalog with better star detection. Currenty
        only gaia and simbad are available.
    """
    if real_loc is None:
        real_loc = coords
    if phot is None:
        fig,wcs,outsize = _DESI_phot(coords[0],coords[1],size)
        if fig is None:
            if coords[1] > -10:
                phot = 'PS1'
            else:
                phot = 'SkyMapper'
        else:
            phot = 'DESI'
            cat = _delve_objects(real_loc[0],real_loc[1])
            #fig = _add_sources(fig,real_loc,cat,error)

    if phot == 'PS1':
        fig,wcs,outsize = _Panstarrs_phot(coords[0],coords[1],size)
        cat = None
        #fig = _add_sources(fig,cat)

    elif phot.lower() == 'skymapper':
        fig,wcs,outsize = _Skymapper_phot(coords[0],coords[1],size)
        cat = _skymapper_objects(real_loc[0],real_loc[1])
        #fig = _add_sources(fig,real_loc,cat,error)
    elif phot is None:
        print('Photometry name invalid.')
        fig = None
        wcs = None
        return None,None,None,None,None
        
    if phot is not None:
        if check == 'simbad':
            sbad = simbad_sources(real_loc[0],real_loc[1],size/60**2)
            cat = check_simbad(cat,sbad)
        elif check == 'gaia':
            gaia = get_gaia(real_loc[0],real_loc[1],size/60**2)
            cat = check_gaia(cat,gaia)
    else:
        if check == 'simbad':
            cat = simbad_sources(real_loc[0],real_loc[1],size/60**2)
        elif check == 'gaia':
            cat = get_gaia(real_loc[0],real_loc[1],size/60**2)
    
    if cat is not None:
        fig = _add_sources(fig,real_loc,cat)

    plt.close()

    return fig,wcs,outsize, phot, cat