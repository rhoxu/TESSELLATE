import numpy as np
from copy import deepcopy
from tessellate.localisation import PSF_Fitter
import pandas as pd
from tqdm import tqdm 
from PRF import TESS_PRF
import pickle
from astropy.io import fits
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from astropy.wcs import WCS

datadir='/fred/oz335/_local_TESS_PRFs/'


def Predict_PSF_Quality(mag, sep, clf): 

    mag = np.atleast_1d(mag)
    sep = np.atleast_1d(sep)
    
    X = np.column_stack([mag, np.log10(sep + 1)])
    
    quality = clf.predict_proba(X)[:, 1]
    
    return quality[0] if len(quality) == 1 else quality

def Fit_Star(image,star,wcs,prf,plot_star=False):

    image_size = 5
    central_size = 3

    # -- Generate frame by finding the brightest pixel -- #
    x,y = wcs.all_world2pix(star.ra,star.dec,0)
    xint_trial = np.round(x).astype(int)
    yint_trial = np.round(y).astype(int)
    frame = deepcopy(image[yint_trial-1:yint_trial+2,xint_trial-1:xint_trial+2])
    iy, ix = np.unravel_index(np.nanargmax(frame), frame.shape)
    brightesty = yint_trial + (iy - 1)  # shift from 3x3 center
    brightestx = xint_trial + (ix - 1)
    frame = deepcopy(image[brightesty-image_size//2:brightesty+image_size//2+1,brightestx-image_size//2:brightestx+image_size//2+1])

    # -- Fit PSF -- #
    PSF_fitter = PSF_Fitter(image_size,prf,central_shape=central_size)
    PSF_fitter.fit_psf(frame,0.5,0.5)

    if plot_star:
        normframe = frame - np.nanmedian(frame)
        normframe /= np.nansum(normframe)
        fitpsf = PSF_fitter.psf

        # produce mask from which minimisation is calculated
        mask = slice(image_size//2-central_size//2,image_size//2+central_size//2+1),slice(image_size//2-central_size//2,image_size//2+central_size//2+1)

        fig,ax = plt.subplots(ncols=3,figsize=(15,5))
        ax[0].imshow(normframe,origin='lower',cmap='Greys_r',vmin=0,vmax=0.3)
        ax[0].scatter(image_size//2+PSF_fitter.source_x,image_size//2+PSF_fitter.source_y,c='r',label='PSF Fit')
        ax[0].scatter(image_size//2+x-brightestx,image_size//2+y-brightesty,c='g',label='Gaia')
        ax[0].legend()
        ax[0].set_xlabel('Median Subtracted + Normalised Image')
        

        ax[1].imshow(abs(fitpsf-normframe),origin='lower',cmap='Greys_r',vmin=0,vmax=5e-2)
        ax[1].set_xlabel('PSF Fit - MedSub+Norm Image')
        ax[1].text(image_size//2-1,image_size,f'PSF Fit:{np.nansum((fitpsf-normframe)[mask]**2)}')

        for i in range(3):
            for j in range(3):
                ax[1].text(
                    image_size//2-1+j, image_size//2-1+i, 
                    f"{abs(fitpsf-normframe)[image_size//2-1+i, image_size//2-1+j]**2:.2g}",   # 2 significant figures
                    ha="center", va="center",
                    color="red", fontsize=6
                )

        PSF_fitter.source(shiftx=x-brightestx,shifty=y-brightesty)
        fitpsf = PSF_fitter.psf
        ax[2].imshow(abs(fitpsf-normframe),origin='lower',cmap='Greys_r',vmin=0,vmax=5e-2)
        ax[2].set_xlabel('Gaia Loc - MedSub+Norm Image')
        ax[2].text(image_size//2-1,image_size,f'Gaia:{np.nansum((fitpsf-normframe)[mask]**2)}')

        
        for i in range(3):
            for j in range(3):
                ax[2].text(
                    image_size//2-1+j, image_size//2-1+i, 
                    f"{abs(fitpsf-normframe)[image_size//2-1+i, image_size//2-1+j]**2:.2g}",   # 2 significant figures
                    ha="center", va="center",
                    color="red", fontsize=6
                )    

    xGaia = x
    yGaia = y
    xFit = PSF_fitter.source_x + brightestx
    yFit = PSF_fitter.source_y + brightesty

    return xGaia,yGaia,xFit,yFit

def Fit_Cut_Stars(image,stars,wcs,sector,cam,ccd,cut):

    xsGaia = []
    ysGaia = []
    xsFit = []
    ysFit = []
    for i in tqdm(range(len(stars)),desc=f'Fitting sources S{sector}C{cam}C{ccd}C{cut}',position=0, leave=True,dynamic_ncols=False,ascii=True):
        star = stars.iloc[i]
        x,y = wcs.all_world2pix(star.ra,star.dec,0)
        xint_trial = np.round(x).astype(int)
        yint_trial = np.round(y).astype(int)
        
        prf = TESS_PRF(cam=cam,ccd=ccd,sector=sector,
                    colnum=xint_trial,rownum=yint_trial,localdatadir=datadir+'Sectors4+')
        
        xGaia,yGaia,xFit,yFit = Fit_Star(image,stars.iloc[i],wcs,prf,plot_star=False)
        xsGaia.append(xGaia)
        ysGaia.append(yGaia)
        xsFit.append(xFit)
        ysFit.append(yFit)
    
    return xsGaia,ysGaia,xsFit,ysFit
    

def Fit_CCD(image_path,gaia_cat,sector,cam,ccd,n=4,cut=None,quality_thresh=0.8,min_stars=100):

    # -- Open image and wcs -- #
    with fits.open(image_path) as f:
        image = f[1].data
        wcs = WCS(f[1].header)

    # -- Open model for determining "good" sources -- #
    with open('psf_quality_rf_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['clf']

    # -- Estimate quality of sources for fitting-- #
    sources_x,sources_y = wcs.all_world2pix(gaia_cat.ra,gaia_cat.dec,0)
    gaia_cat['xccd'] = sources_x
    gaia_cat['yccd'] = sources_y
    gaia_cat = gaia_cat[(gaia_cat.xccd>44+6)&(gaia_cat.xccd<44+2048-6)&
                        (gaia_cat.yccd>6)&(gaia_cat.yccd<2048-6)] 


    # gaia_mid_brightness = gaia_cat[(gaia_cat.mag < 16)&(gaia_cat.mag >9)]
    ra_rad = np.deg2rad(gaia_cat['ra'].to_numpy())
    dec_rad = np.deg2rad(gaia_cat['dec'].to_numpy())
    coords = np.column_stack((dec_rad,ra_rad))
    tree = BallTree(coords, metric='haversine')
    distances, _ = tree.query(coords, k=2)
    min_sep_rad = distances[:, 1]
    min_sep_deg = np.rad2deg(min_sep_rad)
    gaia_cat['min_dist_arcsec'] = min_sep_deg * 3600
    gaia_cat['psf_quality'] = Predict_PSF_Quality(gaia_cat.mag,gaia_cat.min_dist_arcsec,clf)

    # -- Determine corners of n cuts -- #
    intervals = 2048/n
    cut_cornersX = [44 + i*intervals for i in range(n)]
    cut_cornersY = [i*intervals for i in range(n)]
    cut_corners = np.meshgrid(cut_cornersX,cut_cornersY)
    cut_corners = np.floor(np.stack((cut_corners[0],cut_corners[1]),axis=2).reshape(n**2,2)).astype(int)
    if cut is None:
        cuts = range(1,n**2+1)
    else:
        cuts = [cut]

    # -- Iterate through cuts and fit -- # 
    sources = pd.DataFrame()
    for i,cut in enumerate(cuts):
        corner = cut_corners[i]

        cut_stars = gaia_cat[(gaia_cat.xccd>corner[0])&
                             (gaia_cat.xccd<corner[0]+intervals)&
                             (gaia_cat.yccd>corner[1])&
                             (gaia_cat.yccd<corner[1]+intervals)]

        # -- Extract at least #min_stars# stars -- #
        done = False
        cut_quality_thresh = quality_thresh
        while not done:
            good_stars = cut_stars[cut_stars.psf_quality >= cut_quality_thresh]
            if len(good_stars) >= min_stars:
                done = True
                if cut_quality_thresh != quality_thresh:
                    print(f'    reduced cut {cut} quality thresh to {cut_quality_thresh}')
            cut_quality_thresh -= 0.05

        # -- Fit -- #
        xsGaia,ysGaia,xsFit,ysFit = Fit_Cut_Stars(image,good_stars,wcs,sector,cam,ccd,cut)
        
        good_stars['xGaia'] = xsGaia
        good_stars['yGaia'] = ysGaia
        good_stars['xPSF'] = xsFit
        good_stars['yPSF'] = ysFit

        sources = pd.concat([sources,good_stars])

    return sources