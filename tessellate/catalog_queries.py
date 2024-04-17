import pandas as pd
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astroquery.vizier import Vizier
import numpy as np
from astropy.io import fits 
from astropy.wcs import WCS 
from sklearn.cluster import DBSCAN
 
def cross_match(obs_cat, viz_cat,tol=2*21,variable=True):
    dist = np.sqrt((obs_cat.ra[:,np.newaxis] - viz_cat.ra[np.newaxis,:])**2 + (obs_cat.dec[:,np.newaxis] - viz_cat.dec[np.newaxis,:])**2)
    closest = np.argmin(dist,axis=1)
    valid = np.nanmin(dist,axis=1) < tol/60**2
    matched = closest[valid]
    if variable:
        obs_cat['Type'] = 'none'
        obs_cat['Prob'] = 0
        obs_cat['Type'].iloc[valid] = viz_cat['Type'].iloc[matched]
        obs_cat['Prob'].iloc[valid] = viz_cat['Prob'].iloc[matched]
    else:
        obs_cat['GaiaID'] = 0 
        obs_cat['GaiaID'].iloc[valid] = viz_cat['Source'].iloc[matched]
    return obs_cat
 
def get_catalog(catalog, centre, width, height,gaia=False):
    coords = SkyCoord(ra=centre[0]*u.deg,
                      dec=centre[1]*u.deg)
    v = Vizier(row_limit=-1)
    if gaia:
        t_result = v.query_region(coords, width=width*u.deg,
                                  height=height*u.deg,
                                  catalog=catalog,
                                  column_filters={'Gmag':'<19.5'})
    else:
        t_result = v.query_region(coords, width=width*u.deg,
                                  height=height*u.deg,
                                  catalog=catalog)
 
    if len(t_result) > 0:
        return t_result[catalog].to_pandas()
    else:
        return None
 
def find_variables(coords,viz_cat,width,height):
    # Fetch and save catalogs from Vizier.
    # Gaia Variable Catalog
    varisum = get_catalog("I/358/varisum",coords,width,height)
    # ASASN Variable Catalog
    asasn = get_catalog("II/366/catalog",coords,width,height)
    # DES RRLyrae Catalog
    des_var = get_catalog("J/AJ/158/16/table11",coords,width,height)
    if (varisum is None) & (asasn is None) & (des_var is None):
        print("No known variables found ...")
        varcat = [0]
    else:
        if varisum is not None:
            varisum['Type'] = 'None'
            varisum['Prob'] = 0
            keys = list(varisum.keys())[12:-2]
            for key in keys:
                t = varisum[key].values > 0
                varisum['Type'].iloc[t] = key
                varisum['Prob'].iloc[t] = varisum[key].values[t]
            varisum = varisum.rename(columns={'RA_ICRS': 'ra',
                                              'DE_ICRS': 'dec'})
            varisum = pd.DataFrame(varisum, columns=['ra','dec','Type','Prob'])
 
        if asasn is not None:
            asasn = asasn.rename(columns={'RAJ2000': 'ra',
                                          'DEJ2000': 'dec'})
            asasn = pd.DataFrame(asasn, columns=['ra','dec','Type','Prob'])
        else:
            asasn = None
 
        if des_var is not None:
            des_var = des_var.rename(columns={'RAJ2000': 'ra',
                                              'DEJ2000': 'dec'})
            des_var = pd.DataFrame(des_var, columns=['ra','dec'])
            des_var['Type'] = 'DES_var'
            des_var['Prob'] = 1
        else:
            des_var = None
 
        variables = pd.concat([varisum, asasn, des_var])
    
    obs = cross_match(viz_cat, variables)
    return obs

def gaia_stars(obs_cat,size=2*21,mag_limit=19.5):
    coords = SkyCoord(ra=obs_cat.ra.values*u.deg,
                      dec=obs_cat.dec.values*u.deg)
    v = Vizier(row_limit=-1)
    gaia = v.query_region(coords, catalog=["I/355/gaiadr3"],
                                 radius=Angle(size, "arcsec"),column_filters={'Gmag':f'<{mag_limit}'})
    gaia = gaia['I/355/gaiadr3'].to_pandas()
    gaia = gaia.rename(columns={'RA_ICRS': 'ra',
                                'DE_ICRS': 'dec'})
    obs = cross_match(obs_cat,gaia,tol=size,variable=False)
    return obs 

####### Download variables

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

def get_variable_cats(coords,width,height):
    varisum = get_catalog("I/358/varisum",coords,width,height)
    # ASASN Variable Catalog
    asasn = get_catalog("II/366/catalog",coords,width,height)
    # DES RRLyrae Catalog
    des_var = get_catalog("J/AJ/158/16/table11",coords,width,height)
    if (varisum is None) & (asasn is None) & (des_var is None):
        print("No known variables found ...")
        varcat = [0]
    else:
        if varisum is not None:
            varisum['Type'] = 'None'
            varisum['Prob'] = 0
            keys = list(varisum.keys())[12:-2]
            for key in keys:
                t = varisum[key].values > 0
                varisum['Type'].iloc[t] = key
                varisum['Prob'].iloc[t] = varisum[key].values[t]
            varisum = varisum.rename(columns={'RA_ICRS': 'ra',
                                              'DE_ICRS': 'dec'})
            varisum = pd.DataFrame(varisum, columns=['ra','dec','Type','Prob'])
 
        if asasn is not None:
            asasn = asasn.rename(columns={'RAJ2000': 'ra',
                                          'DEJ2000': 'dec'})
            asasn = pd.DataFrame(asasn, columns=['ra','dec','Type','Prob'])
        else:
            asasn = None
 
        if des_var is not None:
            des_var = des_var.rename(columns={'RAJ2000': 'ra',
                                              'DEJ2000': 'dec'})
            des_var = pd.DataFrame(des_var, columns=['ra','dec'])
            des_var['Type'] = 'RRLyrae'
            des_var['Prob'] = 1
        else:
            des_var = None
 
        variables = pd.concat([varisum, asasn, des_var])
    return variables

def create_external_var_cat(image_path,save_path):
    file = _Extract_fits(image_path)
    wcsItem = WCS(file[1].header)
    file.close()
    center = wcsItem.all_pix2world(wcsItem.pixel_shape[1]/2,wcsItem.pixel_shape[0]/2,0)
    corners = wcsItem.calc_footprint()
    ra = corners[:,1]
    dec = corners[:,1]
    dist = np.max(np.sqrt((ra-center[0])**2+(dec-center[1])**2)) + 1/60
    varcat = get_variable_cats(center,dist,dist)
    varcat.to_csv(save_path+'/variable_catalog.csv',index=False)


def cross_match_DB(cat1,cat2,distance=2*21,njobs=-1):
    all_ra = np.append(cat1['ra'].values,cat2['ra'].values)
    all_dec = np.append(cat1['dec'].values,cat2['dec'].values)
    cat2_ind = len(cat1)

    p = np.array([all_ra,all_dec]).T
    cluster = DBSCAN(eps=distance/60**2,min_samples=2,n_jobs=njobs).fit(p)
    labels = cluster.labels_
    unique_labels = set(labels)
    cat1_id = []
    cat2_id = []

    for label in unique_labels:
        if label > -1:
            inds = np.where(labels == label)[0]
            if (inds < cat2_ind).any():
                if len(inds) > 2:
                    dra = all_ra[np.where(labels == label)[0]]
                    ddec = all_dec[np.where(labels == label)[0]]
                    d = (dra - dra[0])**2 + (ddec - ddec[0])**2
                    args = np.argsort(d)
                    inds = inds[args[:2]]
                cat1_id += [inds[0]]
                cat2_id += [inds[1] - cat2_ind]
    return cat1_id,cat2_id

def match_result_to_cat(result,cat,columns=['Type','Prob'],distance=2*21,min_entry=2):
    ids = np.unique(result['objid'].values)
    ra = []; dec = []
    Id = []
    for id in ids:
        if len(result.iloc[result['objid'].values == id]) >=min_entry:
            Id += [id]
            ra += [result.loc[result['objid'] == id, 'ra'].mean()]
            dec += [result.loc[result['objid'] == id, 'dec'].mean()]
    pos = {'objid':Id,'ra':ra,'dec':dec}
    id_ind, cat_ind = cross_match_DB(pos,cat,distance)
    obj_id = ids[id_ind]
    for column in columns:
        result[column] = 0
    for i in range(len(obj_id)):
        for column in columns:
            result.loc[result['objid']==obj_id[i],column] = cat[column].values[cat_ind[i]]
    return result 