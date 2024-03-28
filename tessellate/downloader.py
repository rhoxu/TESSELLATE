import os
import datetime
from astropy.time import Time
from glob import glob
import multiprocessing
from joblib import Parallel, delayed 
from tqdm import tqdm
import numpy as np

def _find_lines(fileline,cam,ccd,time,lower,upper):
    """
    Trawls through base file to find FFI files that are on the right camera/ccd combination.
    Also has functionality for adding specific time intervals.
    """
     
    if "-{}-{}-".format(cam,ccd) in fileline:

        if time is not None:
            year = fileline[20:24]
            daynum = fileline[24:27]
            hour = fileline[27:29]
            minute = fileline[29:31]
            sec = fileline[31:33]
            date = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j")
            month = date.month
            day = date.day
            imagetime = '{}-{}-{}T{}:{}:{}'.format(year,month,day,hour,minute,sec)
            imagetime = Time(imagetime, format='isot', scale='utc').mjd
            
            if time-lower <= imagetime <= time+upper:
                return f'{fileline[:4]} --silent {fileline[5:]}'
            else:
                return None
        else:
            return f'{fileline[:4]} --silent {fileline[5:]}'
    else:
        return None
    
def _download_line(fileline):
    """
    Download fileline, used for parallel.
    """

    os.system(fileline)

def Download_cam_ccd_FFIs(path,sector,cam,ccd,time,lower,upper,number):
    """
    Downloads FFIs of interest (each ~34 MB).
    """
    homepath = os.getcwd() 

    if not glob(f'{path}/*_ffic.sh'):    # gets base download file from MAST
        os.chdir(path)
        os.system(f'curl --silent -O https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_ffic.sh')
        os.chdir(homepath)

    with open(f'{path}/tesscurl_sector_{sector}_ffic.sh') as file:
        filelines = file.readlines()

    # -- If base path is not temporary, creates a subdirectory -- #
    newpath = f'{path}/Cam{cam}/Ccd{ccd}/' 
    
    if not os.path.exists(f'{path}/Cam{cam}'):
        os.mkdir(f'{path}/Cam{cam}')
    if not os.path.exists(newpath):
        os.mkdir(newpath)

    os.chdir(newpath)    
    print(os.getcwd())
    
    # -- Collates all lines for FFIs of interest -- #
    goodlines = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(_find_lines)(file,cam,ccd,time,lower,upper) for file in filelines)
    goodlines = np.array(goodlines)
    goodlines = goodlines[goodlines!=None]
    
    # -- If all are to be downloaded, download in parallel -- #
    if number == 'all':
        inds = np.linspace(0,len(goodlines)-1,len(goodlines)).astype(int)
        Parallel(n_jobs=multiprocessing.cpu_count())(delayed(_download_line)(goodlines[ind]) for ind in tqdm(inds, position=0, leave=True))
    elif number > 10:
        inds = np.linspace(0,number-1,number).astype(int)
        Parallel(n_jobs=multiprocessing.cpu_count())(delayed(_download_line)(goodlines[ind]) for ind in tqdm(inds, position=0, leave=True))
    else:
        for i in range(number):     # download {number} files from the goodlines
            _download_line(goodlines[i])
            print(f'Done {i+1}', end='\r')

    os.chdir(homepath)