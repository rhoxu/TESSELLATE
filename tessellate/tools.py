import numpy as np
import os

def _remove_ffis(data_path,sector,n,cams,ccds,cuts):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}')
            os.system('rm -r -f image_files')

    os.chdir(home_path)

def _remove_cubes(data_path,sector,n,cams,ccds,cuts):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}')
            os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_cube.fits')
            os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_wcs.fits')
            os.system(f'rm -f cubed.txt')

    os.chdir(home_path)

def _remove_cuts(data_path,sector,n,cams,ccds,cuts):

    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                os.system(f'rm -r -f {data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')

def _remove_reductions(data_path,sector,n,cams,ccds,cuts):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                os.system(f'rm -f *.npy')
                os.system(f'rm -f reduced.txt')
                os.system(f'rm -f detected_events.csv')
                os.system(f'rm -f detected_sources.csv')
                os.system('rm -r -f figs')
                os.system('rm -r -f lcs')    

    os.chdir(home_path)

def _remove_search(data_path,sector,n,cams,ccds,cuts):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                os.system(f'rm -f detected_events.csv')
                os.system(f'rm -f detected_sources.csv')
                #os.system('rm -r -f figs')
                #os.system('rm -r -f lcs')    

    os.chdir(home_path)
    
def _remove_plots(data_path,sector,n,cams,ccds,cuts):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                os.system('rm -r -f figs')
                os.system('rm -r -f lcs')    

    os.chdir(home_path)

def delete_files(filetype,data_path,sector,n=4,cams='all',ccds='all',cuts='all'):

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
        function(data_path,sector,n,cams,ccds,cuts)
    else:
        e = 'Invalid filetype! Valid types: "ffis" , "cubes" , "cuts" , "reductions" , "search", "plot". '
        raise AttributeError(e)