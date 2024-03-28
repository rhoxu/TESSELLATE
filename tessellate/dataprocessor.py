import os
from glob import glob

import tessreduce as tr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import multiprocessing
from joblib import Parallel, delayed 

from time import time as t

from astrocut import CubeFactory
from astrocut import CutoutFactory
from astropy.io import fits
from astropy import wcs

from .downloader import Download_cam_ccd_FFIs

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

def _get_wcs(path):
    """
    Get WCS data from a file in the path
    """

    if glob(f'{path}/*ffic.fits'):
        filepath = glob(f'{path}/*ffic.fits')[0]
    else:
        print('No Data!')
        return
    file = _Extract_fits(filepath)
    wcsItem = wcs.WCS(file[1].header)
    file.close()
    return wcsItem

def _cut_properties(wcsItem,n):

        intervals = 2048/n

        cutCornersX = [44 + i*intervals for i in range(n)]
        cutCornersY = [i*intervals for i in range(n)]
        cutCorners = np.meshgrid(cutCornersX,cutCornersY)
        cutCorners = np.floor(np.stack((cutCorners[0],cutCorners[1]),axis=2).reshape(n**2,2))

        intervals = np.ceil(intervals)
        rad = np.ceil(intervals / 2)

        cutCentrePx = cutCorners + rad
        cutCentreCoords = np.array(wcsItem.all_pix2world(cutCentrePx[:,0],cutCentrePx[:,1],0)).transpose()

        return cutCorners,cutCentrePx,cutCentreCoords,rad

def _parallel_cuts(sector,cam,ccd,cut,n,cube_path,file_path,coords,size,verbose):

    name = f'sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
    if os.path.exists(f'{file_path}/Cut{cut}of{n**2}/{name}'):
        print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
    else:
        if verbose > 0:
            print(f'Cutting Cam {cam} CCD {ccd} cut {cut} (of {n**2})')
        
        my_cutter = CutoutFactory() # astrocut class
        _Save_space(f'{file_path}/Cut{cut}of{n**2}')

        # -- Cut -- #
        cut_file = my_cutter.cube_cut(cube_path, 
                                        f"{coords[0]} {coords[1]}", 
                                        (size,size), 
                                        output_path = f'{file_path}/Cut{cut}of{n**2}',
                                        target_pixel_file = name,
                                        verbose=(verbose>1)) 

        if verbose > 0:
            print(f'Cam {cam} CCD {ccd} cut {cut} complete.')
            print('\n')

class DataProcessor():

    def __init__(self,sector,verbose=1,path=None) -> None:

        self.sector = sector
        self.verbose = verbose

        if path[-1] == '/':
            path = path[:-1]
        self.path = path

        self._make_path(False)

    def _make_path(self,delete):
        """
        Creates a folder for the path. 
        """

        if self.path is None:
            _Save_space('temporary',delete=delete)
            self.path = './temporary'
        else:
            _Save_space(f'{self.path}/Sector{self.sector}')
            self.path = f'{self.path}/Sector{self.sector}'

    def download(self,cam=None,ccd=None,number='all',time=None):
        """
        Function for downloading FFIs from MAST archive.

        ------
        Inputs
        ------
        cam : int
            specific camera, default None
        ccd : int
            desired ccd, default None

        -------
        Options:
        -------
        number : int
            if not None, downloads this many
        time : float (MJD)
            if not None, downloads only FFIs within a day of this time
        
        """

        combinations = []
        if cam is not None:
            if ccd is not None:
                combinations.append((cam,ccd))
            else:
                combinations.extend([(cam,ccd) for ccd in range(1,5)])
        else:
            for cam in range(1,5):
                combinations.extend([(cam,ccd) for ccd in range(1,5)])
        
        if time is not None:  # sets number of days before and after given time to download inside
            lower = 1
            upper = 1
        else:
            lower = None
            upper = None
        
        for cam,ccd in combinations:
            if self.verbose > 0:
                print('\n')
                print(_Print_buff(50,f'Downloading Sector {self.sector} Cam {cam} CCD {ccd}'))
            Download_cam_ccd_FFIs(self.path,self.sector,cam,ccd,time,lower,upper,number=number) 
    
    def find_cuts(self,cam,ccd,n,plot=True,proj=True,coord=None):
        """
        Function for finding cuts.

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts will be made

        -------
        Options:
        -------
        plot : bool
            if True, plot cuts 
        proj : bool
            if True, plot cuts with WCS grid
        coord : (ra,dec)
            if not None, plots coord 
        
        """

        newpath = f'{self.path}/Cam{cam}/Ccd{ccd}'
        wcsItem = _get_wcs(newpath)

        if wcsItem is None:
            print('WCS Extraction Failed')
            return

        cutCorners, cutCentrePx, cutCentreCoords, cutSize = _cut_properties(wcsItem,n)

        if plot:
            # -- Plots data -- #
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            
            if proj:
                ax = plt.subplot(projection=wcsItem)
                ax.set_xlabel(' ')
                ax.set_ylabel(' ')
            else:
                ax = plt.subplot()

            if coord is not None:
                coordPx = wcsItem.all_world2pix(coord[0],coord[1],0)
                ax.scatter(coordPx[0],coordPx[1],s=10)
                ax.text(x=coordPx[0],y=coordPx[1]+10,s=f'{coordPx[0]:.1f},{coordPx[1]:.1f}')
            
            # -- Real rectangle edge -- #
            rectangleTotal = patches.Rectangle((44,0), 2048, 2048,edgecolor='black',facecolor='none',alpha=0.5)
            
            # -- Sets title -- #
            ax.set_title(f'Camera {cam} CCD {ccd}')
            ax.set_xlim(0,2136)
            ax.set_ylim(0,2078)
            ax.grid()

            ax.add_patch(rectangleTotal)
                
            # -- Adds cuts -- #
            colours = iter(plt.cm.rainbow(np.linspace(0, 1, n**2)))

            for corner in cutCorners:
                c = next(colours)
                rectangle = patches.Rectangle(corner,2*cutSize,2*cutSize,edgecolor=c,
                                              facecolor='none',alpha=1)
                ax.add_patch(rectangle)
                
        return cutCorners, cutCentrePx, cutCentreCoords, cutSize

    def make_cube(self,cam,ccd,delete_files=False):
        """
        Make cube for this cam,ccd.
        
        ------
        Inputs
        ------
        cam : int
            desired camera 
        ccd : int
            desired ccd

        -------
        Options
        -------
        delete_files : bool  
            deletes all FITS files once cube is made
        cubing_message : str
            custom printout message for self.verbose > 0

        -------
        Creates
        -------
        Data cube fits file in path.

        """

        # -- Generate Cube Path -- #
        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
        cube_path = f'{file_path}/{cube_name}'

        if os.path.exists(cube_path):
            print(f'Cam {cam} CCD {ccd} cube already exists!')
            return

        input_files = glob(f'{file_path}/*ffic.fits')  # list of fits files in path
        if len(input_files) < 1:
            print('No files to cube!')
            return  
        
        deleted = _Remove_emptys(input_files)  # remove empty downloaded fits files
        if self.verbose > 1:
            print(f'Deleted {deleted} corrupted file/s.')
                    
        input_files = glob(f'{file_path}/*ffic.fits')  # regather list of good fits files
        if len(input_files) < 1:
            print('No files to cube!')
            return    

        if self.verbose > 1:
            print(f'Number of files to cube = {len(input_files)}')
            size = len(input_files) * 0.0355
            print(f'Estimated cube size = {size:.2f} GB')

        # -- Allows for a custom cubing message (kinda dumb) -- #
        if self.verbose > 0:
            print(_Print_buff(50,f'Cubing Sector {self.sector} Cam {cam} CCD {ccd}'))
        
        # -- Make Cube -- #
        cube_maker = CubeFactory()
        cube_file = cube_maker.make_cube(input_files,cube_file=cube_path,verbose=self.verbose>1,max_memory=500)

        # -- If true, delete files after cube is made -- #
        if delete_files:
            homedir = os.getcwd()
            os.chdir(file_path)
            os.system('rm *ffic.fits')
            os.chdir(homedir)
    
    def make_cuts(self,cam,ccd,n,cut):
        """
        Make cut(s) for this CCD.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd

        ------
        Creates
        ------
        Save files for cut(s) in path.

        """
        try:
            _, _, cutCentreCoords, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False)
        except:
            print('Something wrong with finding cuts!')
            return
        
        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        if not os.path.exists(file_path):
            print('No data to cut!')
            return
                
        # -- Generate Cube Path -- #
        cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
        cube_path = f'{file_path}/{cube_name}'
        
        name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
        if os.path.exists(f'{file_path}/Cut{cut}of{n**2}/{name}'):
            print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
        else:
            if self.verbose > 0:
                print(f'Cutting Cam {cam} CCD {ccd} cut {cut} (of {n**2})')
            
            my_cutter = CutoutFactory() # astrocut class
            coords = cutCentreCoords[cut-1]

            _Save_space(f'{file_path}/Cut{cut}of{n**2}')
                            
            # -- Cut -- #
            self.cut_file = my_cutter.cube_cut(cube_path, 
                                                f"{coords[0]} {coords[1]}", 
                                                (cutSize*2,cutSize*2), 
                                                output_path = f'{file_path}/Cut{cut}of{n**2}',
                                                target_pixel_file = name,
                                                verbose=(self.verbose>1)) 

            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd} cut {cut} complete.')
                print('\n')

    def reduce(self,cam,ccd,n,cut):
        """
        Reduces a cut on a ccd using TESSreduce. bkg correlation 
        correction and final calibration are disabled due to time constraints.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 split cuts

        -------
        Creates
        -------
        Fits file in path with reduced data.

        """
        
        filepath = f'{self.path}/Cam{cam}/Ccd{ccd}'

        cutFolder = f'{filepath}/Cut{cut}of{n**2}'
        cutName = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
        cutPath = f'{cutFolder}/{cutName}'

        fluxName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy'
        maskName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Mask.npy'
        timesName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Times.npy'

        if os.path.exists(fluxName):
            if self.verbose > 0:
                print(f'Cam {cam} Chip {ccd} cut {cut} already reduced!')
        else:
            ts = t()
            if self.verbose > 0:
                print(f'--Reduction Cam {cam} Chip {ccd} Cut {cut} (of {n**2})--')
            
            try: 
                # -- Defining so can be deleted if failed -- #
                tessreduce = 0

                # -- reduce -- #
                tessreduce = tr.tessreduce(tpf=cutPath,sector=self.sector,reduce=True,corr_correction=False,
                                            calibrate=False,catalogue_path=cutFolder)
                
                if self.verbose > 0:
                    print(f'--Reduction Complete (Time: {((t()-ts)/60):.2f} mins)--')
                    print('\n')
                #tw = t()   # write timeStart
                
                # -- Saves information out as Numpy Arrays -- #
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Times.npy',tessreduce.lc[0])
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy',tessreduce.flux)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Mask.npy',tessreduce.mask)

                del (tessreduce)

            except:
                # -- Deletes Memory -- #
                del(tessreduce)

                if self.verbose > 0:
                    print(f'Reducing Cam {cam} Chip {ccd} Cut {cut} Failed :( Time Elapsed: {((t()-ts)/60):.2f} mins.')
                    print('\n')
                pass 