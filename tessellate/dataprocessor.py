import os
import numpy as np
from time import time as t

from .tools import _Save_space, _Remove_emptys, _Extract_fits, _Print_buff
    
def _get_wcs(path,wcs_path_check,verbose=1):
    """
    Get WCS data from a file in the path
    """

    import shutil
    from glob import glob
    from astropy.wcs import WCS

    if os.path.exists(wcs_path_check):
        wcsFile = _Extract_fits(wcs_path_check)
        wcsItem = WCS(wcsFile[1].header)
    else:
        if glob(f'{path}/*ffic.fits'):
            done = False
            i = 0
            while not done:
                filepath = glob(f'{path}/*ffic.fits')[i]
                file = _Extract_fits(filepath)
                wcsItem = WCS(file[1].header)
                file.close()
                if wcsItem.get_axis_types()[0]['coordinate_type'] == 'celestial':
                    done = True
                    shutil.copy2(filepath,wcs_path_check)
                else:
                    i += 1
        else:
            if verbose>0:
                print('No Data!')
            return

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

    from astrocut import CutoutFactory

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

    def download(self,cam,ccd,number='all',time=None):
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

        from .downloader import Download_cam_ccd_FFIs
        
        if self.verbose > 0:
            print(_Print_buff(50,f'Downloading Sector {self.sector} Cam {cam} CCD {ccd}'))
        Download_cam_ccd_FFIs(self.path,self.sector,cam,ccd,time,None,None,number=number) 
    
    def find_cuts(self,cam,ccd,n,plot=True,proj=True,coord=None,verbose=1):
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

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        newpath = f'{self.path}/Cam{cam}/Ccd{ccd}'
        wcsItem = _get_wcs(f'{newpath}/image_files',f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits') 
        # if not os.path.exists(f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits'):
        #     wcs_save = wcsItem.to_fits()
        #     wcs_save.writeto(f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits')

        if wcsItem is None:
            if verbose > 0:
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
    
    def _make_part_cube(self,cam,ccd,input_files):

        from astrocut import CubeFactory
        from astropy.time import Time
        import datetime

        # -- Generate Cube Path -- #
        broad_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        
        dates = []
        for f in input_files:
            year = f[56:60]
            daynum = f[60:63]
            hour = f[63:65]
            minute = f[65:67]
            sec = f[67:69]
            date = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j")
            month = date.month
            day = date.day
            imagetime = '{}-{}-{}T{}:{}:{}'.format(year,month,day,hour,minute,sec)
            imagetime = Time(imagetime, format='isot', scale='utc').mjd
            dates.append(imagetime)

        sortedDates = np.array(sorted(dates))
        differences = np.diff(sortedDates)
        idx = np.where(differences == np.nanmax(differences))[0][0]+1

        cube1_files = []
        cube2_files = []
        for f in input_files:
            year = f[56:60]
            daynum = f[60:63]
            hour = f[63:65]
            minute = f[65:67]
            sec = f[67:69]
            date = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j")
            month = date.month
            day = date.day
            imagetime = '{}-{}-{}T{}:{}:{}'.format(year,month,day,hour,minute,sec)
            imagetime = Time(imagetime, format='isot', scale='utc').mjd
            if imagetime in sortedDates[:idx]:
                cube1_files.append(f)
            elif imagetime in sortedDates[idx:]:
                cube2_files.append(f)  
        cubes = [cube1_files,cube2_files]

        for i in range(2):
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            if not os.path.exists(f'{broad_path}/Part{i+1}'):
                os.mkdir(f'{broad_path}/Part{i+1}')
            cube_path = f'{broad_path}/Part{i+1}/{cube_name}'

            if self.verbose > 0:
                print(_Print_buff(50,f'Cubing Sector {self.sector} Cam {cam} CCD {ccd} Part {i+1}'))

            # -- Make Cube -- #
            cube_maker = CubeFactory()
            cube_file = cube_maker.make_cube(cubes[i],cube_file=cube_path,verbose=self.verbose>1,max_memory=500)
            print('\n')


    def make_cube(self,cam,ccd,part=False):
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

        from astrocut import CubeFactory
        from glob import glob


        # -- Generate Cube Path -- #
        broad_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        file_path = f'{broad_path}/image_files'

        # if os.path.exists(cube_path):
        #     print(f'Cam {cam} CCD {ccd} cube already exists!')
        #     return

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

        if part:
            self._make_part_cube(cam,ccd,input_files)
        else:
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{broad_path}/{cube_name}'

            # -- Allows for a custom cubing message (kinda dumb) -- #
            if self.verbose > 0:
                print(_Print_buff(50,f'Cubing Sector {self.sector} Cam {cam} CCD {ccd}'))
            
            # -- Make Cube -- #
            cube_maker = CubeFactory()
            cube_file = cube_maker.make_cube(input_files,cube_file=cube_path,verbose=self.verbose>1,max_memory=500)

    def _make_part_cuts(self,cam,ccd,n,cut,file_path,cutCentreCoords, cutSize):

        from astrocut import CutoutFactory

        for i in range(2):

            # -- Generate Cube Path -- #
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{file_path}/Part{i+1}/{cube_name}'
        
            name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            if self.verbose > 0:
                print(f'Cutting Cam {cam} CCD {ccd} Cut {cut} (of {n**2}) Part {i+1}')
        
            my_cutter = CutoutFactory() # astrocut class
            coords = cutCentreCoords[cut-1]

            _Save_space(f'{file_path}/Part{i+1}/Cut{cut}of{n**2}')
                        
            # -- Cut -- #
            self.cut_file = my_cutter.cube_cut(cube_path, 
                                                f"{coords[0]} {coords[1]}", 
                                                (cutSize*2,cutSize*2), 
                                                output_path = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}',
                                                target_pixel_file = name,
                                                verbose=(self.verbose>1)) 

            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd} Cut {cut} Part {i+1} complete.')
                print('\n')

            with open(f'{file_path}/Part{i+1}/Cut{cut}of{n**2}/cut.txt', 'w') as file:
                file.write('Cut!')
    
    def make_cuts(self,cam,ccd,n,cut,part=False):
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

        from astrocut import CutoutFactory

        try:
            _, _, cutCentreCoords, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False)
        except:
            print('Something wrong with finding cuts!')
            return
        
        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        if not os.path.exists(file_path):
            print('No data to cut!')
            return
        
        if part:
            self._make_part_cuts(cam,ccd,n,cut,file_path,cutCentreCoords,cutSize)
        else:   
            # -- Generate Cube Path -- #
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{file_path}/{cube_name}'
            
            name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            # if os.path.exists(f'{file_path}/Cut{cut}of{n**2}/{name}'):
            #     print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
            # else:
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

    def _reduce_part_cuts(self,cam,ccd,n,cut,filepath):

        for i in range(2):
            cutFolder = f'{filepath}/Part{i+1}/Cut{cut}of{n**2}'
            cutName = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            cutPath = f'{cutFolder}/{cutName}'
            fluxName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy'
            if os.path.exists(fluxName):
                if self.verbose > 0:
                    print(f'Cam {cam} Chip {ccd} Cut {cut} Part {i+1} already reduced!')
            else:
                ts = t()
                if self.verbose > 0:
                    print(f'--Reduction Cam {cam} Chip {ccd} Cut {cut} (of {n**2}) Part {i+1} --')

                # -- Defining so can be deleted if failed -- #
                tessreduce = 0

                # -- reduce -- #
                tessreduce = tr.tessreduce(tpf=cutPath,sector=self.sector,reduce=True,corr_correction=True,
                                            calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',
                                            prf_path='/fred/oz335/_local_TESS_PRFs')
                
                if self.verbose > 0:
                    print(f'--Reduction Part {i+1} Complete (Time: {((t()-ts)/60):.2f} mins)--')
                    print('\n')
                #tw = t()   # write timeStart
                
                # -- Saves information out as Numpy Arrays -- #
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Times.npy',tessreduce.lc[0])
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy',tessreduce.flux)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Background.npy',tessreduce.bkg)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Ref.npy',tessreduce.ref)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Mask.npy',tessreduce.mask)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Shifts.npy',tessreduce.shift)

                del (tessreduce)

                with open(f'{cutFolder}/reduced.txt', 'w') as file:
                    file.write(f'Reduced with TESSreduce version {tr.__version__}.')


    def reduce(self,cam,ccd,n,cut,part=False):
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
            n**2 part cuts

        -------
        Creates
        -------
        Fits file in path with reduced data.

        """

        import tessreduce as tr
        
        filepath = f'{self.path}/Cam{cam}/Ccd{ccd}'

        if part:
            self._reduce_part_cuts(cam,ccd,n,cut,filepath)
        else:
            cutFolder = f'{filepath}/Cut{cut}of{n**2}'
            cutName = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            cutPath = f'{cutFolder}/{cutName}'

            # fluxName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy'

            # if os.path.exists(fluxName):
            #     if self.verbose > 0:
            #         print(f'Cam {cam} Chip {ccd} cut {cut} already reduced!')
            # else:
            ts = t()
            if self.verbose > 0:
                print(f'--Reduction Cam {cam} Chip {ccd} Cut {cut} (of {n**2})--')
                
            # -- Defining so can be deleted if failed -- #
            tessreduce = 0

            # -- reduce -- #
            tessreduce = tr.tessreduce(tpf=cutPath,sector=self.sector,reduce=True,corr_correction=True,
                                        calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',
                                        prf_path='/fred/oz335/_local_TESS_PRFs')
            
            if self.verbose > 0:
                print(f'--Reduction Complete (Time: {((t()-ts)/60):.2f} mins)--')
                print('\n')
            #tw = t()   # write timeStart
            
            # -- Saves information out as Numpy Arrays -- #
            np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Times.npy',tessreduce.lc[0])
            np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy',tessreduce.flux)
            np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Background.npy',tessreduce.bkg)
            np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Ref.npy',tessreduce.ref)
            np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Mask.npy',tessreduce.mask)
            np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Shifts.npy',tessreduce.shift)

            del (tessreduce)