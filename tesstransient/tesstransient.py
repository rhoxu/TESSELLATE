import os
from glob import glob

import tessreduce as tr

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import scipy

from time import time as t

from astropy.time import Time

from tessellate import Tessellate
from tessellate.dataprocessor import DataProcessor,_get_wcs
from tessellate.detector import Detector

from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
    
class TessTransient():
    
    def __init__(self,ra,dec,eventtime,data_path,job_output_path,working_path,error=0,n=4,verbose=1,run=False):
        """
        Transient Localisation for TESS!

        ------
        Inputs
        ------
        ra : float
            degree right ascension of event estimation
        dec : float
            degree declination of event estimation
        eventtime : float 
            MJD reference time for event
        error : float
            degree 1 dimensional error for event

        -------
        Options:
        -------
        verbose : int  
            0 = no printout , 1 (default) = partial printout, 2 = full printout
        path : str
            location of folder, creates temporary folder if None 
        eventname : str
            name of event to be encoded into downloads
        delete : bool
            deletes previous temporary folder if True
        
        """

        # Given
        self.ra = ra
        self.dec = dec
        self.error = error
        self.eventtime = eventtime
        self.verbose = verbose
        self.n = n

        self.data_path = data_path
        self.job_output_path = job_output_path
        self.working_path = working_path

        print('\n')
        self._location_observed()
        _ = self._sector_suggestions()

        self.ErrorEllipse = None
        self.neighbours = None

        if self.obs & run:
            print('\n')
            self.run()

    def _location_observed(self):
        """
        Check whether any part of error region is on a TESS CCD. 
        """

        obj = tr.spacetime_lookup(self.ra,self.dec,self.eventtime,print_table=self.verbose>1)   # look up location using TESSreduce - requires custom 

        obs = False
        for sector in obj:
            if sector[-1] == True:
                if self.verbose>0:
                    print('Event occured within TESS FoV!')
                self.sector = sector[2]
                self.cam = sector[3]    # only returned in custom TR package
                self.ccd = sector[4]   # only returned in custom TR package
                obs=True
       
        if not obs:
            obs = self._check_surroundings()    # check if part of error region observed but not estimate itself

        self.obs = obs
       
        if not obs:
            if self.verbose > 0:
                print('Event did not occur within TESS FoV :(')
            self.sector = 'Event did not occur within TESS FoV :('
            self.cam = 'Event did not occur within TESS FoV :('
            self.ccd = 'Event did not occur within TESS FoV :('
            return
            
    def _check_surroundings(self):
        """
        Check if error region overlaps with TESS CCD.
        """

        # -- Generate a bunch of coords in all directions around RA,DEC -- #
        distances = np.arange(2,self.error,2)
        distances = np.append(distances,self.error)
        raEll = []
        decEll = []
        for jj in np.linspace(0,2*np.pi-np.pi/4,8):
            for d in distances:
                raEll.append(self.ra + d*np.cos(jj))
                decEll.append(self.dec + d*np.sin(jj))
        coords = np.array(tuple(zip(raEll,decEll)))

        # -- Check if any of the coords overlap with TESS -- #
        goodcoords = []
        for coord in coords:
            try: 
                obj = tr.spacetime_lookup(coord[0],coord[1],self.eventtime,print_table=False)
                for sector in obj:
                    if sector[-1] == True:
                        goodcoords.append(coord)
            except:
                pass
        goodcoords = np.array(goodcoords)

        # -- If any do, find whether the RA,DEC is outside border of CCDs, or in a gap between CCDs -- #
        obs = False
        if len(goodcoords) != 0:
            goodestcoords = np.array(goodcoords)

            grb_point = np.array((165,26))
            grb_point = np.reshape(grb_point,(1,2))
            dist = scipy.spatial.distance.cdist(goodestcoords,grb_point)
            min_dist = min(dist)[0]
            
            meanRA = np.mean(goodestcoords[:,0])
            meanDEC = np.mean(goodestcoords[:,1])
            
            mean_point = np.array((meanRA,meanDEC))
            mean_point = np.reshape(mean_point,(1,2))
            
            meandist = math.dist(grb_point[0],mean_point[0])

            between = False
            if meandist < min_dist:    # if the mean distance is less than the minimum, event is between ccds 
                between = True

                obj = tr.spacetime_lookup(goodcoords[int(len(goodcoords)/2)][0],goodcoords[int(len(goodcoords)/2)][1],self.eventtime,print_table=False)
                for sector in obj:
                    if sector[-1] == True:
                        self.sector = sector[2]
                        self.cam = sector[3]
                        self.ccd = sector[4]
                        obs=True

            else:       # get an estimate on how much of error region is on CCDs

                percentage = ((min_dist)/(2*3.2))*100
                                # if min_dist == 2:
                                #     min_dist = '<2'

                                #     print('Nearest Point = <2 degrees ({:.1f}% of {}σ error)'.format(percentage,2))
                                # else:
                                #     print('Nearest Point = {:.1f} degrees ({:.1f}% of {}σ error)'.format(min_dist,percentage,2))

                obj = tr.spacetime_lookup(meanRA,meanDEC,self.eventtime,print_table=False)
                for sector in obj:
                    if sector[-1] == True:
                        self.sector = sector[2]
                        self.cam = sector[3]
                        self.ccd = sector[4]
                        obs=True       

            if self.verbose > 0:
                if between:
                    print('Event located between ccds')
                else:
                    print('Part of error region observed!') 

        return obs 

    def _secure_err_px(self,wcsItem,raEll,decEll):

        xs = []
        ys = []
        for i in range(len(raEll)):
            try:
                coord = wcsItem.all_world2pix(raEll[i],decEll[i],0)
                xs.append(coord[0])
                ys.append(coord[1])
            except:
                pass
        return [xs,ys]
    
    def _find_impacted_cuts(self,ellipse,cutCorners,cutSize,cutCentrePx):

        intersects = []

        for j in range(self.n**2):
            if j not in intersects:
                for i in range(len(ellipse[0])):
                    point = ellipse[:,i]
                    point = Point(point)
                    polygon = Polygon([(cutCorners[j,0],cutCorners[j,1]), (cutCorners[j,0],cutCorners[j,1]+2*cutSize),
                                        (cutCorners[j,0]+2*cutSize,cutCorners[j,1]),(cutCorners[j,0]+2*cutSize,cutCorners[j,1]+2*cutSize)])
                    if polygon.contains(point):
                        intersects.append(j)
                        break

        intersects = np.array(intersects)
        notIntersects = np.setdiff1d(np.linspace(0,self.n**2-1,self.n**2), intersects).astype(int)
        inside = []
        for cut in notIntersects:
            centre = Point(cutCentrePx[cut])
            polygon = Polygon(np.array(list(zip(ellipse[0],ellipse[1]))))
            if polygon.contains(centre):
                inside.append(cut)

        return intersects,inside
    
    def _gen_ellipse(self,wcsItem):

        # -- Creates a 'circle' in realspace for RA,Dec -- #
        raEll = []
        decEll = []
        for ii in np.linspace(0,2*np.pi,1000):
            raEll.append(self.ra + self.error*np.cos(ii))
            decEll.append(self.dec + self.error*np.sin(ii))
        
        # -- Converts circle to pixel space -- #
        try:
            errpixels = wcsItem.all_world2pix(raEll,decEll,0)  
        except:
            errpixels = self._secure_err_px(wcsItem,raEll,decEll)  
        ellipse = np.array(errpixels)

        return ellipse
        
    def find_error_ellipse(self,cam=None,ccd=None,plot=True,proj=True):

        if cam is None:
            cam = self.cam
        if ccd is None:
            ccd = self.ccd

        data_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}'
        if len(glob(f'{data_path}/image_files/*ffic.fits')) == 0:
            data_processor = DataProcessor(sector=self.sector,path=self.data_path,verbose=0)
            data_processor.download(cam=cam,ccd=ccd,number=1)
            
        wcsItem = _get_wcs(f'{data_path}/image_files',f'{data_path}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits')
        
        ellipse = self._gen_ellipse(wcsItem)
        
        if plot:

            d = DataProcessor(sector=self.sector,path=self.data_path)
            cutCorners, cutCentrePx, cutCentreCoords, cutSize = d.find_cuts(cam,ccd,self.n,plot=False)

            intersects,inside = self._find_impacted_cuts(ellipse,cutCorners,cutSize,cutCentrePx)
            interesting = np.union1d(intersects,inside)

            # -- Plots data -- #
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            
            if proj:
                ax = plt.subplot(projection=wcsItem)
                ax.set_xlabel(' ')
                ax.set_ylabel(' ')
            else:
                ax = plt.subplot()

            coordPx = wcsItem.all_world2pix(self.ra,self.dec,0)
            ax.scatter(coordPx[0],coordPx[1],s=40,c='black',marker='*')
            #ax.text(x=coordPx[0],y=coordPx[1]+10,s=f'{coordPx[0]:.1f},{coordPx[1]:.1f}')
            
            # -- Real rectangle edge -- #
            rectangleTotal = patches.Rectangle((44,0), 2048, 2048,edgecolor='black',facecolor='none',alpha=0.5)
            
            # -- Sets title -- #
            ax.set_title(f'Camera {cam} CCD {ccd}')
            ax.set_xlim(0,2136)
            ax.set_ylim(0,2078)
            ax.grid()

            ax.add_patch(rectangleTotal)
                
            # -- Adds cuts -- #
            colours = iter(plt.cm.rainbow(np.linspace(0, 1, self.n**2)))

            for i,corner in enumerate(cutCorners):
                c = next(colours)
                if i in interesting:
                    c2 = np.copy(c)
                    c2[-1] = 0.3
                    rectangle = patches.Rectangle(corner,2*cutSize,2*cutSize,edgecolor=c,
                                                    facecolor=c2,lw=2)
                else:
                    rectangle = patches.Rectangle(corner,2*cutSize,2*cutSize,edgecolor=c,
                                                    facecolor='none',alpha=1,lw=2)
                ax.add_patch(rectangle)
            
            ax.plot(ellipse[0],ellipse[1],color='black',linewidth=5,marker='.')
        
        return ellipse
    
    def _prelim_size_check(self,border):
        """
        Finds, based on initial ccd cube, whether proposed neighbour cut is 
        large enough for the whole download process to be worth it.
        
        ------
        Inputs
        ------
        border : str
            which border to look at, defines conditions. 
        """
        
        ellipse = self.find_error_ellipse(plot=False)

        # -- Generates ellipse cutoff conditions based on border direction -- #
        if border == 'left':
            condition = (ellipse[0] <= 0) & (ellipse[1] >= 0) & (ellipse[1] <= 2078)
        elif border == 'right':
            condition = (ellipse[0] >= 2136) & (ellipse[1] >= 0) & (ellipse[1] <= 2078)
        elif border == 'up':
            condition = (ellipse[0] >= 0) & (ellipse[0] <= 2136) & (ellipse[1] >= 2078)
        elif border == 'down':
            condition = (ellipse[0] >= 0) & (ellipse[0] <= 2136) & (ellipse[1] <= 0)
        elif border == 'upleft':
            condition = (ellipse[0] <= 0) & (ellipse[1] >= 2078) 
        elif border == 'upright':
            condition = (ellipse[0] >= 2136) & (ellipse[1] >= 2078) 
        elif border == 'downleft':
            condition = (ellipse[0] <= 0) & (ellipse[1] <= 0) 
        elif border == 'downright':
            condition = (ellipse[0] >= 2136) & (ellipse[1] <= 0) 

        # -- Cuts ellipse -- #
        where = np.where(condition)
        ellipse = ellipse[:,where]
        ellipse = ellipse[:,0,:]
        
        # -- Calculate size of cut required to encompass ellipse region -- #
        if len(ellipse[0]) > 0:
            x1 = max(ellipse[0])
            x2 = min(ellipse[0])
            x = abs(x1 - x2)
            
            y1 = max(ellipse[1])
            y2 = min(ellipse[1])
            y = abs(y1 - y2)
            
            size = x*y
        
        else:
            size = 0
            
        return size
    
    def find_neighbour_ccds(self,verbose=True):
        """
        Uses the camera/ccd of the GRB and error ellipse pixels to 
        find the neighbouring camera/ccd combinations that contain 
        some part of the ellipse. Pretty poorly coded, but I can't be 
        bothered cleaning it and debugging again haha.

        -------
        Options
        -------
        verbose : bool
            override printout

        -------
        Creates
        -------
        self.neighbours - List of tuples of cam,ccd combinations required
        
        """
        
        # -- Create ccd and inversion array that contain information 
        #    on the orientations of the TESS ccd as given by manual. 
        #    Note that the ccd array is flipped horizontally from 
        #    the manual as our positive x-axis goes to the right -- #
        ccdArray = np.array([[(4,3),(1,2)],[(4,3),(1,2)],[(2,1),(3,4)],[(2,1),(3,4)]])
        invertArray = np.array([[(True,True),(False,False)],
                                [(True,True),(False,False)],
                                [(True,True),(False,False)],
                                [(True,True),(False,False)]])
        
        # -- Check north/south pointing and initialise cam array accordingly -- #
        if self.dec > 0:
            north = True
        else:
            north = False
            
        if north:
            camArray = np.array([4,3,2,1])
        else:
            camArray = np.array([1,2,3,4])
            
        
        # -- Find the ccdArray index based on self.cam/ccd -- #
        
        for i,cam in enumerate(camArray):
            if self.cam == cam:
                camIndex = i
                
        for i in range(len(ccdArray[camIndex])):
            for j in range(len(ccdArray[camIndex][i])):
                if self.ccd == ccdArray[camIndex][i][j]:
                    ccdIndex = (i,j) # row, column
            
        total_index = (camIndex,ccdIndex) # [camIndex,(in-cam-row,in-cam-column)]
        
        # -- Create error ellipse and use max/min values to find if the ellipse
        #    intersects the up,down,left,right box edges -- #
        ellipse = self.find_error_ellipse(plot=False)
        
        right = False
        left = False
        up = False
        down = False
        
        if max(ellipse[0]) > 2136:
            right = True
        if min(ellipse[0]) < 0:
            left = True
        if max(ellipse[1]) > 2078:
            up = True
        if min(ellipse[1]) < 0:
            down = True
            
        # -- Check if inversion is required and adjust accordingly-- #
        self.invert = invertArray[total_index[0]][total_index[1][0]][total_index[1][1]]
       
        if self.invert:
            right2 = left
            left = right
            right = right2
            up2 = down
            down = up
            up = up2

        # -- Check for diagonals -- #
        upright = False
        upleft = False
        downright = False
        downleft = False
    
        if up and right:
            upright = True
        if up and left:
            upleft = True
        if down and right:
            downright = True
        if down and left:
            downleft = True
            
        # -- Calculate the neighbouring ccd information. If px area of 
        #    neighbour ccd is <70,000, ccd is disregarded as unworthy 
        #    of full download process. -- #
        neighbour_ccds = []    

        if left:
            if ccdIndex[1] == 1:
                leftccd = camArray[camIndex],ccdArray[camIndex][ccdIndex[0]][0]
                
                if self.invert:
                    size = self._prelim_size_check('right')
                    if size > 10000:
                        leftccd = (leftccd[0],leftccd[1],'Right')
                        neighbour_ccds.append(leftccd)
                    else:
                        neighbour_ccds.append('Right ccd too small')
                else:
                    size = self._prelim_size_check('left')
                    if size > 10000:
                        leftccd = (leftccd[0],leftccd[1],'Left')
                        neighbour_ccds.append(leftccd)
                    else:
                        neighbour_ccds.append('Left ccd too small')
                
        if right:
            if ccdIndex[1] == 0:
                rightccd = camArray[camIndex],ccdArray[camIndex][ccdIndex[0]][1]
                
                if self.invert:
                    size = self._prelim_size_check('left')
                    if size > 10000:
                        rightccd = (rightccd[0],rightccd[1],'Left')
                        neighbour_ccds.append(rightccd)
                    else:
                        neighbour_ccds.append('Left ccd too small')
                else:
                    size = self._prelim_size_check('right')
                    if size > 10000:
                        rightccd = (rightccd[0],rightccd[1],'Right')
                        neighbour_ccds.append(rightccd)
                    else:
                        neighbour_ccds.append('Right ccd too small')
                
        if up:
            if not (total_index[0] == 0) & (total_index[1][0] == 0):
                if total_index[1][0] == 0:
                    upCam = camIndex - 1
                    upCcd = (1,total_index[1][1])
                    upccd = camArray[upCam],ccdArray[upCam][1][total_index[1][1]]
                else:
                    upCam = camIndex
                    upCcd = (0,total_index[1][1])
                    upccd = camArray[upCam],ccdArray[upCam][0][total_index[1][1]]
                
                if self.invert:
                    size = self._prelim_size_check('down')
                    if size > 10000:
                        upccd = (upccd[0],upccd[1],'Down')
                        neighbour_ccds.append(upccd)
                    else:
                        neighbour_ccds.append('Down ccd too small')
                else:
                    size = self._prelim_size_check('up')
                    if size > 10000:
                        upccd = (upccd[0],upccd[1],'Up')
                        neighbour_ccds.append(upccd)
                    else:
                        neighbour_ccds.append('Up ccd too small')
                            
        if down:
            if not (total_index[0] == 3) & (total_index[1][0] == 1):
                if total_index[1][0] == 1:
                    downCam = camIndex + 1
                    downCcd = (0,total_index[1][1])
                    downccd = camArray[downCam],ccdArray[downCam][0][total_index[1][1]]
                else:
                    downCam = camIndex
                    downCcd = (1,total_index[1][1])
                    downccd = camArray[downCam],ccdArray[downCam][1][total_index[1][1]]
                
                if self.invert:
                    size = self._prelim_size_check('up')
                    if size > 10000:
                        downccd = (downccd[0],downccd[1],'Up')
                        neighbour_ccds.append(downccd)
                    else:
                        neighbour_ccds.append('Up ccd too small')
                else:
                    size = self._prelim_size_check('down')
                    if size > 10000:
                        downccd = (downccd[0],downccd[1],'Down')
                        neighbour_ccds.append(downccd)
                    else:
                        neighbour_ccds.append('Down ccd too small')

        
        if upright:
            if not (total_index[0] == 0) & (total_index[1][0] == 0) | (ccdIndex[1] == 1):
                if total_index[1][0] == 0:
                    urCam = camIndex - 1
                    urccd = camArray[urCam],ccdArray[urCam][1][1]
                else:
                    urCam = camIndex
                    urccd = camArray[urCam],ccdArray[urCam][0][1]
                
                if self.invert:
                    size = self._prelim_size_check('downleft')
                    if size > 10000:
                        urccd = (urccd[0],urccd[1],'Downleft')
                        neighbour_ccds.append(urccd)
                    else:
                        neighbour_ccds.append('Downleft ccd too small')
                else:
                    size = self._prelim_size_check('upright')
                    if size > 10000:
                        urccd = (urccd[0],urccd[1],'Upright')
                        neighbour_ccds.append(urccd)
                    else:
                        neighbour_ccds.append('Upright ccd too small')
                    
        
        if upleft:
            if not (total_index[0] == 0) & (total_index[1][0] == 0) | (ccdIndex[1] == 0):
                if total_index[1][0] == 0:
                    ulCam = camIndex - 1
                    ulccd = camArray[ulCam],ccdArray[ulCam][1][0]
                else:
                    ulCam = camIndex
                    ulccd = camArray[ulCam],ccdArray[ulCam][0][0]
                
                if self.invert:
                    size = self._prelim_size_check('downright')
                    if size > 10000:
                        ulccd = (ulccd[0],ulccd[1],'Downright')
                        neighbour_ccds.append(ulccd)
                    else:
                        neighbour_ccds.append('Downright ccd too small')
                else:
                    size = self._prelim_size_check('upleft')
                    if size > 10000:
                        ulccd = (ulccd[0],ulccd[1],'Upleft')
                        neighbour_ccds.append(ulccd)
                    else:
                        neighbour_ccds.append('Upleft ccd too small')

        if downright:
            if not (total_index[0] == 3) & (total_index[1][0] == 1) | (ccdIndex[1] == 1):
                if total_index[1][0] == 1:
                    drCam = camIndex + 1
                    drccd = camArray[drCam],ccdArray[drCam][0][1]
                else:
                    drCam = camIndex
                    drccd = camArray[drCam],ccdArray[drCam][1][1]
                
                if self.invert:
                    size = self._prelim_size_check('upleft')
                    if size > 10000:
                        drccd = (drccd[0],drccd[1],'Upleft')
                        neighbour_ccds.append(drccd)
                    else:
                        neighbour_ccds.append('Upleft ccd too small')
                else:
                    size = self._prelim_size_check('downright')
                    if size > 10000:
                        drccd = (drccd[0],drccd[1],'Downright')
                        neighbour_ccds.append(drccd)
                    else:
                        neighbour_ccds.append('Downright ccd too small')
            
        if downleft:
            if not (total_index[0] == 3) & (total_index[1][0] == 1) | (ccdIndex[1] == 0):
                if total_index[1][0] == 1:
                    dlCam = camIndex + 1
                    dlccd = camArray[dlCam],ccdArray[dlCam][0][0]
                else:
                    dlCam = camIndex
                    dlccd = camArray[dlCam],ccdArray[dlCam][1][0]
                
                if self.invert:
                    size = self._prelim_size_check('upright')
                    if size > 10000:
                        dlccd = (dlccd[0],dlccd[1],'Upright')
                        neighbour_ccds.append(dlccd)
                    else:
                        neighbour_ccds.append('Upright ccd too small')
                else:
                    size = self._prelim_size_check('downleft')
                    if size > 10000:
                        dlccd = (dlccd[0],dlccd[1],'Downleft')
                        neighbour_ccds.append(dlccd)
                    else:
                        neighbour_ccds.append('Downleft ccd too small')

        # -- prints information -- #
        if verbose & (self.verbose > 0):

            if north:
                print('Pointing: North')
            else:
                print('Pointing: South')
            
            print(f'This CCD: Camera {self.cam}, CCD {self.ccd}.')
            print('------------------------------')
            print('Neighbouring CCDs Required:')
            if neighbour_ccds != []:
                for item in neighbour_ccds:
                    if type(item) == str:
                        print(item)
                    else:
                        print(f'Camera {item[0]}, CCD {item[1]} ({item[2]}).')
                        
            else:
                print('No neighbour CCDs available/required.')
            print('\n')
                        
        # -- Removes disregarded ccd info to create self.neighbours -- #
        if neighbour_ccds != []:
            self.neighbours = []
            for item in neighbour_ccds:
                if type(item) == tuple:
                    self.neighbours.append(item[:-1])

    def _sector_suggestions(self):

        primary_mission = range(1,28)       # ~1200 FFIs , 30 min cadence
        secondary_mission = range(28,56)    # ~3600 FFIs , 10 min cadence
        tertiary_mission = range(56,100)    # ~12000 FFIs , 200 sec cadence

        if self.sector in primary_mission:
            self._interval = 1/48
            cube_time_sug = '45:00'
            cube_mem_sug = 20

            cut_time_sug = '20:00'
            cut_mem_sug = 20

            reduce_time_sug = '1:00:00'
            reduce_cpu_sug = 32

            search_time_sug = '20:00'
            search_cpu_sug = 32
            
            plot_time_sug = '20:00'
            plot_cpu_sug = 32

        elif self.sector in secondary_mission:
            self._interval = 1/(144)
            cube_time_sug = '1:45:00'
            cube_mem_sug = 20

            cut_time_sug = '2:00:00'
            cut_mem_sug = 20

            reduce_time_sug = '1:15:00'
            reduce_cpu_sug = 32

            search_time_sug = '20:00'
            search_cpu_sug = 32
            
            plot_time_sug = '20:00'
            plot_cpu_sug = 32

        elif self.sector in tertiary_mission:
            self._interval = 200/86400
            cube_time_sug = '6:00:00'
            cube_mem_sug = 20

            cut_time_sug = '3:00:00'
            cut_mem_sug = 20

            reduce_time_sug = '3:00:00'
            reduce_cpu_sug = 32

            search_time_sug = '1:00:00'
            search_cpu_sug = 32
            
            plot_time_sug = '20:00'
            plot_cpu_sug = 32

        suggestions = [[cube_time_sug,cube_mem_sug],
                       [cut_time_sug,cut_mem_sug],
                       [reduce_time_sug,reduce_cpu_sug],
                       [search_time_sug,search_cpu_sug],
                       [plot_time_sug,plot_cpu_sug]]
        
        return suggestions

    def _run_tessellate(self,cam=None,ccd=None):

        if cam is None:
            cam = self.cam
        if ccd is None:
            ccd = self.ccd

        suggestions = self._sector_suggestions()
        cubing = suggestions[0]
        cutting = suggestions[1]
        reducing = suggestions[2]
        searching = suggestions[3]
        plotting = suggestions[4]

        run = Tessellate(data_path=self.data_path,working_path=self.working_path,job_output_path=self.job_output_path,
                        sector=self.sector,cam=cam,ccd=ccd,n=self.n,cuts='all',
                        download=True,download_number='all',
                        make_cube=True,cube_time=cubing[0],cube_mem=cubing[1],
                        make_cuts=True,cut_time=cutting[0],cut_mem=cutting[1],
                        reduce=True,reduce_time=reducing[0],reduce_cpu=reducing[1],
                        search=True,search_time=searching[0],search_cpu=searching[1],
                        plot=True,plot_time=plotting[0],plot_cpu=plotting[1],
                        delete=False,reset_logs=False,overwrite=False)
        
    def run(self):

        self.find_neighbour_ccds(verbose=True)
        go = input('Run Tessellate? [y/n] ')
        done = False
        while not done:
            if go == 'y':
                done = True
                print('\n')
            elif go == 'n':
                print('Aborted\n')
                return
            else:
                go = input('Invalid format! Run Tessellate? [y/n] ')

        self._run_tessellate()
        for cam,ccd in self.neighbours:
            self._run_tessellate(cam,ccd)

    def _cut_events_inside_ellipse(self,cam,ccd,cut,timestart,timeend,eventduration):
        
        detector = Detector(sector=self.sector,cam=cam,ccd=ccd,n=self.n,data_path=self.data_path)
        detector._gather_results(cut)

        events = detector.events

        return events[(events['mjd_start'].values > timestart) & (events['mjd_start'].values < timeend)]# & (events['mjd_duration'].values < eventduration)]

    def _cut_events_intersecting_ellipse(self,cam,ccd,cut,ellipse,timestart,timeend,eventduration):
        
        detector = Detector(sector=self.sector,cam=cam,ccd=ccd,n=self.n,data_path=self.data_path)
        detector._gather_results(cut)

        events = detector.events

        events = events[(events['mjd_start'].values > timestart) & (events['mjd_start'].values < timeend)]# & (events['mjd_duration'].values < eventduration)]

        good = []
        for i,event in events.iterrows():
            point = Point((event['xccd'],event['yccd']))
            polygon = Polygon(np.array(list(zip(ellipse[0],ellipse[1]))))
            if polygon.contains(point):
                good.append(i)
        
        events = events.iloc[np.array(good)]

        return events

    def _gather_detection_tables(self,cam,ccd,timeStartBuffer,eventDuration):
        """
        timeBuffer is in minutes
        """

        data_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}'
        wcsItem = _get_wcs(f'{data_path}/image_files',f'{data_path}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits')
        ellipse = self._gen_ellipse(wcsItem)
        d = DataProcessor(sector=self.sector,path=self.data_path)
        cutCorners, cutCentrePx, cutCentreCoords, cutSize = d.find_cuts(cam,ccd,self.n,plot=False)
        intersects,inside = self._find_impacted_cuts(ellipse,cutCorners,cutSize,cutCentrePx)

        timeStart = self.eventtime-2*self._interval 
        timeEnd = self.eventtime + timeStartBuffer/1440

        tables = []
        for cut in inside:
            cut += 1
            tables.append(self._cut_events_inside_ellipse(cam,ccd,cut,timeStart,timeEnd,eventDuration))
        
        for cut in intersects:
            cut += 1
            tables.append(self._cut_events_intersecting_ellipse(cam,ccd,cut,ellipse,timeStart,timeEnd,eventDuration))

        return pd.concat(tables)




        