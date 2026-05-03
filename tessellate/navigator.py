import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

import pandas as pd
from time import time as clock
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 


from .tools import RoundToInt, Generate_LC, Frame_Bin

class Navigator():

    def __init__(self,sector,cam,ccd,data_path='/fred/oz335/TESSdata',n=8):

        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.data_path = data_path
        self.n = n

        self.path = f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}'

        # -- Results -- #
        self.sources = None
        self.events = None
        self.objects = None

        # -- Data -- #
        self.time = None
        self.flux = None
        self.ref = None
        self.mask = None
        self.bkg = None

    


    # ----------------------------- Gathering data products ----------------------------- #
 
    def gather_results(self,cut,sources=True,events=True,objects=True):
        """
        Gather detection csvs.
        """

        from .localisation import CutWCS
        import ast

        path = f'{self.path}/Cut{cut}of{self.n**2}'

        if sources:
            try:
                self.sources = pd.read_csv(f'{path}/detected_sources.csv')    # raw detection results
            except:
                print('No detected sources file found')
                self.sources = None

        if events:
            try:
                self.events = pd.read_csv(f'{path}/detected_events.csv')    # temporally located with same object id
                self.events['crossbin_ids'] = self.events['crossbin_ids'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            except:
                print('No detected events file found')
                self.events = None

        if objects: 
            try:
                self.objects = pd.read_csv(f'{path}/detected_objects.csv')    # temporally and spatially located with same object id
            except:
                print('No detected objects file found')
                self.objects = None
        
        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=cut,n=self.n)
        
    def gather_data(self,cut,flux=True,time=True,segments=True,ref=False,mask=False,bkg=False,verbose=True):
        """
        Gather reduced data.
        """

        from .localisation import CutWCS

        if verbose:
            ts = clock()
            print(f'Loading Cut {cut} Data...',end='\r')

        base = f'{self.path}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}'

        if flux:
            self.flux = np.load(base + '_ReducedFlux.npy')
            self.cut = cut

        if time:
            self.time = np.load(base + '_Times.npy')

        if ref:
            self.ref = np.load(base + '_Ref.npy')

        if bkg:
            self.bkg = np.load(base + '_Background.npy')

        if mask:
            self.mask = np.load(base + '_Mask.npy')
        
        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=cut,n=self.n)

        if verbose:
            print(f'Loading Cut {cut} Data -- done! ({clock()-ts:.0f}s)')





    # ----------------------------- Filtering sources, events, objects ----------------------------- #

    def filter_events(self,cut=None,starkiller=False,asteroidkiller=False,strapkiller=False,
                      lower=None,upper=None,image_sig_max=None,frame_bin=None,
                      lc_sig_max=None,lc_sig_med=None,min_events=None,max_events=None,
                      bkg_std=None,boundary_buffer=None,flux_sign=None,classification=None,
                      psf_like=None,galactic_latitude=None,centroid_err=None,crossbins=True):
         
        """
        Returns a dataframe of the events in the cut, with options to filter by various parameters.
        """

        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        else:
            self.gather_results(cut)

        # -- Remove events near sources in reduction source mask (ie. stars) -- #
        if starkiller:
            events = deepcopy(self.events.loc[self.events.gaia_id == '-'])
        else:
            events = deepcopy(self.events)

        # -- Remove asteroids from the results -- #
        if asteroidkiller:
            events = events.loc[~(events.classification == 'Asteroid')]
            if 'Asteroid' in events.keys():
                events = events.loc[events.Asteroid == 0]

        # -- Remove events near strap structures -- #
        if strapkiller:
            events = events.loc[events.source_mask < 4]

        # -- Pick out frame bins -- #
        if frame_bin is not None:
            events = events.loc[events.frame_bin == frame_bin]

        # -- Remove anything picked up at a different resolution -- #
        if not crossbins:
            events = events[events['crossbin_ids'].apply(lambda x: len(x) == 0)]

        # -- Remove events within 'boundary_buffer' of the boundary -- #
        if boundary_buffer is not None:
            
            from .tools import Get_Tess_Downlink

            self.gather_data(cut, flux=False, verbose=False)
            
            mask = pd.Series(True, index=events.index)
            for frame_bin in np.unique(events.frame_bin):
                bin_events = events[events.frame_bin == frame_bin]
                time = Frame_Bin(self.sector, self.cam, self.time, frame_bin=frame_bin)
                break_idx = Get_Tess_Downlink(self.sector, self.cam, time)
                last_idx = len(time) - 1

                boundaries = [0, break_idx, break_idx + 1, last_idx]

                boundary_frames = set()
                for b in boundaries:
                    for offset in range(int(-boundary_buffer/frame_bin), int(boundary_buffer/frame_bin) + 1):
                        idx = b + offset
                        if 0 <= idx <= last_idx:
                            boundary_frames.add(idx)

                bin_mask = ~bin_events.apply(
                    lambda row: any(frame in range(row.frame_start, row.frame_end + 1) for frame in boundary_frames),
                    axis=1
                )
                mask[bin_events.index] = bin_mask

            events = events[mask]

        # -- Restrict to galactic latitudes higher than given value -- #
        if galactic_latitude is not None:
            if type(galactic_latitude) == float:
                galactic_latitude = [galactic_latitude,90]
            elif type(galactic_latitude) == list:
                pass
            else:
                e = 'Galactic latitude must be a float or a list of floats!'
                raise ValueError(e)
            events = events.loc[(abs(events.gal_b) >= min(galactic_latitude)) & (abs(events.gal_b) <= max(galactic_latitude))]

        # -- Filter based on classification -- #
        if classification is not None:
            is_negation = classification.startswith(('!', '~'))
            classification_stripped = classification.lstrip('!~').lower()
            if classification_stripped in ['var', 'variable']:
                classification = classification = ['VCR', 'VRRLyr', 'VEB','VLPV','VST','VAGN','VRM','VMSO','RRLyrae']  
            else:
                classification = [classification_stripped]

            if is_negation:
                events = events[~events.classification.str.lower().isin([classification[i].lower() for i in range(len(classification))])]
            else:
                events = events[events.classification.str.lower().isin([classification[i].lower() for i in range(len(classification))])]

        # -- Filter by upper and lower limits on number of detections within each event -- #
        if upper is not None or lower is not None:
            mask = pd.Series(True, index=events.index)
            for frame_bin in np.unique(events.frame_bin):
                bin_idx = events[events.frame_bin == frame_bin].index
                if upper is not None:
                    mask[bin_idx] &= events.loc[bin_idx, 'frame_duration'] <= upper / frame_bin
                if lower is not None:
                    mask[bin_idx] &= events.loc[bin_idx, 'frame_duration'] >= lower / frame_bin
            events = events[mask]

        # -- Filter by various parameters -- #
        if lc_sig_max is not None:
            events = events.loc[events.lc_sig_max >= lc_sig_max]
        if lc_sig_med is not None:
            events = events.loc[events.lc_sig_med >= lc_sig_med]
        if image_sig_max is not None:
            events = events.loc[events.image_sig_max >= image_sig_max]
        if min_events is not None:
            events = events.loc[events.total_events >= min_events]
        if max_events is not None:
            events = events.loc[events.total_events <= max_events]
        if bkg_std is not None:
            events = events.loc[events.bkg_std <= bkg_std]
        if flux_sign is not None:
            events = events.loc[events.flux_sign == flux_sign]
        if psf_like is not None:
            events = events.loc[events.psf_like>=psf_like]
        if centroid_err is not None:
            events = events.loc[(events.xcentroid_err**2 + events.ycentroid_err**2) <= 2 * centroid_err**2]

        return events

    def filter_objects(self,cut=None,
                       ra=None,dec=None,distance=40,
                       min_events=None,max_events=None,frame_bin=None,
                       classification=None,flux_sign=None,centroid_err=None,
                       lc_sig_max=None,psf_like=None,galactic_latitude=None,
                       min_eventlength_frame=None,max_eventlength_frame=None,
                       min_eventlength_mjd=None,max_eventlength_mjd=None):
        
        """
        Filter self.objects based on these main things.
        """
        
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        else:
            self.gather_results(cut)

        objects = deepcopy(self.objects)

        # -- Filter based on location and distance -- #
        if (ra is not None) & (dec is not None):
            if type(ra) == float:
                target_coord = SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
            elif 'd' in dec:
                SkyCoord(ra=ra,dec=dec)
            else:
                e = 'Please specify coordinates in (deg,deg) or (hms,dms)!'
                raise ValueError(e)
            
            source_coords = SkyCoord(ra=objects.ra.values*u.degree,dec=objects.dec.values*u.degree)
            separations = target_coord.separation(source_coords)
            objects = objects[separations<distance*u.arcsec]

        # -- Filter based on the number of events associated with the object -- #
        if min_events is not None:
            objects = objects[objects.n_events >= min_events]
        if max_events is not None:
            objects = objects[objects.n_events <= max_events]

        # -- Restrict to galactic latitudes higher than given value -- #
        if galactic_latitude is not None:
            if type(galactic_latitude) == float:
                galactic_latitude = [galactic_latitude,90]
            elif type(galactic_latitude) == list:
                pass
            else:
                e = 'Galactic latitude must be a float or a list of floats!'
                raise ValueError(e)
            objects = objects.loc[(abs(objects.gal_b) >= min(galactic_latitude)) & (abs(objects.gal_b) <= max(galactic_latitude))]

        # -- Filter based on classification -- #
        if classification is not None:
            is_negation = classification.startswith(('!', '~'))
            classification_stripped = classification.lstrip('!~').lower()
            if classification_stripped in ['var', 'variable']:
                classification = classification = ['VCR', 'VRRLyr', 'VEB','VLPV','VST','VAGN','VRM','VMSO']  # Replace with variable classes
            else:
                classification = [classification_stripped]

            if is_negation:
                objects = objects[~objects.classification.str.lower().isin([classification[i].lower() for i in range(len(classification))])]
            else:
                objects = objects[objects.classification.str.lower().isin([classification[i].lower() for i in range(len(classification))])]

        # -- Filter by various parameters -- #
        if frame_bin is not None:
            objects = objects.loc[objects.frame_bin == frame_bin]
        if flux_sign is not None:
            objects = objects[objects.flux_sign==flux_sign]
        if lc_sig_max is not None:
            objects = objects[objects.lc_sig_max>=lc_sig_max]
        if psf_like is not None:
            objects = objects[objects.psf_maxsig>=psf_like]
        if centroid_err is not None:
            objects = objects[objects.xcentroid_err**2+objects.ycentroid_err**2 <= 2*centroid_err**2]
        if min_eventlength_frame is not None:
            objects = objects[objects.min_eventlength_frame>=min_eventlength_frame]
        if max_eventlength_frame is not None:
            objects = objects[objects.max_eventlength_frame<=max_eventlength_frame]
        if min_eventlength_mjd is not None:
            objects = objects[objects.min_eventlength_mjd>=min_eventlength_mjd]
        if max_eventlength_mjd is not None:
            objects = objects[objects.max_eventlength_mjd<=max_eventlength_mjd]

        return objects

    def crossbin_events(self,crossbin_id=None,objid=None,eventid=None):
        """
        Isolate events that are matched at different time binning.
        """

        if crossbin_id is not None:
            return self.events[self.events['crossbin_ids'].apply(lambda x: crossbin_id in x)]
        elif (objid is not None)&(eventid is not None):
            event = self.events[(self.events.objid==objid)&(self.events.eventid==eventid)].iloc[0]
            matches = pd.DataFrame()
            for crossbin_id in event.crossbin_ids:
                matches = pd.concat([matches,self.crossbin_events(crossbin_id)])
            return matches.drop_duplicates()

    # ----------------------------- Extracting light curves / images of events ----------------------------- #

    def event_lc(self,objid,eventid,cut=None,frame_buffer=10,plot=True,frame_bin=None):
        """
        Extract an aperture light curve for a desired event (objid/eventid pair).
        """

        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        elif cut != self.cut:
            self.gather_data(cut)
            self.gather_results(cut)
        
        # -- Isolate and extract event with frame buffer either side -- #
        event = self.events[(self.events.objid==objid)&(self.events.eventid==eventid)].iloc[0]

        time, flux = (Frame_Bin(self.sector, self.cam, self.time, self.flux, event.frame_bin) if event.frame_bin > 1 else (self.time, self.flux))

        x = RoundToInt(event.xint)     # x coordinate of the source
        y = RoundToInt(event.yint)      # y coordinate of the source
        frame_start = RoundToInt(event.frame_start)        # Start frame of the event
        frame_end = RoundToInt(event.frame_end)            # End frame of the event

        frame_start = np.max([frame_start-frame_buffer,0])
        frame_end = np.min([frame_end+frame_buffer+1,len(time)-1])

        t,f = Generate_LC(time,flux,x,y,frame_start,frame_end,radius=1.5)

        # -- Plot lightcurve -- #
        if plot:
            cadence = np.median(np.diff(time))
            fig,ax = plt.subplots()
            ax.plot(t,f,'x-',c='k')
            ax.axvspan(t[frame_buffer]-cadence/2,t[frame_buffer+event.frame_end-event.frame_start]+cadence/2,color='C1',alpha=0.4)
            ax.set_xlabel('Time (MJD)')
            ax.set_ylabel('TESS Counts')
            if event.frame_bin > 1:
                rawt,rawf = Generate_LC(self.time,self.flux,x,y,frame_start*event.frame_bin,frame_end*event.frame_bin,radius=1.5)
                ax.plot(rawt,rawf,'.',c='k',alpha=0.3)

            if frame_bin is not None and frame_bin > event.frame_bin:
                largertime, largerflux = (Frame_Bin(self.sector, self.cam,self.time, self.flux, frame_bin))
                largert,largerf = Generate_LC(largertime,largerflux,x,y,int(frame_start/frame_bin*event.frame_bin),int(frame_end/frame_bin*event.frame_bin),radius=1.5)
                ax.plot(largert, largerf, '^', c='r', alpha=0.8)

        
        return t,f
    

    def event_frames(self,objid,eventid,cut=None,frame_buffer=2,frame_interval=1,image_size=11,plot=True,frame_bin=None):
        """
        Extract cutout images for chosen event.
        """

        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        elif cut != self.cut:
            self.gather_data(cut)
            self.gather_results(cut)

        # -- Isolate event -- #
        event = self.events[(self.events.objid==objid)&(self.events.eventid==eventid)].iloc[0]

        if frame_bin is None:
            frame_bin = event.frame_bin
            brightest_frame = RoundToInt(event.frame_max)
            frame_start =  RoundToInt(event.frame_start)
            frame_end =  RoundToInt(event.frame_end)
            frame_interval =  RoundToInt(frame_interval * frame_bin)
        else:
            brightest_frame = RoundToInt(event.frame_max*event.frame_bin/frame_bin)
            frame_start =  RoundToInt(event.frame_start *event.frame_bin/frame_bin)
            frame_end =  RoundToInt(event.frame_end * event.frame_bin/frame_bin)
            frame_interval =  RoundToInt(frame_interval * frame_bin * event.frame_bin/frame_bin)
            frame_buffer = RoundToInt(frame_buffer * event.frame_bin/frame_bin)

        time, flux = (Frame_Bin(self.sector, self.cam,self.time, self.flux, frame_bin) if frame_bin > 1 else (self.time, self.flux))
    
        # -- Find frames to be included based on buffer and interval, ensuring brightest is included -- #
        frames = np.arange(frame_start-frame_buffer, frame_end+1,frame_interval).astype(int)
        if brightest_frame not in frames:
            frames = np.sort(np.append(frames,brightest_frame))
        while len(frames) < 5:
            frames = np.append(frames,frames[-1]+frame_interval)
        frames[frames<0] = 0
        frames[frames>len(time)]=len(time)-1
        frames = np.unique(frames)

        # -- Define cutout -- #
        x = RoundToInt(event.xint) 
        y = RoundToInt(event.yint) 
        xmin = max(x-image_size//2,0)
        xmax = min(x+image_size//2,flux.shape[1])
        ymin = max(y-image_size//2,0)
        ymax = min(y+image_size//2,flux.shape[1])

        images = flux[frames,ymin:ymax+1,xmin:xmax+1]

        # -- Plot 5 images around the brightest frame -- #
        if plot:
            fig,ax = plt.subplots(ncols=5,figsize=(15,15))
            brightest_loc = np.where(frames==brightest_frame)[0][0]
            vmax = np.percentile(images[brightest_loc,image_size//2-1:image_size//2+2,image_size//2-1:image_size//2+2],80)
            vmin = np.percentile(images[brightest_loc,image_size//2-1:image_size//2+2,image_size//2-1:image_size//2+2],16)
            for i in range(5):
                im = ax[i].imshow(images[brightest_loc-2+i],origin='lower',cmap='gray',vmax=vmax,vmin=vmin)
                
                add = ' Stacked ' if frame_bin > 1 else ' '

                if i == 2:
                    ax[i].set_title(f'Brightest{add}Frame ({brightest_frame})')
                else:
                    ax[i].set_title(f'{add}Frame ({frames[brightest_loc-2+i]})')
            fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04,label='TESS Counts')    
            
        return images

    def object_lc(self,objid,cut=None):
        """
        Extract an aperture light curve for a desired object.
        """

        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        elif cut != self.cut:
            self.gather_data(cut)
            self.gather_results(cut)
        
        # -- Isolate object and generate light curve -- #
        obj = self.objects[self.objects['objid']==objid].iloc[0]
        x = RoundToInt(obj['xcentroid'])     # x coordinate of the source
        y = RoundToInt(obj['ycentroid'])     # x coordinate of the source

        time, flux = (Frame_Bin(self.sector, self.cam,self.time, self.flux, obj.frame_bin) if obj.frame_bin > 1 else (self.time, self.flux))

        t,f = Generate_LC(time,flux,x,y)

        return t,f
    
    def object_frames(self,objid,cut=None,image_size=11):
        """
        Extract ful sector cutout images for chosen object.
        """

        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        elif cut != self.cut:
            self.gather_data(cut)
            self.gather_results(cut)

        # -- Isolate object and define image boundary -- #
        obj = self.objects[self.objects.objid==objid].iloc[0]

        x = RoundToInt(obj.xcentroid) 
        xmin = max(x-image_size//2,0)
        xmax = min(x+image_size//2,self.flux.shape[1])

        y = RoundToInt(obj.ycentroid) 
        ymin = max(y-image_size//2,0)
        ymax = min(y+image_size//2,self.flux.shape[1])

        time, flux = (Frame_Bin(self.sector, self.cam,self.time, self.flux, obj.frame_bin) if obj.frame_bin > 1 else (self.time, self.flux))

        return flux[:,ymin:ymax+1,xmin:xmax+1]




    # ----------------------------- Figure Generation ----------------------------- #

    def asteroid_tracks(self,cut=None):
        """
        Plot trajectories of asteroids sorted by their asteroid_id.
        """

        asteroids = self.filter_events(cut,classification='Asteroid')
        nonasteroids = self.filter_events(cut,asteroidkiller=True)

        cmap = plt.get_cmap('tab20')
        t_min_global = self.events['mjd_max'].min()
        t_max_global = self.events['mjd_max'].max()
        # t_range_full = np.linspace(t_min_global, t_max_global, 300)

        fig,ax = plt.subplots(ncols=3,figsize=(15,5))
        fig.subplots_adjust(wspace=0.3)

        linked = asteroids[asteroids.asteroid_id>0]
        unlinked = asteroids[asteroids.asteroid_id==0]

        ax[0].scatter(linked.xcentroid,linked.ycentroid,c=cmap(linked['asteroid_id']%20),s=5)
        ax[1].scatter(linked.xcentroid,linked.mjd_max,c=cmap(linked['asteroid_id']%20),s=5)
        ax[2].scatter(linked.ycentroid,linked.mjd_max,c=cmap(linked['asteroid_id']%20),s=5)

        ax[0].scatter(unlinked.xcentroid,unlinked.ycentroid,edgecolor='k',facecolor='None',s=10,alpha=0.5,marker='^')
        ax[1].scatter(unlinked.xcentroid,unlinked.mjd_max,edgecolor='k',facecolor='None',s=10,alpha=0.5,marker='^')
        ax[2].scatter(unlinked.ycentroid,unlinked.mjd_max,edgecolor='k',facecolor='None',s=10,alpha=0.5,marker='^')

        ax[0].scatter(nonasteroids.xcentroid,nonasteroids.ycentroid,c='gray',s=1,alpha=0.1)
        ax[1].scatter(nonasteroids.xcentroid,nonasteroids.mjd_max,c='gray',s=1,alpha=0.1)
        ax[2].scatter(nonasteroids.xcentroid,nonasteroids.mjd_max,c='gray',s=1,alpha=0.1)

        maxsize = 2048//self.n
        ax[0].set_xlim(0, maxsize); ax[0].set_ylim(0, maxsize)
        ax[1].set_xlim(0, maxsize); ax[1].set_ylim(t_min_global, t_max_global)
        ax[2].set_xlim(0, maxsize); ax[2].set_ylim(t_min_global, t_max_global)
        ax[0].set_title('x vs y') 
        ax[1].set_title('x vs t') 
        ax[2].set_title('y vs t')
        ax[0].set_xlabel('xcentroid')
        ax[0].set_ylabel('ycentroid')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('mjd_max')
        ax[2].set_xlabel('ycentroid')
        ax[2].set_ylabel('mjd_max')


        # for tr in final_tracks:
        #     c = cmap(int(tr.track_id) % 20)
            
        #     # Predict across the ENTIRE time range
        #     px, py = tr.predict(t_range_full)
            
        #     # Optional: Set a transparency based on 'n_points' or 'residual' 
        #     # so we trust solid lines more than sketchy ones
        #     line_alpha = 0.5 if tr.n_points > 4 else 0.2
            
        #     ax[0].plot(px, py, '-', color=c, lw=1.0, alpha=line_alpha, zorder=2)
        #     ax[1].plot(px, t_range_full, '-', color=c, lw=1.0, alpha=line_alpha, zorder=2)
        #     ax[2].plot(py, t_range_full, '-', color=c, lw=1.0, alpha=line_alpha, zorder=2)



    def external_photometry(self,objid,eventid=None,cut=None,tess_grid=5,sigma=3,phot=None,check='gaia'):
        """
        Look up legacy imaging for region around object/event location.
        """

        from .external_photometry import event_cutout
        from matplotlib.patches import Polygon


        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        elif cut != self.cut:
            self.gather_data(cut)
            self.gather_results(cut)

        print('Getting Photometry...')

        # -- Define the localisation and error points in RA,Dec space -- #
        theta = np.linspace(0, 2*np.pi, 1000)
        if type(eventid) == int:
            event = self.events[(self.events.objid==objid) & (self.events.eventid==eventid)].iloc[0] 

            xint = RoundToInt(event.xcentroid)
            yint = RoundToInt(event.ycentroid)
            ra_obj = event.ra
            dec_obj = event.dec
            
            # error_x_rad = min(sigma*event.xcentroid_err,0.5)
            # error_y_rad = min(sigma*event.ycentroid_err,0.5)
            error_x_rad = sigma*event.xcentroid_err
            error_y_rad = sigma*event.ycentroid_err
            errorX = event.xcentroid + error_x_rad*np.cos(theta)
            errorY = event.ycentroid + error_y_rad*np.sin(theta)

        else:
            obj = self.objects[self.objects['objid']==objid].iloc[0]

            xint = RoundToInt(obj.xcentroid)
            yint = RoundToInt(obj.ycentroid)
            ra_obj = obj.ra
            dec_obj = obj.dec

            # error_x_rad = min(sigma*obj.xcentroid_err,0.5)
            # error_y_rad = min(sigma*obj.ycentroid_err,0.5)
            error_x_rad = sigma*obj.xcentroid_err
            error_y_rad = sigma*obj.ycentroid_err
            errorX = obj.xcentroid + error_x_rad*np.cos(theta)
            errorY = obj.ycentroid + error_y_rad*np.sin(theta)
    
        RA,DEC = self.wcs.all_pix2world(xint,yint,0)
        errorRA,errorDEC = self.wcs.all_pix2world(errorX,errorY,0)

        # -- Extract a cutout of the region -- #
        if check == 'gaia_local':
            path = f'{self.path}/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
        fig, wcs, size, photometry,cat,im = event_cutout((RA,DEC),20*tess_grid,phot=phot,check=path)
        if fig is None:
            return None,None,None,None,None
        axes = fig.get_axes()
        axes[0].set_xlim(size,0)
        axes[0].set_ylim(0,size)
        
        # -- Add centroid localisation, error, and legend to cutout -- #
        axes[0].scatter(ra_obj,dec_obj, transform=axes[0].get_transform('fk5'),
                    edgecolors='red',marker='x',s=50,facecolors="red",linewidths=2,label='Target')
        
        # axes[0].scatter(errorRA,errorDEC, transform=axes[0].get_transform('fk5'),
        #             edgecolors='red',marker='.',s=15,lw=1)
        ellipse_patch = Polygon(np.column_stack([errorRA, errorDEC]),closed=True,transform=axes[0].get_transform('fk5'),
                                fill=False, edgecolor='red',linewidth=1.5)
        axes[0].add_patch(ellipse_patch)

        legend = axes[0].legend(loc=2,facecolor="black",fontsize=10)
        for text in legend.get_texts():
                text.set_color("white")
        

        # -- Generate and plot TESS pixel edges on the axes -- #
        number = tess_grid//2 + 1
        xRange = np.arange(xint-number,xint+number)
        yRange = np.arange(yint-number,yint+number)

        lines = []
        for x in xRange:
            line = np.linspace((x,yRange[0]),(x,yRange[-1]),100)
            lines.append(line)
        for y in yRange:
            line = np.linspace((xRange[0],y),(xRange[-1],y),100)
            lines.append(line)

        ys = []
        for j,line in enumerate(lines):
            if j in [0,2*number]:
                color = 'red'
                lw = 5
                alpha = 0.7
            elif j in [2*number-1,4*number-1]:
                color = 'cyan'
                lw = 5
                alpha = 0.7
            else:
                color='white'
                lw = 2
                alpha = 0.3

            line_ra,line_dec = self.wcs.all_pix2world(line[:,0]+0.5,line[:,1]+0.5,0)
            line_x,line_y = wcs.all_world2pix(line_ra,line_dec,0)

            if j in [0,5]:
                ys.append(np.mean(y))
            good = (line_x>0)&(line_y>0)&(line_x<size)&(line_y<size)
            line_x = line_x[good]
            line_y = line_y[good]
            if len(line_x) > 0:
                axes[0].plot(line_x,line_y,color=color,alpha=alpha,lw=lw)

        return fig, cat, (ra_obj,dec_obj), wcs,im
    

    @staticmethod
    def Plot_Object_LC_Frame(sector,cam,rawtimes,rawflux,events,objid,eventtype,save_path=None,latex=True,zoo_mode=False):
        """
        Plot an object's light curve and image cutout.
        """
        from matplotlib.lines import Line2D
        import matplotlib.patches as patches
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        
        # -- Use Latex in the plots -- #
        if latex:
            plt.rc('text', usetex=latex)

        # -- Select sources associated with the object id -- #
        events =  events[events.objid==objid]      
        total_events = RoundToInt(np.nanmean(events.total_events.values))   #  Number of events associated with the object id

        frame_bin = int(events.iloc[0].frame_bin)
                                                
        # -- Compile source list based on if plotted source contains all in one -- #
        if type(eventtype) == str:
            if eventtype.lower() == 'separate':
                pass
            elif eventtype.lower() == 'all':

                frame_starts = events['frame_start'].values
                frame_ends = events['frame_end'].values

                # Sets this one "event" to include all the times between first and last detection #
                event = deepcopy(events.iloc[0])
                event.frame_start = events.frame_start.min()
                event.mjd_start = events.mjd_start.min()
                event.frame_end = events.frame_end.max()
                event.mjd_end = events.mjd_end.min()
                event.mjd_duration = event.mjd_end - event.mjd_start
                event.frame = (event.frame_end + event.frame_start) / 2 
                event.mjd = (event.mjd_end + event.mjd_start) / 2 

                # Sets the x and y coordinates to the brightest source in the event #
                brightest = np.where(events.image_sig_max == np.nanmax(events.image_sig_max))[0][0]
                brightest = deepcopy(events.iloc[brightest])
                event.xccd = brightest.xccd
                event.yccd = brightest.yccd
                event.xint = brightest.xint
                event.yint = brightest.yint
                event.xcentroid = brightest.xcentroid
                event.ycentroid = brightest.ycentroid

                events = event.to_frame().T       # "events" in now a single event
                
        elif type(eventtype) == int:
            events = deepcopy(events.iloc[events.eventid.values == eventtype])
        elif type(eventtype) == list:
            events = deepcopy(events.iloc[events.eventid.isin(eventtype).values])
        else:
            m = "No valid eventtype option selected, input either 'all', 'separate', an integer event id, or list of integers."
            raise ValueError(m)


        # -- Bin time if event is binned -- #
        times, flux = (Frame_Bin(sector, cam, rawtimes, rawflux, frame_bin) if frame_bin > 1 else (rawtimes, rawflux))


        # -- Generates time for plotting and finds breaks in the time series based on the median and standard deviation - #
        time = times - times[0]             
        med = np.nanmedian(np.diff(time))           
        std = np.nanstd(np.diff(time))              
        break_ind = np.where(np.diff(time) > med+1*std)[0]
        break_ind = np.append(break_ind,len(time)) 
        break_ind += 1
        break_ind = np.insert(break_ind,0,0)

        if frame_bin > 1:
            rawtime = rawtimes - rawtimes[0]             
            med = np.nanmedian(np.diff(rawtime))           
            std = np.nanstd(np.diff(rawtime))              
            raw_break_ind = np.where(np.diff(rawtime) > med+1*std)[0]
            raw_break_ind = np.append(raw_break_ind,len(rawtime)) 
            raw_break_ind += 1
            raw_break_ind = np.insert(raw_break_ind,0,0)


        # -- Iterates over each source in the events dataframe and generates plot -- #
        for i in range(len(events)):
            eventid = events.eventid.iloc[i]          # Select event ID
            event = deepcopy(events.iloc[i])             # Select source 
            x = RoundToInt(event.xcentroid)      # x coordinate of the source
            y = RoundToInt(event.ycentroid)      # y coordinate of the source
            frame_start = RoundToInt(event.frame_start)        # Start frame of the event
            frame_end = RoundToInt(event.frame_end)            # End frame of the event

            _,f = Generate_LC(times,flux,x,y,radius=1.5)
            if frame_bin > 1:
                _,rawf = Generate_LC(rawtimes,rawflux,x,y,radius=1.5)

            # Find brightest frame in the event #
            if frame_end - frame_start >= 2:
                brightestframe = frame_start + np.where(abs(f[frame_start:frame_end]) == np.nanmax(abs(f[frame_start:frame_end])))[0][0]
            else:
                brightestframe = frame_start
            try:
                brightestframe = int(brightestframe)
            except:
                brightestframe = int(brightestframe[0])
            if brightestframe >= len(flux):   # If the brightest frame is out of bounds, set it to the last frame
                brightestframe -= 1
            if frame_end >= len(flux):         # If the frame end is out of bounds, set it to the last frame
                frame_end -= 1

            # Generate zoom light curve around event #
            fstart = frame_start-20
            if fstart < 0:
                fstart = 0
            zoom = f[fstart:frame_end+20]

            # Create the figure and axes for the plot #
            fig,ax = plt.subplot_mosaic([[1,1,1,2,2],[1,1,1,3,3]],figsize=(7*1.1,5.5*1.1),constrained_layout=True)

            # Invisibly plot event into main panel to extract ylims for zoom inset plot # 
            ax[1].plot(time[fstart:frame_end+20],zoom,'k',alpha=0)          
            insert_ylims = ax[1].get_ylim()

            # Plot each segment of the light curve in black, with breaks in the time series #
            for i in range(len(break_ind)-1):
                ax[1].plot(time[break_ind[i]:break_ind[i+1]],f[break_ind[i]:break_ind[i+1]],'k',alpha=0.8)
                # if frame_bin > 1:
                #     ax[1].plot(rawtime[raw_break_ind[i]:raw_break_ind[i+1]],rawf[raw_break_ind[i]:raw_break_ind[i+1]],
                #                'k',alpha=0.3,marker='.',ls='')


            ylims = ax[1].get_ylim()
            ax[1].set_ylim(ylims[0],ylims[1]+(abs(ylims[0]-ylims[1])))
            ax[1].set_xlim(np.min(time),np.max(time))

            # Differences between Zooniverse mode and normal mode #
            if zoo_mode:
                ax[1].set_title('Is there a transient in the orange region?',fontsize=15)   
                ax[1].set_ylabel('Brightness',fontsize=15,labelpad=10)
                ax[1].set_xlabel('Time (days)',fontsize=15)
                
                axins = ax[1].inset_axes([0.02, 0.55, 0.96, 0.43])      # add inset axes for zoomed in view of the event
                axins.yaxis.set_tick_params(labelleft=False,left=False)
                axins.xaxis.set_tick_params(labelbottom=False,bottom=False)
                ax[1].yaxis.set_tick_params(labelleft=False,left=False)

            else:
                ax[1].set_title(f"{event['TSS Catalogue']}   |   ObjID: {event.objid}",fontsize=15)   
                ax[1].set_ylabel('Counts (e/s)',fontsize=15,labelpad=10)
                ax[1].set_xlabel(f'Time (MJD - {np.round(times[0],3)})',fontsize=15)

                axins = ax[1].inset_axes([0.1, 0.55, 0.86, 0.43])       # add inset axes for zoomed in view of the event
        

            # Generate a coloured span during the event #
            cadence = np.median(np.diff(time))
            if eventtype == 'all':
                for i in range(len(frame_starts)):
                    axins.axvspan(time[frame_starts[i]]-cadence/2,time[frame_ends[i]]+cadence/2,color='C1',alpha=0.4)
            else:
                axins.axvspan(time[frame_start]-cadence/2,time[frame_end]+cadence/2,color='C1',alpha=0.4)


            # Plot full light curve in inset axes #
            for i in range(len(break_ind)-1):
                if zoo_mode:
                    axins.plot(time[break_ind[i]:break_ind[i+1]],f[break_ind[i]:break_ind[i+1]],'k',alpha=0.8)
                else:
                    axins.plot(time[break_ind[i]:break_ind[i+1]],f[break_ind[i]:break_ind[i+1]],'k',alpha=0.8,marker='.')

            if frame_bin > 1:
                for i in range(len(raw_break_ind)-1):
                    axins.plot(rawtime[raw_break_ind[i]:raw_break_ind[i+1]],rawf[raw_break_ind[i]:raw_break_ind[i+1]],
                                'k',alpha=0.3,marker='.',ls='')

            # Change the x and y limits of the inset axes to focus on the event #
            if (frame_end - frame_start) > 2:
                duration = time[frame_end] - time[frame_start]
            else:
                duration = 2
            fe = frame_end + 20
            if fe >= len(time):
                fe = len(time) - 1
            duration = int(event['frame_duration'])
            if duration < 4:
                duration = 4
            xmin = frame_start - 3*duration
            xmax = frame_end + 3*duration
            if xmin < 0:
                xmin = 0
            if xmax >= len(time):
                xmax = len(time) - 1
            xmin = time[frame_start] - (3*duration * cadence)
            xmax = time[frame_end] + (3*duration * cadence)
            if xmin <= 0:
                xmin = 0
            if xmax >= np.nanmax(time):
                xmax = np.nanmax(time)
            axins.set_xlim(xmin,xmax)
            axins.set_ylim(insert_ylims[0],insert_ylims[1])

            # Colour the inset axes spines #
            mark_inset(ax[1], axins, loc1=3, loc2=4, fc="none", ec="r",lw=2)
            plt.setp(axins.spines.values(), color='r',lw=2)
            plt.setp([axins.get_xticklines(), axins.get_yticklines()], color='C3')

            # Define max and min brightness for frame plot based on closer 5x5 cutout of brightest frame #
            bright_frame = flux[brightestframe,y-1:y+2,x-1:x+2]   
            vmin = np.percentile(flux[brightestframe],16)
            try:
                vmax = np.percentile(bright_frame,80)
            except:
                vmax = vmin + 20
            if vmin >= vmax:
                vmin = vmax - 5

            # Define and imshow the cutout image (19x19) #
            ymin = y - 9
            if ymin < 0:
                ymin = 0 
            xmin = x -9
            if xmin < 0:
                xmin = 0
            cutout_image = flux[:,ymin:y+10,xmin:x+10]
            ax[2].imshow(cutout_image[brightestframe],cmap='gray',origin='lower',vmin=vmin,vmax=vmax)
            ax[2].scatter(event.xcentroid - xmin, event.ycentroid - ymin, color='r', s=50, marker='x', lw=2)

            # Add labels, remove axes #
            ax[2].set_title('Brightest image',fontsize=15)
            ax[2].get_xaxis().set_visible(False)
            ax[2].get_yaxis().set_visible(False)
            ax[3].get_xaxis().set_visible(False)
            ax[3].get_yaxis().set_visible(False)
            
            # Find the first frame after the brightest frame that is at least 1 hour later #
            try:
                tdiff = np.where(time-time[brightestframe] >= 1/24)[0][0]
            except:
                tdiff = np.where(time[brightestframe] - time >= 1/24)[0][-1]
            after = tdiff
            if after >= len(cutout_image):
                after = len(cutout_image) - 1 

            # Plot the cutout image 1 hour later #
            ax[3].imshow(cutout_image[after],
                        cmap='gray',origin='lower',vmin=vmin,vmax=vmax)
            
            ax[3].set_title('1 hour later',fontsize=15)
            ax[3].annotate('', xy=(0.2, 1.15), xycoords='axes fraction', xytext=(0.2, 1.), 
                                arrowprops=dict(arrowstyle="<|-", color='r',lw=3))
            ax[3].annotate('', xy=(0.8, 1.15), xycoords='axes fraction', xytext=(0.8, 1.), 
                                arrowprops=dict(arrowstyle="<|-", color='r',lw=3))
            

            # Add 3x3 rectangle around the centre of the cutout image #
            if zoo_mode:
                rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin),5,5, linewidth=3, edgecolor='r', facecolor='none')
                ax[2].add_patch(rect)
                rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin),5,5, linewidth=3, edgecolor='r', facecolor='none')
                ax[3].add_patch(rect)
            else:
                # Draw base square with left/bottom red
                rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin), 5, 5,linewidth=3, edgecolor='r', facecolor='none')
                ax[2].add_patch(rect)
                # Overlay cyan top/right edge
                ax[2].add_line(Line2D([x - 2.5 - xmin, x + 2.5 - xmin],[y + 2.5 - ymin, y + 2.5 - ymin],color='c', linewidth=3))
                ax[2].add_line(Line2D([x + 2.5 - xmin, x + 2.5 - xmin],[y - 2.5 - ymin, y + 2.5 - ymin],color='c', linewidth=3))
                
                rect = patches.Rectangle((x-2.5 - xmin, y-2.5 - ymin), 5, 5,linewidth=3, edgecolor='r', facecolor='none')
                ax[3].add_patch(rect)
                # Overlay cyan top/right edge
                ax[3].add_line(Line2D([x - 2.5 - xmin, x + 2.5 - xmin],[y + 2.5 - ymin, y + 2.5 - ymin],color='c', linewidth=3))
                ax[3].add_line(Line2D([x + 2.5 - xmin, x + 2.5 - xmin],[y - 2.5 - ymin, y + 2.5 - ymin],color='c', linewidth=3))
                
            # Save the figure if a save path is provided #
            if save_path is not None:
                save_name = f'{save_path}_object{objid}'

                                                                            
                if eventtype == 'all':
                    plt.savefig(f'{save_name}_all_events.png', bbox_inches = "tight")
                else:
                    plt.savefig(f'{save_name}_event{eventid}of{total_events}.png', 
                                bbox_inches = "tight")
                    
                                            
        return [times,f], cutout_image, fig


    def plot_object(self,objid,event='separate',cut=None,save_name=None,save_path=None,phot_check='gaia_local',
                    latex=True,zoo_mode=False,external_phot=False,save_combined_path=None,tess_grid=3):
        """
        Plot the lightcurve and images of a given object/event.
        """
        
        # -- Check for saving information -- #
        if (save_combined_path is not None) & (not external_phot):
            print('Warning: save_combined_path is given, but external_phot is False. This will not save the photometry cutout.')
            print('Setting save_combined_path to None.')
            save_combined_path = None

        # -- Use Latex in the plots -- #
        if latex:
            plt.rc('text', usetex=latex)
                                                   
        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')
        elif cut != self.cut:
            self.gather_data(cut)
            self.gather_results(cut)

        # -- If saving is desired -- #
        if save_path is not None:
            from .tools import _Check_dirs

            if save_path[-1] != '/':
                save_path+='/'
            _Check_dirs(save_path)
            if save_name is None:
                save_name = f'Sec{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}'
            save_path = save_path + save_name

        # -- Isolate object and send to plotting function -- #
        obj = self.objects[self.objects.objid==objid].iloc[0]
        obj.lc,obj.cutout,obj.lc_fig = Navigator.Plot_Object_LC_Frame(self.sector,self.cam,self.time,self.flux,
                                                                      self.events,objid,event,save_path,latex,zoo_mode) 

        # -- If external photometry is requested, generate the WCS and cutout -- #
        if external_phot:
            if phot_check == 'local':
                phot_check = f'{self.path}/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
            fig, cat, coord,_,_ = self.external_photometry(objid,event,cut,tess_grid=tess_grid,check=phot_check)
            if fig is None:
                return obj
            
            obj.photometry = fig
            obj.cat = cat
            obj.coord = coord

        # -- Save a combined image with both the lightcurve and external photometry together -- #
        if save_combined_path is not None:
            from .tools import _Save_space
            import io
            from PIL import Image

            _Save_space(save_combined_path)

            buf1 = io.BytesIO()
            buf2 = io.BytesIO()
            obj.lc_fig.savefig(buf1, format='png', dpi=150,bbox_inches='tight')
            obj.photometry.savefig(buf2, format='png', dpi=150,bbox_inches='tight')

            # Load with Pillow
            img1 = Image.open(buf1)
            img2 = Image.open(buf2)

            # Match heights instead of widths
            max_height = max(img1.height, img2.height)
            if img1.height != max_height:
                img1 = img1.resize((int(img1.width * max_height / img1.height), max_height), Image.LANCZOS)
            if img2.height != max_height:
                img2 = img2.resize((int(img2.width * max_height / img2.height), max_height), Image.LANCZOS)

            # Combine horizontally
            combined_width = int((img1.width + img2.width)*1.05)
            combined_img = Image.new('RGB', (combined_width, max_height), (255, 255, 255))
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (int(img1.width+combined_width/22), 0))

            # Save final combined PNG
            combined_img.save(f"{save_combined_path}/S{self.sector}C{self.cam}C{self.ccd}C{self.cut}O{objid}E{event}.png", dpi=(150,150))
        
        return obj