from .tools import delete_files, _Print_buff, _Save_space

from time import time as t
from time import sleep

from glob import glob
import os

import numpy as np


class Tessellate():
    """
    Parent Class for Tessellation runs.
    """

    def __init__(self,data_path,sector=None,cam=None,ccd=None,n=None,
                 verbose=2,download_number=None,cuts=None,
                 job_output_path=None,working_path=None,
                 cube_time=None,cube_mem=None,cut_time=None,cut_mem=None,
                 reduce_time=None,reduce_cpu=None,search_time=None,search_cpu=None,
                 plot_time=None,plot_cpu=None,
                 download=None,make_cube=None,make_cuts=None,reduce=None,search=None,
                 plot=None,delete=None,overwrite=None,reset_logs=None,detect_mode='both',time_bin=None,
                 go=True):
        
        """
        Initialise.
        
        ------
        Inputs
        ------
        data_path : str 
            /path/to/data/location (use '/fred/oz335/TESSdata')
        job_output_path : str
            /path/to/job_log/output/location
        working_path : str
            /path/to/working_directory
        
        The Remaining INPUTS/OPTIONS/PROPERTIES can == 'None', will require input upon initialisation.

        sector : int
            TESS sector to operate upon
        cam : int, 'all'
            desired cam(s)
        ccd : int, 'all'
            desired ccd(s)

        ------
        Options
        ------

        n : int
            n^2 number of cuts will be made
        verbose : int
            0 = nothing, 1 = some output, 2 = full output
        download_number : 'all', int
            number of FFIs to download and operate on
        cuts : 'all', int
            specific cuts to operate upon per cam/ccd

        download : bool
            if True, download FFIs
        make_cube : bool
            if True, make cube
        make_cuts : bool
            if True, make cuts
        reduce : bool
            if True, reduce cuts
        search : bool
            if True, run search pipeline
        delete : bool
            if True, delete FFIs after process
        
        ------
        Run Properties
        ------

        cube_time : str
            time allowed for cube generation, form = 'hh:mm:ss'
        cube_cpu : int
            number of cpu to request for cube generation
        cube_mem : int
            memory/cpu for cube generation (in GB)

        cut_time : str
            time allowed for cut generation, form = 'hh:mm:ss'
        cut_cpu : int
            number of cpu to request for cut generation
        cut_mem : int
            memory/cpu for cut generation (in GB)

        reduce_time : str
            time allowed for cut reduction, form = 'hh:mm:ss'
        reduce_cpu : int
            number of cpu to request for cut reduction
        reduce_mem : int
            memory/cpu for cut reduction (in GB)

        search_time : str
            time allowed for transient event search, form = 'hh:mm:ss'
        search_cpu : int
            number of cpu to request for transient event search
        search_mem : int
            memory/cpu for transient event search (in GB)

        """
        
        if (job_output_path is None) | (working_path is None):
            m = 'Ensure you specify paths for job output and working path!\n'
            raise ValueError(m)
        else:
            self.job_output_path = job_output_path
            self.working_path = working_path

        self.sector = sector
        self.data_path = data_path

        self.cam = cam
        self.ccd = ccd
        self.verbose=verbose

        self.download_number = download_number
        self.n = n
        self.cuts = cuts

        self.cube_time = cube_time
        self.cube_mem = cube_mem
        self.cube_cpu = None

        self.cut_time = cut_time
        self.cut_mem = cut_mem
        self.cut_cpu = None

        self.reduce_time = reduce_time
        self.reduce_cpu = reduce_cpu
        self.reduce_mem = None

        self.search_time = search_time
        self.search_mem = None
        self.search_cpu = search_cpu
        
        self.plot_time = plot_time
        self.plot_mem = None
        self.plot_cpu = plot_cpu
        self.detect_mode = detect_mode
        self.time_bin = time_bin

        self.skip = []

        # -- Allows for no actual initialisation (TessTransient) -- #
        if go:
            # -- Confirm Run Properties -- #
            message = self._run_properties() 

            # -- Get time/cpu/memory suggestions depending on sector -- #
            suggestions = self._sector_suggestions()  

            # -- Ask for which tessellation steps to perform -- #
            message, download, make_cube, make_cuts, reduce, search, plot, delete = self._which_processes(message,download, make_cube, make_cuts, reduce, search, plot, delete)

            # -- Ask for inputs -- #
            if download:
                message = self._download_properties(message)

            if make_cube:
                message = self._cube_properties(message,suggestions[0])
                _Save_space(f'{job_output_path}/tessellate_cubing_logs')

            if make_cuts:
                message = self._cut_properties(message,suggestions[1])
                _Save_space(f'{job_output_path}/tessellate_cutting_logs')

            if reduce:
                message = self._reduce_properties(message,make_cuts,suggestions[2])
                _Save_space(f'{job_output_path}/tessellate_reduction_logs')

            if search:
                cutting_reducing = make_cuts | reduce
                message = self._search_properties(message,cutting_reducing,suggestions[3])
                _Save_space(f'{job_output_path}/tessellate_search_logs')

            if plot:
                message = self._plotting_properties(message,search,suggestions[4])
                _Save_space(f'{job_output_path}/tessellate_plotting_logs')

            # -- Check for overwriting -- #
            if overwrite != False:
                message = self._overwrite_suggestions(message, make_cube, make_cuts, reduce, search,plot)
            else:
                self.overwrite = None

            # -- Reset Job Logs -- #
            if reset_logs != False:
                message = self._reset_logs(message,make_cube,make_cuts,reduce,search,plot)

            # -- Run Processes -- #
            if download:
                self.download(message)

            if make_cube:
                self.make_cube()
            
            if make_cuts:
                self.make_cuts(cubing=make_cube)

            if reduce:
                self.reduce()    

            if search:
                self.transient_search(reducing=reduce) 
            
            if plot:
                self.transient_plot(searching=search)

            if delete:
                delete_files(filetype='ffis',data_path=self.data_path,sector=self.sector,n=self.n,part=False)  

    def _sector_suggestions(self):
        """
        Generate suggestions for slurm job runtime, cpu allocation, memory based on sector (tested but temperamental)
        """

        primary_mission = range(1,28)       # ~1200 FFIs , 30 min cadence
        secondary_mission = range(28,56)    # ~3600 FFIs , 10 min cadence
        tertiary_mission = range(56,100)    # ~12000 FFIs , 200 sec cadence

        self.part = False
        if self.sector in primary_mission:
            cube_time_sug = '45:00'
            cube_mem_sug = '20G'
            cube_mem_req = 60

            cut_time_sug = '20:00'
            cut_mem_sug = '20G'
            cut_mem_req = 20

            reduce_time_sug = '1:00:00'
            reduce_cpu_sug = '32'
            reduce_mem_req = 90

            search_time_sug = '20:00'
            search_cpu_sug = '32'
            search_mem_req = 50
            
            plot_time_sug = '20:00'
            plot_cpu_sug = '32'
            plot_mem_req = 50

        elif self.sector in secondary_mission:
            cube_time_sug = '1:45:00'
            cube_mem_sug = '20G'
            cube_mem_req = 140

            cut_time_sug = '2:00:00'
            cut_mem_sug = '20G'
            cut_mem_req = 60

            reduce_time_sug = '1:15:00'
            reduce_cpu_sug = '32'
            reduce_mem_req = 160

            search_time_sug = '30:00'
            search_cpu_sug = '32'
            search_mem_req = 70
            
            plot_time_sug = '20:00'
            plot_cpu_sug = '32'
            plot_mem_req = 70

        elif self.sector in tertiary_mission:
            self.part = True

            cube_time_sug = '6:00:00'
            cube_mem_sug = '20G'
            cube_mem_req = 200

            cut_time_sug = '3:00:00'
            cut_mem_sug = '20G'
            cut_mem_req = 100

            reduce_time_sug = '3:00:00'
            reduce_cpu_sug = '32'
            reduce_mem_req = 200

            search_time_sug = '1:00:00'
            search_cpu_sug = '32'
            search_mem_req = 60
            
            plot_time_sug = '20:00'
            plot_cpu_sug = '32'
            plot_mem_req = 50

        suggestions = [[cube_time_sug,cube_mem_sug,cube_mem_req],
                       [cut_time_sug,cut_mem_sug,cut_mem_req],
                       [reduce_time_sug,reduce_cpu_sug,reduce_mem_req],
                       [search_time_sug,search_cpu_sug,search_mem_req],
                       [plot_time_sug,plot_cpu_sug,plot_mem_req]]
        
        return suggestions

    def _run_properties(self):
        """
        Confirm sector, cam, ccd properties.
        """

        message = ''

        print('\n')
        print(_Print_buff(50,f'Initialising Tessellation Run'))
        message += _Print_buff(50,f'Initialising Tessellation Run')+'\n'

        if self.sector is None:
            sector = input('   - Sector = ')
            message += f'   - Sector = {sector}\n'
            done = False
            while not done:
                try:
                    sector = int(sector)
                    if 0 < sector < 100:
                        self.sector = sector
                        done = True
                    else:
                        sector = input('      Invalid choice! Sector = ')
                        message += f'      Invalid choice! Sector = {sector}\n'
                except:
                    sector = input('      Invalid choice! Sector = ')
                    message += f'      Invalid choice! Sector = {sector}\n'
        elif 0 < self.sector < 100:
            print(f'   - Sector = {self.sector}')
            message += f'   - Sector = {self.sector}\n'
        else:
            e = f'Invalid Sector Input of {self.sector}\n'
            raise ValueError(e)


        if self.cam is None:
            cam = input('   - Cam [1,2,3,4,all] = ')
            message += f'   - Cam [1,2,3,4,all] = {cam}\n'
            done = False
            while not done:
                if cam in ['1','2','3','4']:
                    self.cam = [int(cam)]
                    done = True
                elif cam.lower() == 'all':
                    self.cam = [1,2,3,4]
                    done = True
                elif cam == '34':
                    self.cam = [3,4]
                    done = True
                else:
                    cam = input('      Invalid choice! Cam [1,2,3,4,all] = ')
                    message += f'      Invalid choice! Cam [1,2,3,4,all] = {cam}\n'
        elif self.cam == 'all':
            print(f'   - Cam = all')
            message += '   - Cam = all\n'
            self.cam = [1,2,3,4]
        elif self.cam in [1,2,3,4]:
            print(f'   - Cam = {self.cam}')
            self.cam = [self.cam]
            message += f'   - Cam = {self.cam}\n'
        else:
            e = f'Invalid Camera Input of {self.cam}\n'
            raise ValueError(e)
        

        if self.ccd is None:
            ccd = input('   - CCD [1,2,3,4,all] = ')
            message += f'   - CCD [1,2,3,4,all] = {ccd}\n'
            done = False
            while not done:
                if ccd in ['1','2','3','4']:
                    self.ccd = [int(ccd)]
                    done = True
                elif ccd.lower() == 'all':
                    self.ccd = [1,2,3,4]
                    done = True
                else:
                    ccd = input('      Invalid choice! CCD [1,2,3,4,all] = ')
                    message += f'      Invalid choice! CCD [1,2,3,4,all] = {ccd}\n'
        elif self.ccd == 'all':
            print(f'   - CCD = all')
            message += '   - CCD = all\n'
            self.ccd = [1,2,3,4]
        elif self.ccd in [1,2,3,4]:
            print(f'   - CCD = {self.ccd}')
            self.ccd = [self.ccd]
            message += f'   - CCD = {self.ccd}\n'
        else:
            e = f'Invalid CCD Input of {self.ccd}\n'
            raise ValueError(e)
        
        print('\n')
        message += '\n'

        return message
    
    def _which_processes(self,message,download,make_cube,make_cuts,reduce,search,plot,delete):

        if download is None:
            d = input('   - Download FFIs? [y/n] = ')
            message += f'   - Download FFIs? [y/n] = {d}\n'
            done = False
            while not done:
                if d.lower() == 'y':
                    download = True
                    done=True
                elif d.lower() == 'n':
                    download = False
                    done=True
                else:
                    d = input('      Invalid choice! Download FFIs? [y/n] = ')
                    message += f'      Invalid choice! Download FFIs? [y/n] = {d}\n'

        if make_cube is None:
            d = input('   - Make Cube(s)? [y/n] = ')
            message += f'   - Make Cube(s)? [y/n] = {d}\n'
            done = False
            while not done:
                if d.lower() == 'y':
                    make_cube = True
                    done=True
                elif d.lower() == 'n':
                    make_cube = False
                    done=True
                else:
                    d = input('      Invalid choice! Make Cube(s)? [y/n] = ')
                    message += f'      Invalid choice! Make Cube(s)? [y/n] = {d}\n'

        if make_cuts is None:
            d = input('   - Make Cut(s)? [y/n] = ')
            message += f'   - Make Cut(s)? [y/n] = {d}\n'
            done = False
            while not done:
                if d.lower() == 'y':
                    make_cuts = True
                    done=True
                elif d.lower() == 'n':
                    make_cuts = False
                    done=True
                else:
                    d = input('      Invalid choice! Make Cut(s)? [y/n] = ')
                    message += f'      Invalid choice! Make Cut(s)? [y/n] = {d}\n'

        if reduce is None:
            d = input('   - Reduce Cut(s)? [y/n] = ')
            message += f'   - Reduce Cut(s)? [y/n] = {d}\n'
            done = False
            while not done:
                if d.lower() == 'y':
                    reduce = True
                    done=True
                elif d.lower() == 'n':
                    reduce = False
                    done=True
                else:
                    d = input('      Invalid choice! Reduce Cut(s)? [y/n] = ')
                    message += f'      Invalid choice! Reduce Cut(s)? [y/n] = {d}\n'
            
        if search is None:
            d = input('   - Run Transient Search on Cut(s)? [y/n] = ')
            message += f'   - Run Transient Search on Cut(s)? [y/n] = {d}\n'
            done = False
            while not done:
                if d.lower() == 'y':
                    search = True
                    done=True
                elif d.lower() == 'n':
                    search = False
                    done=True
                else:
                    d = input('      Invalid choice! Run Transient Search on Cut(s)? [y/n] = ')
                    message += f'      Invalid choice! Run Transient Search on Cut(s)? [y/n] = {d}\n'
        
        if plot is None:
            d = input('   - Run Transient Plotting on Cut(s)? [y/n] = ')
            message += f'   - Run Transient Plotting on Cut(s)? [y/n] = {d}\n'
            done = False
            while not done:
                if d.lower() == 'y':
                    plot = True
                    done=True
                elif d.lower() == 'n':
                    plot = False
                    done=True
                else:
                    d = input('      Invalid choice! Run Transient Plotting on Cut(s)? [y/n] = ')
                    message += f'      Invalid choice! Run Transient Plotting on Cut(s)? [y/n] = {d}\n'
            

        if delete is None:
            d = input('   - Delete all FFIs upon completion? [y/n] = ')
            message += f'   - Delete all FFIs upon completion? [y/n] = {d}\n'
            done = False
            while not done:
                if d.lower() == 'y':
                    delete = True
                    done=True
                elif d.lower() == 'n':
                    delete = False
                    done=True
                else:
                    d = input('      Invalid choice! Delete all FFIs upon completion? [y/n] = ')
                    message += f'      Invalid choice! Delete all FFIs upon completion? [y/n] = {d}\n'

        print('\n')
        message += '\n'

        return message, download, make_cube, make_cuts, reduce, search, plot, delete
    
    def _download_properties(self,message):
        """
        Confirm download process properties.
        """

        if self.download_number is None:
            dNum = input("   - Download Number [int,all] = ")
            message += f"   - Download Number [int,all] = {dNum}\n"
            done = False
            while not done:
                try:
                    dNum = int(dNum)
                    if 0 < dNum < 14000:
                        self.download_number = dNum
                        done = True
                    else:
                        dNum = input("      Invalid choice! Download Number [int,all] = ")
                        message += f"      Invalid choice! Download Number [int,all] = {dNum}\n"
                except:
                    if dNum == 'all':
                        self.download_number = dNum
                        done = True
                    else:
                        dNum = input("      Invalid choice! Download Number [int,all] = ")
                        message += f"      Invalid choice! Download Number [int,all] = {dNum}\n"
        elif self.download_number == 'all':
            print(f'   - Download Number = all')
            message += f'   - Download Number = all\n'
        elif 0 < self.download_number < 14000:
            print(f'   - Download Number = {self.download_number}')
            message += f'   - Download Number = {self.download_number}\n'
        else:
            e = f'Invalid Download Number Input of {self.download_number}\n'
            raise ValueError(e)
        
        print('\n')
        message += '\n'

        return message
    
    def _cube_properties(self,message,suggestions):
        """
        Confirm cube generation process properties.
        """

        if self.cube_time is None:
            cube_time = input(f"   - Cube Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
            message += f"   - Cube Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {cube_time}\n"
            done = False
            while not done:
                if ':' in cube_time:
                    self.cube_time = cube_time
                    done = True
                else:
                    cube_time = input(f"      Invalid format! Cube Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                    message += f"      Invalid choice! Cube Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {cube_time}\n"
        else:
            print(f'   - Cube Batch Time = {self.cube_time}')
            message += f"   - Cube Batch Time = {self.cube_time}')\n"

        
        if self.cube_mem is None:
            cube_mem = input(f"   - Cube Mem/CPU ({suggestions[1]} suggested) = ")
            message += f"   - Cube Mem/CPU ({suggestions[1]} suggested) = {cube_mem}\n"
            done = False
            while not done:
                try: 
                    cube_mem = int(cube_mem)
                    if 0<cube_mem < 500:
                        self.cube_mem = cube_mem
                        done=True
                    else:
                        cube_mem = input(f"      Invalid format! Cube Mem/CPU ({suggestions[1]} suggested) = ")
                        message += f"      Invalid choice! Cube Mem/CPU ({suggestions[1]} suggested) = {cube_mem}\n"
                except:
                    if cube_mem[-1].lower() == 'g':
                        self.cube_mem = cube_mem[:-1]
                        done = True
                    else:
                        cube_mem = input(f"      Invalid format! Cube Mem/CPU ({suggestions[1]} suggested) = ")
                        message += f"      Invalid choice! Cube Mem/CPU ({suggestions[1]} suggested) = {cube_mem}\n"            

        elif 0 < self.cube_mem < 500:
            print(f'   - Cube Mem/CPU = {self.cube_mem}G')
            message += f"   - Cube Mem/CPU = {self.cube_mem}G\n"
        else:
            e = f"Invalid Cube Mem/CPU Input of {self.cube_mem}\n"
            raise ValueError(e)
        
                
        if type(self.download_number) == int:
            cube_cpu = input(f"   - Cube Num CPUs [1-32] = ")
            message += f"   - Cube Num CPUs [1-32] = {cube_cpu}\n"
            done = False
            while not done:
                try:
                    cube_cpu = int(cube_cpu)
                    if 0 < cube_cpu < 33:
                        self.cube_cpu = cube_cpu
                        done = True
                    else:
                        cube_cpu = input(f"      Invalid format! Cube Num CPUs [1-32] = ")
                        message += f"      Invalid choice! Cube Num CPUs [1-32] = {cube_cpu}\n"
                except:
                    cube_cpu = input(f"      Invalid format! Cube Num CPUs [1-32] = ")
                    message += f"      Invalid choice! Cube Num CPUs [1-32] = {cube_cpu}\n"
        else:
            if suggestions[2] % int(self.cube_mem) != 0:
                self.cube_cpu = suggestions[2] // int(self.cube_mem) + 1
            else:
                self.cube_cpu = suggestions[2] // int(self.cube_mem)

            print(f'   - Cube Num CPUs Needed = {self.cube_cpu}')
            message += f"   - Cube Num CPUs Needed = {self.cube_cpu}\n"

        print('\n')
        message += '\n'

        return message

    def _cut_properties(self,message,suggestions):
        """
        Confirm cut generation process properties.
        """

        if self.n is None:
            n = input('   - n (Number of Cuts = n^2) = ')
            message += f'   - n (Number of Cuts = n^2) = {n}\n'
            done = False
            while not done:
                try:
                    n = int(n)
                    if n > 0:
                        self.n = n
                        done = True
                    else:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                        message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
                except:
                    n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                    message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
        elif self.n > 0:
            print(f'   - n (Number of Cuts = n^2) = {self.n}')
            message += f'  - n (Number of Cuts = n^2) = {self.n}\n'
        else:
            e = f"Invalid 'n' value Input of {self.n}\n"
            raise ValueError(e)
        
        
        if self.cuts is None:
            cut = input(f'   - Cut [1-{self.n**2},all] = ')
            message += f'   - Cut [1-{self.n**2},all] = {cut}\n'
            done = False
            while not done:
                if cut == 'all':
                    self.cuts = range(1,self.n**2+1)
                    done = True
                elif cut in np.array(range(1,self.n**2+1)).astype(str):
                    self.cuts = [int(cut)]
                    done = True
                else:
                    cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
                    message += f'      Invalid choice! Cut [1-{self.n**2},all] = {cut}\n'
        elif self.cuts == 'all':
            print(f'   - Cut = all')
            message += f'   - Cut = all\n'
            self.cuts = range(1,self.n**2+1)
        elif self.cuts in range(1,self.n**2+1):
            print(f'   - Cut = {self.cuts}')
            message += f'   - Cut = {self.cuts}\n'
            self.cuts = [self.cuts]  
        else:
            e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
            raise ValueError(e)
        
        
        print('\n')
        message += '\n'


        if self.cut_time is None:
            cut_time = input(f"   - Cut Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
            message += f"   - Cut Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {cut_time}\n"
            done = False
            while not done:
                if ':' in cut_time:
                    self.cut_time = cut_time
                    done = True
                else:
                    cut_time = input(f"      Invalid format! Cut Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                    message += f"      Invalid choice! Cut Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {cut_time}\n"
        else:
            print(f'   - Cut Batch Time = {self.cut_time}')
            message += f"   - Cut Batch Time = {self.cut_time}')\n"

        
        if self.cut_mem is None:
            cut_mem = input(f"   - Cut Mem/CPU ({suggestions[1]} suggested) = ")
            message += f"   - Cut Mem/CPU ({suggestions[1]} suggested) = {cut_mem}\n"
            done = False
            while not done:
                try: 
                    cut_mem = int(cut_mem)
                    if 0<cut_mem < 500:
                        self.cut_mem = cut_mem
                        done=True
                    else:
                        cut_mem = input(f"      Invalid format! Cut Mem/CPU ({suggestions[1]} suggested) = ")
                        message += f"      Invalid choice! Cut Mem/CPU ({suggestions[1]} suggested) = {cut_mem}\n"
                except:
                    if cut_mem[-1].lower() == 'g':
                        self.cut_mem = cut_mem[:-1]
                        done = True
                    else:
                        cut_mem = input(f"      Invalid format! Cut Mem/CPU ({suggestions[1]} suggested) = ")
                        message += f"      Invalid choice! Cut Mem/CPU ({suggestions[1]} suggested) = {cut_mem}\n"

        elif 0 < self.cut_mem < 500:
            print(f'   - Cut Mem/CPU = {self.cut_mem}G')
            message += f"   - Cut Mem/CPU = {self.cut_mem}G\n"
        else:
            e = f"Invalid Cut Mem/CPU Input of {self.cut_mem}\n"
            raise ValueError(e)
        
        
        if type(self.download_number) == int:
            cut_cpu = input(f"   - Cut Num CPUs [1-32] = ")
            message += f"   - Cut Num CPUs [1-32] = {cut_cpu}\n"
            done = False
            while not done:
                try:
                    cut_cpu = int(cut_cpu)
                    if 0 < cut_cpu < 33:
                        self.cut_cpu = cut_cpu
                        done = True
                    else:
                        cut_cpu = input(f"      Invalid format! Cut Num CPUs [1-32] = ")
                        message += f"      Invalid choice! Cut Num CPUs [1-32] = {cut_cpu}\n"
                except:
                    cut_cpu = input(f"      Invalid format! Cut Num CPUs [1-32] = ")
                    message += f"      Invalid choice! Cut Num CPUs [1-32] = {cut_cpu}\n"
        else:
            if suggestions[2] % int(self.cut_mem) != 0:
                self.cut_cpu = suggestions[2] // int(self.cut_mem) + 1
            else:
                self.cut_cpu = suggestions[2] // int(self.cut_mem)

            print(f'   - Cut Num CPUs Needed = {self.cut_cpu}')
            message += f"   - Cut Num CPUs Needed = {self.cut_cpu}\n"

        print('\n')
        message += '\n'

        return message

    def _reduce_properties(self,message,cutting,suggestions):
        """
        Confirm reduction process properties.
        """

        if not cutting:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                message += f'   - n (Number of Cuts = n^2) = {n}\n'
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                            message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                        message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
                message += f'  - n (Number of Cuts = n^2) = {self.n}\n'
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)
            
            
            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                message += f'   - Cut [1-{self.n**2},all] = {cut}\n'
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
                        message += f'      Invalid choice! Cut [1-{self.n**2},all] = {cut}\n'
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                message += f'   - Cut = all\n'
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                message += f'   - Cut = {self.cuts}\n'
                self.cuts = [self.cuts]  
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)
            
            
            print('\n')
            message += '\n'


        if self.reduce_time is None:
            reduce_time = input(f"   - Reduce Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
            message += f"   - Reduce Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {reduce_time}\n"
            done = False
            while not done:
                if ':' in reduce_time:
                    self.reduce_time = reduce_time
                    done = True
                else:
                    reduce_time = input(f"      Invalid format! Reduce Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                    message += f"      Invalid choice! Reduce Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {reduce_time}\n"
        else:
            print(f'   - Reduce Batch Time = {self.reduce_time}')
            message += f"   - Reduce Batch Time = {self.reduce_time}')\n"

        
        if self.reduce_cpu is None:
            reduce_cpu = input(f"   - Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = ")
            message += f"   - Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = {reduce_cpu}\n"
            done = False
            while not done:
                try:
                    reduce_cpu = int(reduce_cpu)
                    if 0 < reduce_cpu < 33:
                        self.reduce_cpu = reduce_cpu
                        done = True
                    else:
                        reduce_cpu = input(f"      Invalid format! Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                        message += f"      Invalid choice! Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = {reduce_cpu}\n"
                except:
                    reduce_cpu = input(f"      Invalid format! Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    message += f"      Invalid choice! Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = {reduce_cpu}\n"
        elif 0 < self.reduce_cpu < 33:
            print(f'   - Reduce Num CPUs = {self.reduce_cpu}')
            message += f"   - Reduce Num CPUs = {self.reduce_cpu}\n"
        else:
            e = f"Invalid Reduce CPUs Input of {self.reduce_cpu}\n"
            raise ValueError(e)
        

        if type(self.download_number) == int:
            reduce_mem = input(f"   - Reduce Mem/CPU = ")
            message += f"   - Reduce Mem/CPU = {reduce_mem}\n"
            done = False
            while not done:
                try:
                    reduce_mem = int(reduce_mem)
                    if 0<reduce_mem < 500:
                        self.reduce_mem = reduce_mem
                        done=True
                    else:
                        reduce_mem = input(f"      Invalid format! Reduce Mem/CPU = ")
                        message += f"      Invalid choice! Reduce Mem/CPU = {reduce_mem}\n"
                except:
                    if reduce_mem[-1].lower() == 'g':
                        self.reduce_mem = reduce_mem[:-1]
                        done = True
                    else:
                        reduce_mem = input(f"      Invalid format! Reduce Mem/CPU = ")
                        message += f"      Invalid choice! Reduce Mem/CPU = {reduce_mem}\n"
        else:
            self.reduce_mem = np.ceil(suggestions[2]/self.reduce_cpu).astype(int)
            print(f'   - Reduce Mem/CPU Needed = {self.reduce_mem}')
            message += f'   - Reduce Mem/CPU Needed = {self.reduce_mem}\n'    

        print('\n')
        message += '\n'

        return message
    
    def _search_properties(self,message,cutting_reducing,suggestions):
        """
        Confirm transient search process properties.
        """

        if not cutting_reducing:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                message += f'   - n (Number of Cuts = n^2) = {n}\n'
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                            message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                        message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
                message += f'  - n (Number of Cuts = n^2) = {self.n}\n'
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)
            
            
            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                message += f'   - Cut [1-{self.n**2},all] = {cut}\n'
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
                        message += f'      Invalid choice! Cut [1-{self.n**2},all] = {cut}\n'
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                message += f'   - Cut = all\n'
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                message += f'   - Cut = {self.cuts}\n'
                self.cuts = [self.cuts]  
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)
            
            
            print('\n')
            message += '\n'


        if self.search_time is None:
            search_time = input(f"   - Search Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
            message += f"   - Search Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {search_time}\n"
            done = False
            while not done:
                if ':' in search_time:
                    self.search_time = search_time
                    done = True
                else:
                    search_time = input(f"      Invalid format! Search Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                    message += f"      Invalid choice! Search Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {search_time}\n"
        else:
            print(f'   - Search Batch Time = {self.search_time}')
            message += f"   - Search Batch Time = {self.search_time}')\n"


        if self.search_cpu is None:
            search_cpu = input(f"   - Search Num CPUs [1-32] ({suggestions[1]} suggested) = ")
            message += f"   - Search Num CPUs [1-32] ({suggestions[1]} suggested) = {search_cpu}\n"
            done = False
            while not done:
                try:
                    search_cpu = int(search_cpu)
                    if 0 < search_cpu < 33:
                        self.search_cpu = search_cpu
                        done = True
                    else:
                        search_cpu = input(f"      Invalid format! Search Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                        message += f"      Invalid choice! Search Num CPUs [1-32] ({suggestions[1]} suggested) = {search_cpu}\n"
                except:
                    search_cpu = input(f"      Invalid format! Search Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    message += f"      Invalid choice! Search Num CPUs [1-32] ({suggestions[1]} suggested) = {search_cpu}\n"
        elif 0 < self.search_cpu < 33:
            print(f'   - Search Num CPUs = {self.search_cpu}')
            message += f"   - Search Num CPUs = {self.search_cpu}\n"
        else:
            e = f"Invalid Search CPUs Input of {self.search_cpu}\n"
            raise ValueError(e) 
        
        
        if type(self.download_number) == int:
            search_mem = input(f"   - Search Mem/CPU = ")
            message += f"   - Search Mem/CPU = {search_mem}\n"
            done = False
            while not done:
                try:
                    search_mem = int(search_mem)
                    if 0<search_mem < 500:
                        self.search_mem = search_mem
                        done=True
                    else:
                        search_mem = input(f"      Invalid format! Search Mem/CPU = ")
                        message += f"      Invalid choice! Search Mem/CPU = {search_mem}\n"
                except:
                    if search_mem[-1].lower() == 'g':
                        self.search_mem = search_mem[:-1]
                        done = True
                    else:
                        search_mem = input(f"      Invalid format! Search Mem/CPU = ")
                        message += f"      Invalid choice! Search Mem/CPU = {search_mem}\n"
        else:
            self.search_mem = np.ceil(suggestions[2]/self.search_cpu).astype(int)
            print(f'   - Search Mem/CPU Needed = {self.search_mem}')
            message += f'   - Search Mem/CPU Needed = {self.search_mem}\n'  

        print('\n')
        message += '\n'

        return message
    
    def _plotting_properties(self,message,searching,suggestions):
        """
        Confirm transient search process properties.
        """

        if not searching:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                message += f'   - n (Number of Cuts = n^2) = {n}\n'
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                            message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                        message += f'      Invalid choice! n (Number of Cuts = n^2) = {n}\n'
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
                message += f'  - n (Number of Cuts = n^2) = {self.n}\n'
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)
            
            
            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                message += f'   - Cut [1-{self.n**2},all] = {cut}\n'
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
                        message += f'      Invalid choice! Cut [1-{self.n**2},all] = {cut}\n'
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                message += f'   - Cut = all\n'
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                message += f'   - Cut = {self.cuts}\n'
                self.cuts = [self.cuts]  
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)
            
            
            print('\n')
            message += '\n'


        if self.plot_time is None:
            plot_time = input(f"   - Plotting Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
            message += f"   - Plotting Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {plot_time}\n"
            done = False
            while not done:
                if ':' in plot_time:
                    self.plot_time = plot_time
                    done = True
                else:
                    plot_time = input(f"      Invalid format! Plotting Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                    message += f"      Invalid choice! Plotting Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = {plot_time}\n"
        else:
            print(f'   - Plotting Batch Time = {self.plot_time}')
            message += f"   - Plotting Batch Time = {self.plot_time}')\n"


        if self.plot_cpu is None:
            plot_cpu = input(f"   - Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = ")
            message += f"   - PLotting Num CPUs [1-32] ({suggestions[1]} suggested) = {plot_cpu}\n"
            done = False
            while not done:
                try:
                    plot_cpu = int(plot_cpu)
                    if 0 < plot_cpu < 33:
                        self.plot_cpu = plot_cpu
                        done = True
                    else:
                        plot_cpu = input(f"      Invalid format! Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                        message += f"      Invalid choice! Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = {plot_cpu}\n"
                except:
                    plot_cpu = input(f"      Invalid format! Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    message += f"      Invalid choice! Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = {plot_cpu}\n"
        elif 0 < self.plot_cpu < 33:
            print(f'   - Plotting Num CPUs = {self.plot_cpu}')
            message += f"   - Plotting Num CPUs = {self.plot_cpu}\n"
        else:
            e = f"Invalid Plotting CPUs Input of {self.plot_cpu}\n"
            raise ValueError(e)
        
        
        if type(self.download_number) == int:
            plot_mem = input(f"   - Plot Mem/CPU = ")
            message += f"   - Plot Mem/CPU = {plot_mem}\n"
            done = False
            while not done:
                try:
                    plot_mem = int(plot_mem)
                    if 0<plot_mem < 500:
                        self.plot_mem = plot_mem
                        done=True
                    else:
                        plot_mem = input(f"      Invalid format! Plot Mem/CPU = ")
                        message += f"      Invalid choice! Plot Mem/CPU = {plot_mem}\n"
                except:
                    if plot_mem[-1].lower() == 'g':
                        self.plot_mem = plot_mem[:-1]
                        done = True
                    else:
                        plot_mem = input(f"      Invalid format! Plot Mem/CPU = ")
                        message += f"      Invalid choice! Search Mem/CPU = {plot_mem}\n"
        else:
            self.plot_mem = np.ceil(suggestions[2]/self.plot_cpu).astype(int)
            print(f'   - Plot Mem/CPU Needed = {self.plot_mem}')
            message += f'   - Plot Mem/CPU Needed = {self.plot_mem}\n'  

        print('\n')
        message += '\n'

        return message
    
    def _overwrite_suggestions(self,message, make_cube, make_cuts, reduce, search,plot):

        options = []
        if make_cube:
            options.append('cube')
        if make_cuts:
            options.append('cut')
        if reduce:
            options.append('reduce')
        if search:
            options.append('search')
        if plot:
            options.append('plot')

        done = False
        over = input(f'   - Overwrite any steps? [y,n,{str(options)[1:-1]}] = ')
        message += f'   - Overwrite any steps? [y,n,{str(options)[1:-1]}] = \n{over}'
        while not done:
            
            if over == 'y':
                self.overwrite = 'all'
                done = True
            elif over == 'n':
                self.overwrite = None
                done = True
            else:
                self.overwrite = (over.replace(' ','')).split(',')
                good = True
                for thing in self.overwrite:
                    if thing not in ['cube','cut','reduce','search','plot']:
                        good = False
                if good:
                    done = True
                    message += over
                else:
                    over = input(f"      Invalid choice! Overwrite any steps? [y,n,{str(options)[1:-1]}] = ")
                    message += f"      Invalid choice! Overwrite any steps? [y,n,{str(options)[1:-1]}] = \n"

        
        
        print('\n')
        message += '\n'

        return message
    
    def _reset_logs(self,message,make_cube,make_cuts,reduce,search,plot):
        """
        Reset slurm job logs in provided output path.
        """

        message += 'Delete Past Job Logs? [y/n] :\n'
        if input('Delete Past Job Logs? [y/n] :\n').lower() == 'y':
            if make_cube:
                os.system(f'rm -f {self.job_output_path}/tessellate_cubing_logs/*')
            if make_cuts:
                os.system(f'rm -f {self.job_output_path}/tessellate_cutting_logs/*')
            if reduce:
                os.system(f'rm -f {self.job_output_path}/tessellate_reduction_logs/*')
            if search:
                os.system(f'rm -f {self.job_output_path}/tessellate_search_logs/*')
            if plot:
                os.system(f'rm -f {self.job_output_path}/tessellate_plotting_logs/*')

            message = message + 'y \n'
        else:
            message + 'n \n'
        print('\n')
        message += '\n'

        return message
        
    def download(self,message):
        """
        Download FFIs! 
        (message only is for clarity, tqdm makes weird progress bars so message prints confirmations once job is done.)
        """

        from .dataprocessor import DataProcessor

        for cam in self.cam:
            for ccd in self.ccd:
                if len(glob(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/image_files/*ffic.fits')) > 1000:
                    print(f'Sector {self.sector} Cam {cam} CCD {ccd} data already downloaded!')
                    print('\n')
                else:
                    tDownload = t()
                    data_processor = DataProcessor(sector=self.sector,path=self.data_path,verbose=self.verbose)
                    data_processor.download(cam=cam,ccd=ccd,number=self.download_number)
                    os.system('clear')
                    print(message)
                    print(f'Sector {self.sector} Cam {cam} Ccd {ccd} Download Complete ({((t()-tDownload)/60):.2f} mins).')
                    print('\n')
                    message += f'Sector {self.sector} Cam {cam} Ccd {ccd} Download Complete ({((t()-tDownload)/60):.2f} mins).\n'
                    message += '\n'

    def make_cube(self,overwrite=True):
        """
        Make Cube! 

        Process : Generates a python script for generating cube, then generates and submits a slurm script to call the python script.
        """

        _Save_space(f'{self.working_path}/cubing_scripts')

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/cubing_scripts/S{self.sector}C*')

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('cube' in self.overwrite):
                delete_files('cubes',self.data_path,self.sector,self.n,self.cam,self.ccd,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd: 
                print(_Print_buff(60,f'Making Cube for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')

                # -- Generate Cube Path -- #
                cube_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt'
                if os.path.exists(cube_check):
                    print(f'Cam {cam} CCD {ccd} cube already exists!')
                    print('\n')
                else:

                    # -- Create python file for cubing-- # 
                    print(f'Creating Cubing Python File for Sector{self.sector} Cam{cam}Ccd{ccd}')
                    python_text = f"\
from tessellate import DataProcessor\n\
\n\
processor = DataProcessor(sector={self.sector},path='{self.data_path}',verbose=2)\n\
processor.make_cube(cam={cam},ccd={ccd},part={self.part})\n\
with open(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt', 'w') as file:\n\
    file.write('Cubed!')"   
                
                    with open(f"{self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.py", "w") as python_file:
                        python_file.write(python_text)

                    # -- Create bash file to submit job -- #
                    #print('Creating Cubing Batch File')
                    batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cubing\n\
#SBATCH --output={self.job_output_path}/tessellate_cubing_logs/%A_cubing_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_cubing_logs/%A_cubing_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.cube_time}\n\
#SBATCH --cpus-per-task={self.cube_cpu}\n\
#SBATCH --mem-per-cpu={self.cube_mem}G\n\
\n\
python {self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.py"

                    with open(f"{self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.sh", "w") as batch_file:
                        batch_file.write(batch_text)

                    # -- Submit job -- #
                    #print('Submitting Cubing Batch File')
                    os.system(f'sbatch {self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.sh')
                    print('\n')

    def _get_catalogues(self,cam,ccd,base_path):
        """
        Access internet, find Gaia sources and save for reduction.
        """

        from .dataprocessor import DataProcessor, tr
        from .catalog_queries import create_external_var_cat

        data_processor = DataProcessor(sector=self.sector,path=self.data_path,verbose=self.verbose)
        _,_,cutCentreCoords,rad = data_processor.find_cuts(cam=cam,ccd=ccd,n=self.n,plot=False)

        #image_path = glob(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/*ffic.fits')[0]

        l = self.cut_time.split(':')
        seconds = 1 * int(l[-1]) + 60 * int(l[-2])
        if len(l) == 3:
            seconds += 3600 * int(l[-3])

        # -- Waits for cuts to be completed. If max time exceeded , return not done so cut process is restarted -- #
        completed = []
        message = 'Waiting for Cuts'
        timeStart = t()
        done = True
        i = 0
        while len(completed) < len(self.cuts):
            if (t()-timeStart) > seconds + 600:
                done = False
                break
            else:
                if i > 0:
                    print(message, end='\r')
                    sleep(120)
                message += '.'
                for cut in self.cuts:
                    if cut not in completed:
                        save_path = f'{base_path}/Cut{cut}of{self.n**2}'
                        if os.path.exists(f'{save_path}/variable_catalog.csv'):
                            completed.append(cut)
                        elif os.path.exists(f'{save_path}/cut.txt'):
                            #try:
                            print(f'Generating Catalogues {cut}')
                            if os.path.exists(f'{save_path}/local_gaia_cat.csv'):
                                print('Gaia catalog already made, skipping.')
                            else:
                                rad = rad + 2*60/21
                                cutPath = f'{save_path}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}.fits'
                                tr.external_save_cat(tpf=cutPath,save_path=save_path,maglim=19) # oversize radius by 2 arcmin in terms of tess pixels
                            rad2 = rad*21/60**2
                            create_external_var_cat(center=cutCentreCoords[cut-1],size=rad2,save_path=save_path) # This one queries in degrees!!!!
                            completed.append(cut)
                            try:
                                os.system('rm -r ~/.astropy/cache/astroquery/Vizier/*.pickle')
                            except:
                                pass

                i += 1

        return done

    def make_cuts(self,cubing,overwrite=True):
        """
        Make cuts! 

            Cube_time_sug : dictates how long the cube process should run for, cubing is restarted if need be.
            Cut_time_sug : dictates how long the cut process should run for, cutting is restarted if need be.
        """

        _Save_space(f'{self.working_path}/cutting_scripts')

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/cutting_scripts/S{self.sector}C*')

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('cut' in self.overwrite):
                delete_files('cuts',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd: 
                print(_Print_buff(60,f'Making Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}')) 
                print('\n')
                cube_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt'
                if not os.path.exists(cube_check):
                    if not cubing:
                        e = 'No Cube File Detected to Cut!\n'
                        raise ValueError(e)
                    else:
                        tStart = t()
                        l = self.cube_time.split(':')
                        seconds = 1 * int(l[-1]) + 60 * int(l[-2])
                        if len(l) == 3:
                            seconds += 3600 * int(l[-3])
                        go = False
                        message = 'Waiting for Cube'
                        i = 0
                        while not go:
                            if t()-tStart > seconds + 3600:
                                print('Restarting Cubing')
                                print('\n')
                                self.make_cube(overwrite=False)
                                tStart = t()
                            else:
                                if i > 0:
                                    print(message, end='\r')
                                    sleep(120)
                                if os.path.exists(cube_check):
                                    go = True
                                    print('\n')
                                else:
                                    message += '.'
                                    i += 1
            
                for cut in self.cuts:
                    go = False
                    if self.part:
                        cut_check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/cut.txt'
                        cut_check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/cut.txt'
                        if (os.path.exists(cut_check1)) & (os.path.exists(cut_check1)):
                            print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
                            print('\n')
                        else:
                            go = True
                    else:
                        cut_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/cut.txt'
                        if os.path.exists(cut_check):
                            print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
                            print('\n')
                        else:
                            go = True
                    if go:
                        # -- Create python file for cubing, cutting, reducing a cut-- # 
                        print(f'Creating Cutting Python File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
                        python_text = f"\
from tessellate import DataProcessor\n\
\n\
part = {self.part}\n\
processor = DataProcessor(sector={self.sector},path='{self.data_path}',verbose=2)\n\
processor.make_cuts(cam={cam},ccd={ccd},n={self.n},cut={cut},part=part)\n\
if not part:\n\
    with open(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/cut.txt', 'w') as file:\n\
        file.write('Cut!')"

                        with open(f"{self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
                            python_file.write(python_text)

                        # -- Create bash file to submit job -- #
                        #print('Creating Cutting Batch File')
                        batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Cutting\n\
#SBATCH --output={self.job_output_path}/tessellate_cutting_logs/%A_cutting_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_cutting_logs/%A_cutting_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.cut_time}\n\
#SBATCH --cpus-per-task={self.cut_cpu}\n\
#SBATCH --mem-per-cpu={self.cut_mem}G\n\
\n\
python {self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py"

                        with open(f"{self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
                            batch_file.write(batch_text)

                        #print('Submitting Cutting Batch File')
                        os.system(f'sbatch {self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')
                        print('\n')

                if self.part:
                    for i in range(2):
                        done = self._get_catalogues(cam=cam,ccd=ccd,base_path=f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part{i+1}')
                        if not done:
                            break
                else:
                    done = self._get_catalogues(cam=cam,ccd=ccd,base_path=f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}')
                
                print('\n')
                if not done:
                    print('Restarting Cutting')
                    print('\n')
                    self.make_cuts(cubing=cubing,overwrite=False)


    def reduce(self,overwrite=True):
        """
        Reduce! 
        """

        _Save_space(f'{self.working_path}/reduction_scripts')

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/reduction_scripts/S{self.sector}C*')

        if (overwrite) & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('reduce' in self.overwrite):
                delete_files('reductions',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd: 
                print(_Print_buff(60,f'Reducing Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')
                for cut in self.cuts:
                    if self.part:
                        cut_check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
                        cut_check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
                        if not os.path.exists(cut_check1):
                            e = f'No Source Catalogue Detected for Reduction of Cut {cut} Part 1!\n'
                            raise ValueError(e)
                        elif not os.path.exists(cut_check2):
                            e = f'No Source Catalogue Detected for Reduction of Cut {cut} Part 2!\n'
                            raise ValueError(e)
                    else:
                        cut_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
                        if not os.path.exists(cut_check):
                            e = f'No Source Catalogue Detected for Reduction of Cut {cut}!\n'
                            raise ValueError(e)
                    
                    go = True
                    if self.part:
                        reduced_check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/reduced.txt'
                        reduced_check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/reduced.txt'
                        if (os.path.exists(reduced_check1))&(os.path.exists(reduced_check2)):
                            print(f'Cam {cam} CCD {ccd} Cut {cut} already reduced!')
                            print('\n')
                            go = False
                    else:
                        reduced_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/reduced.txt'
                        if os.path.exists(reduced_check):
                            print(f'Cam {cam} CCD {ccd} Cut {cut} already reduced!')
                            print('\n')
                            go = False
                    
                    if go:                    
                        # -- Create python file for reducing a cut-- # 
                        print(f'Creating Reduction Python File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
                        python_text = f"\
from tessellate import DataProcessor\n\
import os\n\
\n\
part={self.part}\n\
processor = DataProcessor(sector={self.sector},path='{self.data_path}',verbose=2)\n\
processor.reduce(cam={cam},ccd={ccd},n={self.n},cut={cut},part=part)\n\
if not part:\n\
    if os.path.exists('{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_Shifts.npy'):\n\
        with open(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/reduced.txt', 'w') as file:\n\
            file.write('Reduced!')"
            # file.write('Reduced with TESSreduce version {tr.__version__}.')"
                        with open(f"{self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
                            python_file.write(python_text)

                        # -- Create bash file to submit job -- #
                        #print('Creating Reduction Batch File')
                        batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Reduction\n\
#SBATCH --output={self.job_output_path}/tessellate_reduction_logs/%A_reduction_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_reduction_logs/%A_reduction_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.reduce_time}\n\
#SBATCH --cpus-per-task={self.reduce_cpu}\n\
#SBATCH --mem-per-cpu={self.reduce_mem}G\n\
\n\
python {self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py"

                        with open(f"{self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
                            batch_file.write(batch_text)
                                
                        #print('Submitting Reduction Batch File')
                        os.system(f'sbatch {self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')

                        print('\n')

    def _cut_transient_search(self,cam,ccd,cut):

        # -- Create python file for reducing a cut-- # 
        print(f'Creating Transient Search File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
        python_text = f"\
from tessellate import Detector\n\
import os\n\
\n\
part = {self.part}\n\
\n\
if part:\n\
    path1 = '{self.data_path}/{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/detected_events.csv'\n\
    path2 = '{self.data_path}/{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/detected_events.csv'\n\
    if not os.path.exists(path1):\n\
        detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},part=1)\n\
        detector.source_detect(cut={cut},mode='{self.detect_mode}',time_bin={self.time_bin})\n\
    if not os.path.exists(path2):\n\
        detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},part=2)\n\
        detector.source_detect(cut={cut},mode='{self.detect_mode}',time_bin={self.time_bin})\n\
else:\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n})\n\
    detector.source_detect(cut={cut},mode='{self.detect_mode}',time_bin={self.time_bin})"
                    
        with open(f"{self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
            python_file.write(python_text)

        # -- Create bash file to submit job -- #
        #print('Creating Transient Search Batch File')
        batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Search\n\
#SBATCH --output={self.job_output_path}/tessellate_search_logs/%A_search_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_search_logs/%A_search_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.search_time}\n\
#SBATCH --cpus-per-task={self.search_cpu}\n\
#SBATCH --mem-per-cpu={self.search_mem}G\n\
\n\
python {self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py"

        with open(f"{self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
            batch_file.write(batch_text)
                
        #print('Submitting Transient Search Batch File')
        os.system(f'sbatch {self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')

        print('\n')

    def transient_search(self,reducing,overwrite=True):
        """
        Transient Search!
        """

        _Save_space(f'{self.working_path}/detection_scripts')

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/detection_scripts/S{self.sector}C*')

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('search' in self.overwrite):
                delete_files('search',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd:
                print(_Print_buff(60,f'Transient Search for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')
                if not reducing:
                    for cut in self.cuts:
                        if self.part:
                            save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}'
                            save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}'
                            if (os.path.exists(f'{save_path1}/detected_events.csv')) & (os.path.exists(f'{save_path2}/detected_events.csv')):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
                                print('\n')
                            elif (os.path.exists(f'{save_path1}/reduced.txt')) & (os.path.exists(f'{save_path2}/reduced.txt')):
                                self._cut_transient_search(cam,ccd,cut)
                            elif not os.path.exists(f'{save_path1}/reduced.txt'):
                                e = f'No Reduced File Detected for Search of Cut {cut} Part 1!\n'
                                raise ValueError(e)
                            else:
                                e = f'No Reduced File Detected for Search of Cut {cut} Part 2!\n'
                                raise ValueError(e)
                        else:
                            save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
                            if os.path.exists(f'{save_path}/detected_events.csv'):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
                                print('\n')
                            elif os.path.exists(f'{save_path}/reduced.txt'):
                                self._cut_transient_search(cam,ccd,cut)
                            else:
                                e = f'No Reduced File Detected for Search of Cut {cut}!\n'
                                raise ValueError(e)
                            
                else:
                    completed = []
                    message = 'Waiting for Reductions'

                    tStart = t()
                    l = self.reduce_time.split(':')
                    seconds = 1 * int(l[-1]) + 60 * int(l[-2])
                    if len(l) == 3:
                        seconds += 3600 * int(l[-3])
                    else:
                        l.insert(0,0)

                    if (len(self.cam)>1) | (len(self.ccd)>1): 
                        seconds = 42300

                    i = 0
                    while len(completed) < len(self.cuts):
                        if t()-tStart > seconds + 600:
                            print('Restarting Reducing')
                            print('\n')
                            self.reduce_time = f'{int(l[0])+1}:{l[1]}:{l[2]}'
                            self.reduce(overwrite=False)
                            tStart = t()
                        else:
                            if i > 0:
                                print(message, end='\r')
                                sleep(120)
                            for cut in self.cuts:
                                if cut not in completed:
                                    if self.part:
                                        save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}'
                                        save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}'
                                        if (os.path.exists(f'{save_path1}/detected_events.csv')) & (os.path.exists(f'{save_path2}/detected_events.csv')):
                                            print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
                                            print('\n')
                                        elif (os.path.exists(f'{save_path1}/reduced.txt')) & (os.path.exists(f'{save_path2}/reduced.txt')):
                                            self._cut_transient_search(cam,ccd,cut)
                                            completed.append(cut)
                                    else:
                                        save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
                                        if os.path.exists(f'{save_path}/detected_events.csv'):
                                            completed.append(cut)
                                            print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
                                            print('\n')
                                        elif os.path.exists(f'{save_path}/reduced.txt'):
                                            self._cut_transient_search(cam,ccd,cut)
                                            completed.append(cut)
                            i+=1


    def _cut_transient_plot(self,cam,ccd,cut):

        # -- Create python file for reducing a cut-- # 
        print(f'Creating Transient Plotting File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
        python_text = f"\
from tessellate import Detector\n\
\n\
part = {self.part}\n\
if part:\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},part=1)\n\
    detector.plot_ALL(cut={cut},lower=2)\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},part=2)\n\
    detector.plot_ALL(cut={cut},lower=2)\n\
else:\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n})\n\
    detector.plot_ALL(cut={cut},lower=2)"
                    
        with open(f"{self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
            python_file.write(python_text)

        # -- Create bash file to submit job -- #
        #print('Creating Transient Search Batch File')
        batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Plotting\n\
#SBATCH --output={self.job_output_path}/tessellate_plotting_logs/%A_plotting_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_plotting_logs/%A_plotting_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.plot_time}\n\
#SBATCH --cpus-per-task={self.plot_cpu}\n\
#SBATCH --mem-per-cpu={self.plot_mem}G\n\
\n\
python {self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py"

        with open(f"{self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
            batch_file.write(batch_text)
                
        #print('Submitting Transient Search Batch File')
        os.system(f'sbatch {self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')

        print('\n')

    def transient_plot(self,searching=False,overwrite=True): # here
        """
        Transient Search!
        """

        _Save_space(f'{self.working_path}/plotting_scripts')

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/plotting_scripts/S{self.sector}C*')
        
        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('plot' in self.overwrite):
                delete_files('plot',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd:
                print(_Print_buff(60,f'Transient Plotting for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')
                if not searching:
                    for cut in self.cuts:
                        if self.part:
                            save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}'
                            save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}'
                            if (os.path.exists(f'{save_path1}/figs')) & (os.path.exists(f'{save_path2}/figs')):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} plots already made!')
                                print('\n')
                            elif (os.path.exists(f'{save_path1}/detected_events.csv'))&(os.path.exists(f'{save_path2}/detected_events.csv')):
                                self._cut_transient_plot(cam,ccd,cut)
                            elif not os.path.exists(f'{save_path1}/detected_events.csv'):
                                e = f'No Event File Detected for Plotting of Cut {cut} Part 1!\n'
                                raise ValueError(e)
                            else:
                                e = f'No Event File Detected for Plotting of Cut {cut} Part 2!\n'
                                raise ValueError(e)
                        else:
                            save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
                            if os.path.exists(f'{save_path}/figs'):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} plots already made!')
                                print('\n')
                            elif os.path.exists(f'{save_path}/detected_events.csv'):
                                self._cut_transient_plot(cam,ccd,cut)
                            else:
                                e = f'No Event File Detected for Plotting of Cut {cut}!\n'
                                raise ValueError(e)
                            
                        
                else:
                    completed = []
                    message = 'Waiting for Search'

                    tStart = t()
                    l = self.search_time.split(':')
                    seconds = 1 * int(l[-1]) + 60 * int(l[-2])
                    if len(l) == 3:
                        seconds += 3600 * int(l[-3])
                    else:
                        l.insert(0,0)

                    if (len(self.cam)>1) | (len(self.ccd)>1): 
                        seconds = 42300
                        
                    i = 0
                    while len(completed) < len(self.cuts):
                        if t()-tStart > seconds + 600:
                            print('Restarting Search')
                            print('\n')
                            self.search_time = f'{int(l[0])+1}:{l[1]}:{l[2]}'
                            self.transient_search(False,overwrite=False)
                            tStart = t()
                        else:
                            if i > 0:
                                print(message, end='\r')
                                sleep(120)
                            for cut in self.cuts:
                                if cut not in completed:
                                    if self.part:
                                        save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}'
                                        save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}'
                                        if (os.path.exists(f'{save_path1}/figs')) & (os.path.exists(f'{save_path2}/figs')):
                                            print(f'Cam {cam} CCD {ccd} Cut {cut} plots already made!')
                                            print('\n')
                                        elif (os.path.exists(f'{save_path1}/detected_events.csv'))&(os.path.exists(f'{save_path2}/detected_events.csv')):
                                            self._cut_transient_plot(cam,ccd,cut)
                                            completed.append(cut)
                                    else:
                                        save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
                                        if os.path.exists(f'{save_path}/figs'):
                                            completed.append(cut)
                                            print(f'Cam {cam} CCD {ccd} Cut {cut} already plotted!')
                                            print('\n')
                                        elif os.path.exists(f'{save_path}/detected_events.csv'):
                                            self._cut_transient_plot(cam,ccd,cut)
                                            completed.append(cut)
                            i+=1

