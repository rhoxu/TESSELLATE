from .dataprocessor import _Print_buff, DataProcessor
from .detector import *

from glob import glob
from time import sleep

class Tessellate():

    def __init__(self,sector,data_path,cam=None,ccd=None,n=None,
                 verbose=2,download_number=None,cut=None,
                 job_output_path=None,working_path=None,
                 cube_time=None,cut_time=None,reduce_time=None,
                 cube_cpu=None,cut_cpu=None,reduce_cpu=None,
                 cube_mem=None,cut_mem=None,reduce_mem=None,
                 download=False,make_cube=False,make_cuts=False,reduce=False):
        
        if (job_output_path is None) | (working_path is None):
            m = 'Ensure you specify paths for job output and working path!'
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
        self.cut = cut

        self.cube_time = cube_time
        self.cube_cpu = cube_cpu
        self.cube_mem = cube_mem

        self.cut_time = cut_time
        self.cut_cpu = cut_cpu
        self.cut_mem = cut_mem

        self.reduce_time = reduce_time
        self.reduce_cpu = reduce_cpu
        self.reduce_mem = reduce_mem

        message = self._run_properties()

        if download:
            message = self._download_properties(message)

        if make_cube:
            message = self._cube_properties(message)

        if make_cuts:
            message = self._cut_properties(message)

        if reduce:
            message = self._reduce_properties(message,make_cuts)

        message = self._reset_logs(message)

        if download:
            self.download(message)

        if make_cube:
            self.make_cube()
        
        if make_cuts:
            self.make_cuts()

        if reduce:
            self.reduce()        


    def _run_properties(self):

        message = ''

        print('\n')
        print(_Print_buff(50,f'Initialising Sector {self.sector} Tessellation'))
        message += _Print_buff(50,f'Initialising Sector {self.sector} Tessellation')+'\n'

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
    
    def _download_properties(self,message):

        if self.download_number is None:
            dNum = input('   - Download Number = ')
            message += f'   - Download Number = {dNum}\n'
            done = False
            while not done:
                try:
                    dNum = int(dNum)
                    if 0 < dNum < 14000:
                        self.download_number = dNum
                        done = True
                    else:
                        dNum = input('      Invalid choice! Download Number =  ')
                        message += f'      Invalid choice! Download Number = {dNum}\n'
                except:
                    if dNum == 'all':
                        self.download_number = dNum
                        done = True
                    else:
                        dNum = input('      Invalid choice! Download Number =  ')
                        message += f'      Invalid choice! Download Number = {dNum}\n'
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
        message += '/n'

        return message
    
    def _cube_properties(self,message):

        if self.cube_time is None:
            cube_time = input("   - Cube Batch Time ['h:mm:ss'] = ")
            message += f"   - Cube Batch Time ['h:mm:ss'] = {cube_time}\n"
            done = False
            while not done:
                if ':' in cube_time:
                    self.cube_time = cube_time
                    done = True
                else:
                    cube_time = input("      Invalid format! Cube Batch Time ['h:mm:ss'] = ")
                    message += f"      Invalid choice! Cube Batch Time ['h:mm:ss'] = {cube_time}\n"
        else:
            print(f'   - Cube Batch Time = {self.cube_time}')
            message += f"   - Cube Batch Time = {self.cube_time}')\n"

        if self.cube_cpu is None:
            cube_cpu = input("   - Cube Num CPUs [1-32] = ")
            message += f"   - Cube Num CPUs [1-32] = {cube_cpu}\n"
            done = False
            while not done:
                try:
                    cube_cpu = int(cube_cpu)
                    if 0 < cube_cpu < 33:
                        self.cube_cpu = cube_cpu
                        done = True
                    else:
                        cube_cpu = input("      Invalid format! Cube Num CPUs [1-32] = ")
                        message += f"      Invalid choice! Cube Num CPUs [1-32] = {cube_cpu}\n"
                except:
                    cube_cpu = input("      Invalid format! Cube Num CPUs [1-32] = ")
                    message += f"      Invalid choice! Cube Num CPUs [1-32] = {cube_cpu}\n"
        elif 0 < self.cube_cpu < 33:
            print(f'   - Cube Num CPUs = {self.cube_cpu}')
            message += f"   - Cube Num CPUs = {self.cube_cpu}\n"
        else:
            e = f"Invalid Cube CPUs Input of {self.cube_cpu}\n"
            raise ValueError(e)
        
        if self.cube_mem is None:
            cube_mem = input("   - Cube Mem/CPU [10G suggested] = ")
            message += f"   - Cube Mem/CPU [10G suggested] = {cube_mem}\n"
            done = False
            while not done:
                try: 
                    cube_mem = int(cube_mem)
                    if 0<cube_mem < 500:
                        self.cube_mem = cube_mem
                        done=True
                    else:
                        cube_mem = input("      Invalid format! Cube Mem/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Cube Mem/CPU [10G suggested] = {cube_mem}\n"
                except:
                    if cube_mem[-1].lower() == 'g':
                        self.cube_mem = cube_mem[:-1]
                        done = True
                    else:
                        cube_mem = input("      Invalid format! Cube Mem/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Cube Mem/CPU [10G suggested] = {cube_mem}\n"

        elif 0 < self.cube_mem < 500:
            print(f'   - Cube Mem/CPU = {self.cube_mem}G')
            message += f"   - Cube Mem/CPU = {self.cube_mem}G\n"
        else:
            e = f"Invalid Cube Mem/CPU Input of {self.cube_mem}\n"
            raise ValueError(e)

        print('\n')
        message += '\n'

        return message

    def _cut_properties(self,message):

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
        
        if self.cut is None:
            cut = input(f'   - Cut [1-{self.n**2},all] = ')
            message += f'   - Cut [1-{self.n**2},all] = {cut}\n'
            done = False
            while not done:
                if cut == 'all':
                    self.cut = 'all'
                    done = True
                elif cut in np.array(range(1,self.n**2+1)).astype(str):
                    self.cut = int(cut)
                    done = True
                else:
                    cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
                    message += f'      Invalid choice! Cut [1-{self.n**2},all] = {cut}\n'
        elif self.cut == 'all':
            print(f'   - Cut = all')
            message += f'   - Cut = all\n'
            self.cut = 'all'
        elif self.cut in range(1,self.n**2+1):
            print(f'   - Cut = {self.cut}')
            self.cut = self.cut
            message += f'   - Cut = {self.cut}\n'
        else:
            e = f"Invalid Cut Input of {self.cut} with 'n' of {self.n}\n"
            raise ValueError(e)
        
        print('\n')
        message += '\n'

        if self.cut_time is None:
            cut_time = input("   - Cut Batch Time ['h:mm:ss'] = ")
            message += f"   - Cut Batch Time ['h:mm:ss'] = {cut_time}\n"
            done = False
            while not done:
                if ':' in cut_time:
                    self.cut_time = cut_time
                    done = True
                else:
                    cut_time = input("      Invalid format! Cut Batch Time ['h:mm:ss'] = ")
                    message += f"      Invalid choice! Cut Batch Time ['h:mm:ss'] = {cut_time}\n"
        else:
            print(f'   - Cut Batch Time = {self.cut_time}')
            message += f"   - Cut Batch Time = {self.cut_time}')\n"

        if self.cut_cpu is None:
            cut_cpu = input("   - Cut Num CPUs [1-32] = ")
            message += f"   - Cut Num CPUs [1-32] = {cut_cpu}\n"
            done = False
            while not done:
                try:
                    cut_cpu = int(cut_cpu)
                    if 0 < cut_cpu < 33:
                        self.cut_cpu = cut_cpu
                        done = True
                    else:
                        cut_cpu = input("      Invalid format! Cut Num CPUs [1-32] = ")
                        message += f"      Invalid choice! Cut Num CPUs [1-32] = {cut_cpu}\n"
                except:
                    cut_cpu = input("      Invalid format! Cut Num CPUs [1-32] = ")
                    message += f"      Invalid choice! Cut Num CPUs [1-32] = {cut_cpu}\n"
        elif 0 < self.cut_cpu < 33:
            print(f'   - Cut Num CPUs = {self.cut_cpu}')
            message += f"   - Cut Num CPUs = {self.cut_cpu}\n"
        else:
            e = f"Invalid Cut CPUs Input of {self.cut_cpu}\n"
            raise ValueError(e)
        
        if self.cut_mem is None:
            cut_mem = input("   - Cut Mem/CPU [10G suggested] = ")
            message += f"   - Cut Mem/CPU [10G suggested] = {cut_mem}\n"
            done = False
            while not done:
                try: 
                    cut_mem = int(cut_mem)
                    if 0<cut_mem < 500:
                        self.cut_mem = cut_mem
                        done=True
                    else:
                        cut_mem = input("      Invalid format! Cut Mem/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Cut Mem/CPU [10G suggested] = {cut_mem}\n"
                except:
                    if cut_mem[-1].lower() == 'g':
                        self.cut_mem = cut_mem[:-1]
                        done = True
                    else:
                        cut_mem = input("      Invalid format! Cut Mem/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Cut Mem/CPU [10G suggested] = {cut_mem}\n"

        elif 0 < self.cut_mem < 500:
            print(f'   - Cut Mem/CPU = {self.cut_mem}G')
            message += f"   - Cut Mem/CPU = {self.cut_mem}G\n"
        else:
            e = f"Invalid Cut Mem/CPU Input of {self.cut_mem}\n"
            raise ValueError(e)

        print('\n')
        message += '\n'

        return message

    def _reduce_properties(self,message,cutting):

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
            
            if self.cut is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                message += f'   - Cut [1-{self.n**2},all] = {cut}\n'
                done = False
                while not done:
                    if cut == 'all':
                        self.cut = 'all'
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cut = int(cut)
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
                        message += f'      Invalid choice! Cut [1-{self.n**2},all] = {cut}\n'
            elif self.cut == 'all':
                print(f'   - Cut = all')
                message += f'   - Cut = all\n'
                self.cut = 'all'
            elif self.cut in range(1,self.n**2+1):
                print(f'   - Cut = {self.cut}')
                self.cut = self.cut
                message += f'   - Cut = {self.cut}\n'
            else:
                e = f"Invalid Cut Input of {self.cut} with 'n' of {self.n}\n"
                raise ValueError(e)
            
            print('\n')
            message += '\n'

        if self.reduce_time is None:
            reduce_time = input("   - Reduce Batch Time ['h:mm:ss'] = ")
            message += f"   - Reduce Batch Time ['h:mm:ss'] = {reduce_time}\n"
            done = False
            while not done:
                if ':' in reduce_time:
                    self.reduce_time = reduce_time
                    done = True
                else:
                    reduce_time = input("      Invalid format! Reduce Batch Time ['h:mm:ss'] = ")
                    message += f"      Invalid choice! Reduce Batch Time ['h:mm:ss'] = {reduce_time}\n"
        else:
            print(f'   - Reduce Batch Time = {self.reduce_time}')
            message += f"   - Reduce Batch Time = {self.reduce_time}')\n"

        if self.reduce_cpu is None:
            reduce_cpu = input("   - Reduce Num CPUs [1-32] = ")
            message += f"   - Reduce Num CPUs [1-32] = {reduce_cpu}\n"
            done = False
            while not done:
                try:
                    reduce_cpu = int(reduce_cpu)
                    if 0 < reduce_cpu < 33:
                        self.reduce_cpu = reduce_cpu
                        done = True
                    else:
                        reduce_cpu = input("      Invalid format! Reduce Num CPUs [1-32] = ")
                        message += f"      Invalid choice! Reduce Num CPUs [1-32] = {reduce_cpu}\n"
                except:
                    reduce_cpu = input("      Invalid format! Reduce Num CPUs [1-32] = ")
                    message += f"      Invalid choice! Reduce Num CPUs [1-32] = {reduce_cpu}\n"
        elif 0 < self.reduce_cpu < 33:
            print(f'   - Reduce Num CPUs = {self.reduce_cpu}')
            message += f"   - Reduce Num CPUs = {self.reduce_cpu}\n"
        else:
            e = f"Invalid Reduce CPUs Input of {self.reduce_cpu}\n"
            raise ValueError(e)
        
        if self.reduce_mem is None:
            reduce_mem = input("   - Reduce Mem/CPU [10G suggested] = ")
            message += f"   - Reduce Mem/CPU [10G suggested] = {reduce_mem}\n"
            done = False
            while not done:
                try: 
                    reduce_mem = int(reduce_mem)
                    if 0<reduce_mem < 500:
                        self.reduce_mem = reduce_mem
                        done=True
                    else:
                        reduce_mem = input("      Invalid format! Reduce Mem/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Reduce Mem/CPU [10G suggested] = {reduce_mem}\n"
                except:
                    if reduce_mem[-1].lower() == 'g':
                        self.reduce_mem = reduce_mem[:-1]
                        done = True
                    else:
                        reduce_mem = input("      Invalid format! Reduce Mem/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Reduce Mem/CPU [10G suggested] = {reduce_mem}\n"

        elif 0 < self.reduce_mem < 500:
            print(f'   - Reduce Mem/CPU = {self.reduce_mem}G')
            message += f"   - Reduce Mem/CPU = {self.reduce_mem}G\n"
        else:
            e = f"Invalid Reduce Mem/CPU Input of {self.reduce_mem}\n"
            raise ValueError(e)

        print('\n')
        message += '\n'

        return message
    
    def _reset_logs(self,message):

        message += 'Delete Past Job Logs? [y/n] :\n'
        if input('Delete Past Job Logs? [y/n] :\n').lower() == 'y':
            os.system(f'rm {self.job_output_path}/*')
            message = message + 'y \n'
        else:
            message + 'n \n'
        print('\n')
        message += '\n'

        return message
        
    def download(self,message):

        for cam in self.cam:
            for ccd in self.ccd:
                tDownload = t()
                data_processor = DataProcessor(sector=self.sector,path=self.data_path,verbose=self.verbose)
                data_processor.download(cam=cam,ccd=ccd,number=self.download_number)
                os.system('clear')
                print(message)
                print(f'Download Complete ({((t()-tDownload)/60):.2f} mins).')
                print('\n')

    def make_cube(self):

        for cam in self.cam:
            for ccd in self.ccd: 

                # -- Delete old scripts -- #
                if os.path.exists(f'{self.working_path}/cubing_script.sh'):
                    os.system(f'rm {self.working_path}/cubing_script.sh')
                    os.system(f'rm {self.working_path}/cubing_script.py')

                # -- Create python file for cubing, cutting, reducing a cut-- # 
                print(f'Creating Cubing Python File for Cam{cam}Ccd{ccd}')
                python_text = f"\
from tessellate import DataProcessor\n\
\n\
sector = {self.sector}\n\
cam = {cam}\n\
ccd = {ccd}\n\
data_path = f'{self.data_path}'\n\
\n\
processor = DataProcessor(sector=sector,path=data_path,verbose=2)\n\
processor.make_cube(cam=cam,ccd=ccd)"
                
                python_file = open(f"{self.working_path}/cubing_script.py", "w")
                python_file.write(python_text)
                python_file.close()

                # -- Create bash file to submit job -- #
                print('Creating Cubing/Cutting Batch File')
                batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_Sector{self.sector}_Cam{cam}_Ccd{ccd}_Cubing\n\
#SBATCH --output={self.job_output_path}/cubing_job_output_%A.txt\n\
#SBATCH --error={self.job_output_path}/cubing_errors_%A.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.cube_time}\n\
#SBATCH --cpus-per-task={self.cube_cpu}\n\
#SBATCH --mem-per-cpu={self.cube_mem}G\n\
\n\
python {self.working_path}/cubing_script.py"

                batch_file = open(f'{self.working_path}/cubing_script.sh', "w")
                batch_file.write(batch_text)
                batch_file.close()

                print('Submitting Cubing Batch File')
                os.system(f'sbatch {self.working_path}/cubing_script.sh')
                print('\n')

    def _get_catalogues(self,cam,ccd):
                
        data_processor = DataProcessor(sector=self.sector,path=self.data_path,verbose=self.verbose)
        cutCorners,cutCentrePx,cutCentreCoords,rad = data_processor.find_cuts(cam=cam,ccd=ccd,n=self.n,plot=False)

        if self.cut == 'all':
                cuts = range(1,self.n**2+1)
        else:
            cuts = [self.cut]

        # -- Generate Star Catalogue -- #
        for i in cuts:
            message = f'Waiting for Cut {i}'
            found = False
            save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{i}of{self.n**2}'
            cutName = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{i}_of{self.n**2}.fits'

            image_path = glob(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/*ffic.fits')[0]

            while not found:
                if os.path.exists(f'{save_path}/{cutName}'):
                    try:
                        if not os.path.exists(f'{save_path}/local_gaia_cat'):
                            print(f'Generating Catalogue {i}')

                            tr.external_save_cat(radec=cutCentreCoords[i-1],size=2*rad,cutCornerPx=cutCorners[i-1],
                                                    image_path=image_path,save_path=save_path,maglim=19)
                            
                        found = True
                    except:
                        print(message, end='\r')
                        sleep(120)
                        message += '.'
                else:
                    print(message, end='\r')
                    sleep(120)
                    message += '.'

            print('\n')

    def make_cuts(self):

        for cam in self.cam:
            for ccd in self.ccd: 

                # -- Delete old scripts -- #
                if os.path.exists(f'{self.working_path}/cutting_script.sh'):
                    os.system(f'rm {self.working_path}/cutting_script.sh')
                    os.system(f'rm {self.working_path}/cutting_script.py')

                # -- Create python file for cubing, cutting, reducing a cut-- # 
                print(f'Creating Cutting Python File for Cam{cam}Ccd{ccd}')
                python_text = f"\
from tessellate import DataProcessor\n\
\n\
sector = {self.sector}\n\
cam = {cam}\n\
ccd = {ccd}\n\
data_path = f'{self.data_path}'\n\
cut = {self.cut}\n\
n = {self.n}\n\
\n\
processor = DataProcessor(sector=sector,path=data_path,verbose=2)\n\
processor.make_cuts(cam=cam,ccd=ccd,n=n,cut=cut)"

                python_file = open(f"{self.working_path}/cutting_script.py", "w")
                python_file.write(python_text)
                python_file.close()

                # -- Create bash file to submit job -- #
                print('Creating Cutting Batch File')
                batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_Sector{self.sector}_Cam{cam}_Ccd{ccd}_Cutting\n\
#SBATCH --output={self.job_output_path}/cutting_job_output_%A.txt\n\
#SBATCH --error={self.job_output_path}/cutting_errors_%A.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.cut_time}\n\
#SBATCH --cpus-per-task={self.cut_cpu}\n\
#SBATCH --mem-per-cpu={self.cut_mem}G\n\
\n\
python {self.working_path}/cutting_script.py"

                batch_file = open(f'{self.working_path}/cutting_script.sh', "w")
                batch_file.write(batch_text)
                batch_file.close()

                print('Submitting Cutting Batch File')
                os.system(f'sbatch {self.working_path}/cutting_script.sh')
                print('\n')

                self._get_catalogues(cam=cam,ccd=ccd)

    def reduce(self):

        for cam in self.cam:
            for ccd in self.ccd: 

                # -- Delete old scripts -- #
                if os.path.exists(f'{self.working_path}/reduction_script.sh'):
                    os.system(f'rm {self.working_path}/reduction_script.sh')
                    os.system(f'rm {self.working_path}/reduction_script.py')

                # -- Create python file for reducing a cut-- # 
                print('\n')
                print('Creating Reduction Python File')
                python_text = f"\
from DataGrab import DataGrab\n\
from time import time as t\n\
\n\
sector = {self.sector}\n\
cam = {cam}\n\
ccd = {ccd}\n\
data_path = f'{self.data_path}'\n\
cut = {self.cut}\n\
n = {self.n}\n\
\n\
processor = DataProcessor(sector=sector,path=data_path,verbose=2)\n\
processor.reduce(cam=cam,ccd=ccd,n=n,cut=cut)"
                
                python_file = open(f"{self.working_path}/reduction_script.py", "w")
                python_file.write(python_text)
                python_file.close()

                # -- Create bash file to submit job -- #
                print('Creating Reduction Batch File')
                batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_Sector{self.sector}_Cam{cam}_Ccd{ccd}_Reduction\n\
#SBATCH --output={self.job_output_path}/reduction_job_output_%A.txt\n\
#SBATCH --error={self.job_output_path}/reduction_errors_%A.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.reduce_time}\n\
#SBATCH --cpus-per-task={self.reduce_cpu}\n\
#SBATCH --mem-per-cpu={self.reduce_mem}G\n\
\n\
python {self.working_path}/reduction_script.py"

                batch_file = open(f'{self.working_path}/reduction_script.sh', "w")
                batch_file.write(batch_text)
                batch_file.close()
                        
                print('Submitting Reduction Batch File')
                os.system(f'sbatch {self.working_path}/reduction_script.sh')




