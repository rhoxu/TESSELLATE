from .dataprocessor import _Print_buff, DataProcessor
from .detector import *

class Tessellate():

    def __init__(self,sector,data_path,cam=None,ccd=None,n=None,
                 batch_time1=None,batch_time2=None,cpu1=None,cpu2=None,
                 mem1=None,mem2=None,verbose=2,download_number=None,cut=None,
                 job_output_path=None,working_path=None,
                 download=True,make_cube_cuts=True,reduce=True):
        
        self.sector = sector
        self.data_path = data_path

        self.cam = cam
        self.ccd = ccd
        self.verbose=verbose

        self.download_number = download_number
        self.n = n
        self.cut = cut
        
        self.batch_time1 = batch_time1
        self.cpu1 = cpu1
        self.mem1 = mem1

        self.batch_time2 = batch_time2
        self.cpu2 = cpu2
        self.mem2 = mem2

        if (job_output_path is None) | (working_path is None):
            m = 'Ensure you specify paths for job output and working path!'
            raise ValueError(m)
        else:
            self.job_output_path = job_output_path
            self.working_path = working_path

        message = self._run_properties()

        if download:
            self.download(message)
        if make_cube_cuts:
            self.make_cube_cuts()


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
            print(f'   - CCD = {self.cam}')
            self.ccd = [self.ccd]
            message += f'   - CCD = {self.ccd}\n'
        else:
            e = f'Invalid CCD Input of {self.ccd}\n'
            raise ValueError(e)
        
        print('\n')
        message += '\n'
        
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


        if self.batch_time1 is None:
            bt1 = input("   - Batch Time 1 (Cube/Cut) ['h:mm:ss'] = ")
            message += f"   - Batch Time 1 (Cube/Cut) ['h:mm:ss'] = {bt1}\n"
            done = False
            while not done:
                if ':' in bt1:
                    self.batch_time1 = bt1
                    done = True
                else:
                    bt1 = input("      Invalid format! Batch Time 1 (Cube/Cut) ['h:mm:ss'] = ")
                    message += f"      Invalid choice! Batch Time 1 (Cube/Cut) ['h:mm:ss'] = {bt1}\n"
        else:
            print(f'   - Batch Time 1 (Cube/Cut) = {self.batch_time1}')
            message += f"   - Batch Time 1 (Cube/Cut) = {self.batch_time1}')\n"

        if self.cpu1 is None:
            cpu1 = input("   - Num CPUs1 [1-32] = ")
            message += f"   - Num CPUs1 [1-32] = {cpu1}\n"
            done = False
            while not done:
                try:
                    cpu1 = int(cpu1)
                    if 0 < cpu1 < 33:
                        self.cpu1 = cpu1
                        done = True
                    else:
                        cpu1 = input("      Invalid format! Num CPUs1 [1-32] = ")
                        message += f"      Invalid choice! Num CPUs1 [1-32] = {cpu1}\n"
                except:
                    cpu1 = input("      Invalid format! Num CPUs1 [1-32] = ")
                    message += f"      Invalid choice! Num CPUs1 [1-32] = {cpu1}\n"
        elif 0 < self.cpu1 < 33:
            print(f'   - Num CPUs1 = {self.cpu1}')
            message += f"   - Num CPUs1 = {self.cpu1}\n"
        else:
            e = f"Invalid CPUs1 Input of {self.cpu1}\n"
            raise ValueError(e)
        
        if self.mem1 is None:
            mem1 = input("   - Mem1/CPU [10G suggested] = ")
            message += f"   - Mem1/CPU [10G suggested] = {mem1}\n"
            done = False
            while not done:
                try: 
                    mem1 = int(mem1)
                    if 0<mem1 < 500:
                        self.mem1 = mem1
                        done=True
                    else:
                        mem1 = input("      Invalid format! Mem1/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Mem1/CPU [10G suggested] = {mem1}\n"
                except:
                    if mem1[-1].lower() == 'g':
                        self.mem1 = mem1[:-1]
                        done = True
                    else:
                        mem1 = input("      Invalid format! Mem1/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Mem1/CPU [10G suggested] = {mem1}\n"

        elif 0 < self.mem1 < 500:
            print(f'   - Mem1/CPU = {self.mem1}G')
            message += f"   - Mem1/CPU = {self.mem1}G\n"
        else:
            e = f"Invalid Mem1/CPU Input of {self.mem1}\n"
            raise ValueError(e)

        print('\n')
        message += '\n'

        if self.batch_time2 is None:
            bt2 = input("   - Batch Time 2 (Reduce) ['h:mm:ss'] = ")
            message += f"   - Batch Time 2 (Reduce) ['h:mm:ss'] = {bt2}\n"
            done = False
            while not done:
                if ':' in bt2:
                    self.batch_time1 = bt2
                    done = True
                else:
                    bt2 = input("      Invalid format! Batch Time 2 (Reduce) ['h:mm:ss'] = ")
                    message += f"      Invalid choice! Batch Time 2 (Reduce) ['h:mm:ss'] = {bt2}\n"
        else:
            print(f'   - Batch Time 2 (Reduce) = {self.batch_time2}')
            message += f"   - Batch Time 2 (Reduce) = {self.batch_time2}')\n"

        if self.cpu2 is None:
            cpu2 = input("   - Num CPUs2 [1-32] = ")
            message += f"   - Num CPUs2 [1-32] = {cpu2}\n"
            done = False
            while not done:
                try:
                    cpu2 = int(cpu2)
                    if 0 < cpu2 < 33:
                        self.cpu2 = cpu2
                        done = True
                    else:
                        cpu2 = input("      Invalid format! Num CPUs2 [1-32] = ")
                        message += f"      Invalid choice! Num CPUs2 [1-32] = {cpu2}\n"
                except:
                    cpu2 = input("      Invalid format! Num CPUs2 [1-32] = ")
                    message += f"      Invalid choice! Num CPUs2 [1-32] = {cpu2}\n"
        elif 0 < self.cpu2 < 33:
            print(f'   - Num CPUs2 = {self.cpu2}')
            message += f"   - Num CPUs2 = {self.cpu2}\n"
        else:
            e = f"Invalid CPUs2 Input of {self.cpu2}\n"
            raise ValueError(e)
        
        if self.mem2 is None:
            mem2 = input("   - Mem2/CPU [10G suggested] = ")
            message += f"   - Mem2/CPU [10G suggested] = {mem2}\n"
            done = False
            while not done:
                try: 
                    mem2 = int(mem2)
                    if 0<mem2 < 500:
                        self.mem2 = mem2
                        done=True
                    else:
                        mem2 = input("      Invalid format! Mem2/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Mem2/CPU [10G suggested] = {mem2}\n"
                except:
                    if mem2[-1].lower() == 'g':
                        self.mem2 = mem2[:-1]
                        done = True
                    else:
                        mem2 = input("      Invalid format! Mem2/CPU [10G suggested] = ")
                        message += f"      Invalid choice! Mem2/CPU [10G suggested] = {mem2}\n"

        elif 0 < self.mem2 < 500:
            print(f'   - Mem2/CPU = {self.mem2}G')
            message += f"   - Mem2/CPU = {self.mem2}G\n"
        else:
            e = f"Invalid Mem2/CPU Input of {self.mem2}\n"
            raise ValueError(e)
        
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

    def make_cube_cuts(self):

        for cam in self.cam:
            for ccd in self.ccd: 
                # -- Create python file for cubing, cutting, reducing a cut-- # 
                print(f'Creating Cubing/Cutting Python File for Cam{cam}Ccd{ccd}')
                python_text = f"\
from tessellate import DataProcessor\n\
\n\
sector = {self.sector}\n\
cam = {cam}\n\
ccd = {ccd}\n\
data_path = f'{self.data_path}'\n\
download_number = {self.download_number}\n\
cut = {self.cut}\n\
n = {self.n}\n\
\n\
processor = DataProcessor(sector=sector,path=data_path,verbose=2)\n\
processor.make_cube(cam=cam,ccd=ccd)\n\
processor.make_cuts(cam=cam,ccd=ccd,n=n,cut=cut)"

        python_file = open(f"{self.working_path}/task_script1.py", "w")
        python_file.write(python_text)
        python_file.close()

        # -- Create bash file to submit job -- #
        print('Creating Cubing/Cutting Batch File')
        batch_text = f"\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=tessreduce_attempt\n\
#SBATCH --output={self.job_output_path}/job_output_%A.txt\n\
#SBATCH --error={self.job_output_path}/errors_%A.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.batch_time1}\n\
#SBATCH --cpus-per-task={self.cpu1}\n\
#SBATCH --mem-per-cpu={self.mem1}G\n\
\n\
python {self.working_path}/task_script1.py"

        batch_file = open(f'{self.working_path}/task_script1.sh', "w")
        batch_file.write(batch_text)
        batch_file.close()

        print('Submitting Cubing/Cutting Batch File')
        os.system(f'sbatch {self.working_path}/task_script1.sh')
        print('\n')



