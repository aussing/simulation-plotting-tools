import numpy as np
import h5py
import fileinput
import os
import shutil
from tqdm import trange


LITTLEH = 0.6688
UNITMASS = 1e10

def read_write_param_file(folder, dm_list, sn_fac_list, method_list):
    
    sim_short_name = folder.split('sd')[-1][0:2]
    for method in method_list:
        for dm in dm_list:
            if dm == 'wdm_3.5':
                dm = 'wdm'
            for sn_fac in sn_fac_list:
                if not os.path.exists(f'./{folder}/{method}_jobs/'):
                    os.makedirs(f'./{folder}/{method}_jobs/')       
                shutil.copy(f'./example_job_scripts/{method}_jobs/job_{dm}_{sn_fac}.sh', f'./{folder}/{method}_jobs/')
                file = f'./{folder}/{method}_jobs/job_{dm}_{sn_fac}.sh'
                print(f'original example =  ./example_job_scripts/{method}_jobs/job_{dm}_{sn_fac}.sh')
                print(f'Replacing job name in {file}\n')
               
                with fileinput.input(files=file, inplace=True) as f: #,backup='.OG'
                    for lines in f:
                        snap_line = lines.replace('#SBATCH --job-name',f'#SBATCH --job-name sd{sim_short_name}_{method}_{dm}_{sn_fac}')
                        print(snap_line,end='')
        shutil.copy(f'./example_job_scripts/{method}_jobs/test_run', f'./{folder}/{method}_jobs/run_all')
    shutil.copy(f'./example_job_scripts/mega_run_all', f'./{folder}/mega_xrun_all')
                
if __name__ =='__main__':

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if (simulation_name.startswith(("N2048_L65_sd12","N2048_L65_sd14")))]:# and not simulation_name.startswith(("N2048_L65_sd00")))]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(folder_list)

    for folder in folder_list:
        dm_list = ['cdm','wdm_3.5']
        sn_fac_list = ['sn_005','sn_010']
        method_list = ['dtm','rr','li_bf']

        read_write_param_file(folder, dm_list, sn_fac_list, method_list)