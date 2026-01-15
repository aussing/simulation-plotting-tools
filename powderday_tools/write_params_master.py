import numpy as np
import h5py
import fileinput
import os
import shutil
from tqdm import trange


LITTLEH = 0.6688
UNITMASS = 1e10

def read_write_param_file(folder, dm, sn_fac, method):
    if not os.path.exists(f'./{folder}/{method}_params/'):
                os.makedirs(f'./{folder}/{method}_params/')
                
    if dm =='cdm':
        if sn_fac =='sn_005':
            shutil.copy(f'./parameters_master_blank.py', f'./{folder}/{method}_params/cdm_05_params/parameters_master.py')
            file = f'./{folder}/{method}_params/cdm_05_params/parameters_master.py'
            # file = f'./{folder}/cdm_05_params/parameters_model_s{snap_num}.py'
        elif sn_fac =='sn_010':
            shutil.copy(f'./parameters_master_blank.py', f'./{folder}/{method}_params/cdm_10_params/parameters_master.py')
            file = f'./{folder}/{method}_params/cdm_10_params/parameters_master.py'
            # file = f'./{folder}/cdm_10_params/parameters_model_s{snap_num}.py'
        else:
            print('wrong cdm input')
    elif dm == 'wdm_3.5':
        if sn_fac =='sn_005':
            shutil.copy(f'./parameters_master_blank.py', f'./{folder}/{method}_params/wdm_05_params/parameters_master.py')
            file = f'./{folder}/{method}_params/wdm_05_params/parameters_master.py'
            # file = f'./{folder}/wdm_05_params/parameters_model_s{snap_num}.py'
        elif sn_fac =='sn_010':
            shutil.copy(f'./parameters_master_blank.py', f'./{folder}/{method}_params/wdm_10_params/parameters_master.py')
            file = f'./{folder}/{method}_params/wdm_10_params/parameters_master.py'
            # file = f'./{folder}/wdm_10_params/parameters_model_s{snap_num}.py'
        else:
            print('wrong wdm input')
    else:
        print('wrong inputs')
    
    print(f"replacing lines in {file}")

    with fileinput.input(files=file, inplace=True) as f: 
        for lines in f:
            if method != 'li_bf':
                snap_line = lines.replace('dust_grid_type = ',f"dust_grid_type = '{method}'")
            else:
                snap_line = lines.replace('dust_grid_type = ',f"dust_grid_type = 'li_bestfit'")
            print(snap_line,end='')

if __name__ =='__main__':

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if (simulation_name.startswith(("N2048_L65_sd12","N2048_L65_sd14")))]: # and not simulation_name.startswith("N2048_L65_sd00")
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(folder_list)
    # exit()
    for folder in folder_list:
        dm_list = ['cdm','wdm_3.5']
        sn_fac_list = ['sn_005','sn_010']
        method_list = ['dtm','rr','li_bf']

        for dm in dm_list:
            for sn_fac in sn_fac_list:
                print()
                print(f"{dm} - {sn_fac}\n")
                for method in method_list:
                    read_write_param_file(folder, dm, sn_fac, method)
                    