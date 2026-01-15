import numpy as np
import h5py
import fileinput
import os
import shutil
from tqdm import trange


LITTLEH = 0.6688
UNITMASS = 1e10

def find_halo(folder, dm, sn_fac, snap_num):
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')

    halo_pos   = np.array(halo_data['Group']['GroupPos'], dtype=np.float64) 
    halo_M200c = np.array(halo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS
    halo_masstypes = np.array(halo_data['Group']['GroupMassType'], dtype=np.float64)

    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0] #Select largest non-contaminated halo / resimulation target

    halo_pos = [halo_pos[mass_mask][halo_mainID,0],halo_pos[mass_mask][halo_mainID,1],halo_pos[mass_mask][halo_mainID,2]]

    return halo_pos

def read_write_param_file(folder, halo_pos, snap_num, dm, sn_fac, method):
    if not os.path.exists(f'./{folder}/{method}_params/'):
                os.makedirs(f'./{folder}/{method}_params/')
                
    if dm =='cdm':
        if sn_fac =='sn_005':
            if not os.path.exists(f'./{folder}/{method}_params/cdm_05_params'):
                os.makedirs(f'./{folder}/{method}_params/cdm_05_params')
            shutil.copy(f'./parameters_model_blank.py', f'./{folder}/{method}_params/cdm_05_params/parameters_model_cdm_sn_005_s{snap_num}.py')
            file = f'./{folder}/{method}_params/cdm_05_params/parameters_model_cdm_sn_005_s{snap_num}.py'
            # file = f'./{folder}/cdm_05_params/parameters_model_s{snap_num}.py'
        elif sn_fac =='sn_010':
            if not os.path.exists(f'./{folder}/{method}_params/cdm_10_params'):
                os.makedirs(f'./{folder}/{method}_params/cdm_10_params')
            shutil.copy(f'./parameters_model_blank.py', f'./{folder}/{method}_params/cdm_10_params/parameters_model_cdm_sn_010_s{snap_num}.py')
            file = f'./{folder}/{method}_params/cdm_10_params/parameters_model_cdm_sn_010_s{snap_num}.py'
            # file = f'./{folder}/cdm_10_params/parameters_model_s{snap_num}.py'
        else:
            print('wrong cdm input')
    elif dm == 'wdm_3.5':
        if sn_fac =='sn_005':
            if not os.path.exists(f'./{folder}/{method}_params/wdm_05_params'):
                os.makedirs(f'./{folder}/{method}_params/wdm_05_params')
            shutil.copy(f'./parameters_model_blank.py', f'./{folder}/{method}_params/wdm_05_params/parameters_model_wdm_sn_005_s{snap_num}.py')
            file = f'./{folder}/{method}_params/wdm_05_params/parameters_model_wdm_sn_005_s{snap_num}.py'
            # file = f'./{folder}/wdm_05_params/parameters_model_s{snap_num}.py'
        elif sn_fac =='sn_010':
            if not os.path.exists(f'./{folder}/{method}_params/wdm_10_params'):
                os.makedirs(f'./{folder}/{method}_params/wdm_10_params')
            shutil.copy(f'./parameters_model_blank.py', f'./{folder}/{method}_params/wdm_10_params/parameters_model_wdm_sn_010_s{snap_num}.py')
            file = f'./{folder}/{method}_params/wdm_10_params/parameters_model_wdm_sn_010_s{snap_num}.py'
            # file = f'./{folder}/wdm_10_params/parameters_model_s{snap_num}.py'
        else:
            print('wrong wdm input')
    else:
        print('wrong inputs')
    x_cent = np.round(halo_pos[0]*1000,2)
    y_cent = np.round(halo_pos[1]*1000,2)
    z_cent = np.round(halo_pos[2]*1000,2)
    # x_cent = np.round(halo_pos[0],2)
    # y_cent = np.round(halo_pos[1],2)
    # z_cent = np.round(halo_pos[2],2)
    
    print(f"replacing lines in {file}")

    with fileinput.input(files=file, inplace=True) as f: #,backup='.OG'
        for lines in f:
            snap_line = lines.replace('snapshot_num = ',f'snapshot_num = {snap_num}')
            print(snap_line,end='')

    with fileinput.input(files=file, inplace=True) as f:
        for lines_hydro_dir in f:
            line_hydro_dir = lines_hydro_dir.replace('hydro_dir = ',f'hydro_dir = "/fred/oz217/aussing/clean_pd/powderday/{folder}/{dm}/{sn_fac}/"')
            # line_hydro_dir = lines_hydro_dir.replace('hydro_dir = ',f'hydro_dir = "/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn_fac}/"')
            print(line_hydro_dir,end='')

    with fileinput.input(files=file, inplace=True) as f: #,backup='.bak'
        for lines_out_loc in f:
            line_out_loc = lines_out_loc.replace(f'PD_output_dir = ', 
                                                 f"PD_output_dir = '/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/{sn_fac}/snap_'+snapnum_str+'/' " )
            # line_out_loc = lines_out_loc.replace( f'PD_output_dir = f"/fred/oz217/aussing/clean_pd/powderday/{folder}/li_bf_out/{dm}/{sn_fac}/snap_{str(snap_num).zfill(3)}/" ',
            #                                      f"PD_output_dir = '/fred/oz217/aussing/clean_pd/powderday/{folder}/dtm/{dm}/{sn_fac}/snap_'+snapnum_str+'/'",)
            print(line_out_loc,end='')

    with fileinput.input(files=file, inplace=True) as f:
        for lines_x in f:
            line_x = lines_x.replace('x_cent = ',f'x_cent = {x_cent} # ')
            print(line_x,end='')

    with fileinput.input(files=file, inplace=True) as f:
        for lines_y in f:
            line_y = lines_y.replace('y_cent = ',f'y_cent = {y_cent} # ')
            print(line_y,end='')

    with fileinput.input(files=file, inplace=True) as f:
        for lines_z in f:
            line_z = lines_z.replace('z_cent = ',f'z_cent = {z_cent} # ')
            print(line_z,end='')

def write_out_dirs(folder,method,dm,sn_fac,snap_num):
    # pd_out_folder = f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/{sn_fac}/snap_{str(snap_num).zfill(3)}/'
    if not os.path.exists(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/'):
        os.makedirs(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/')

    if not os.path.exists(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/'):
        os.makedirs(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/')

    if not os.path.exists(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/{sn_fac}'):
        os.makedirs(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/{sn_fac}')

    if not os.path.exists(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/{sn_fac}/snap_{str(snap_num).zfill(3)}/'):
        os.makedirs(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/{sn_fac}/snap_{str(snap_num).zfill(3)}/')
    
    # return 

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

        file_list = np.array((5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26))
        for dm in dm_list:
            for sn_fac in sn_fac_list:
                print()
                print(f"{dm} - {sn_fac}\n")
                for snap_num in file_list:  
                    print()
                    halo_pos = find_halo(folder, dm, sn_fac, snap_num)
                    print(halo_pos)
                    for method in method_list:
                        read_write_param_file(folder, halo_pos, snap_num, dm, sn_fac, method)
                        write_out_dirs(folder,method,dm,sn_fac,snap_num)