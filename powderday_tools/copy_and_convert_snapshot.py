## Copies simulation snapshots to correct powderday folders 
## Needs top level folders beforehand !!
## Puts everything in Kpc units because yt hates me

import numpy as np
import h5py
import tqdm
from tqdm import trange
import shutil
import os

def convert(file,dm,SN_fac,snap_num):
    
    file = f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{dm}/{SN_fac}/snapshot_{str(snap_num).zfill(3)}.hdf5'
    print(f'file = {file}')
    # old_snap = h5py.File(file, 'r')
    # a = np.array(old_snap["PartType0/Coordinates"])
    # print(a)
    
    # old_snap.close()

    with h5py.File(file,'r+') as ds:
        print(np.array(ds["PartType0/Coordinates"][0]))
        # print(np.array(ds["PartType0/Coordinates"][0])*1000)
        ds["PartType0/Coordinates"][:] = np.array(ds["PartType0/Coordinates"])*1000.0
        print(np.array(ds["PartType0/Coordinates"][0]))

        ds["PartType0/SmoothingLength"][:] = np.array(ds["PartType0/SmoothingLength"])*1000
        ds["PartType1/Coordinates"][:]     = np.array(ds["PartType1/Coordinates"])*1000
        ds["PartType4/Coordinates"][:]     = np.array(ds["PartType4/Coordinates"])*1000
        ds["PartType5/Coordinates"][:]     = np.array(ds["PartType5/Coordinates"])*1000
        ds['Header'].attrs['BoxSize']      = np.array(ds['Header'].attrs['BoxSize'],dtype=np.float32)*1000
        ds['Parameters'].attrs['BoxSize']  = np.array(ds['Parameters'].attrs['BoxSize'])*1000
        ds["PartType0/Density"][:]         = np.array(ds["PartType0/Density"][:])/1e9

        ds["Parameters"].attrs['UnitLength_in_cm'] = ds["Parameters"].attrs['UnitLength_in_cm']/1000
        
        # print(np.array(ds['Header'].attrs['BoxSize']))
        ds.close()
        
    # new_snap = h5py.File(file, 'r')
    # a = np.array(new_snap["PartType0/Coordinates"])
    # print(a)
    # new_snap.close()
    # print('Finsihd file - ',file)

def copy_file(folder,dm,SN_fac,snap_num):

    print(f'copying file: /fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/snapshot_{str(snap_num).zfill(3)}.hdf5')
    
    if not os.path.exists(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/'):
        os.makedirs(      f'/fred/oz217/aussing/clean_pd/powderday/{folder}/')

    if not os.path.exists(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{dm}/'):
        os.makedirs(      f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{dm}/')

    if not os.path.exists(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{dm}/{SN_fac}/'):
        os.makedirs(      f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{dm}/{SN_fac}/')

    shutil.copy(f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/snapshot_{str(snap_num).zfill(3)}.hdf5',
                f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{dm}/{SN_fac}/')

if __name__=='__main__':

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/') if (simulation_name.startswith(("N2048_L65_sd12","N2048_L65_sd14")) )]:#and not simulation_name.startswith(("N2048_L65_sd07")))]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(folder_list)
    
    for folder in folder_list:
        dm_list = ['cdm', 'wdm_3.5']
        sn_fac_list = ['sn_005','sn_010']
        file_list = np.array((5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26))
        
        for dm in dm_list:
            for sn_fac in sn_fac_list:
                print()
                for snap_num in file_list: 
                    copy_file(folder,dm,sn_fac,snap_num)
        
        for dm in dm_list:
            for sn_fac in sn_fac_list:
                for snap_num in file_list:  
                    # print(file_name)
                    convert(folder,dm,sn_fac,snap_num)

