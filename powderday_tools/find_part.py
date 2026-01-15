import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os
import cmasher as cmr

plt.rcParams['font.size'] = 11

LITTLEH = 0.6688
UNITMASS = 1e10

def set_center(folder):
    if folder=='N2048_L65_sd34920':
        halo_pos = [28.60889,34.54781,38.48793]
        extent = 0.5
    elif folder=='N2048_L65_sd46371':
        halo_pos = [27.48697,35.81327,32.22877]
        extent = 0.4
    return halo_pos,extent

def get_extent(halo_pos,extent):

    xmin,xmax  = halo_pos[0]-extent,halo_pos[0]+extent
    ymin,ymax  = halo_pos[1]-extent,halo_pos[1]+extent
    zmin,zmax  = halo_pos[2]-extent,halo_pos[2]+extent
    
    return xmin,xmax,ymin,ymax,zmin,zmax

def read_snap(folder,dm,SN_fac):

    snapshot = h5py.File(f"./{folder}/{dm}/{SN_fac}/snapshot_026.hdf5")
    fof = h5py.File(f"../../{folder}/{dm}/zoom/output/{SN_fac}/fof_subhalo_tab_026.hdf5")
    group_len = np.array(fof['Group']['GroupLen'], dtype=np.int64)[0]
    group_len_type = np.array(fof['Group']['GroupLenType'], dtype=np.int64)[0]
    print(group_len)
    print(group_len_type)
    gas_coords = np.array(snapshot['PartType0']['Coordinates'], dtype=np.float64)[0:group_len_type[0]]
    high_dm_coords = np.array(snapshot['PartType1']['Coordinates'], dtype=np.float64)[0:group_len_type[1]]
    star_coords = np.array(snapshot['PartType4']['Coordinates'], dtype=np.float64)[0:group_len_type[4]]

    return gas_coords,high_dm_coords,star_coords

def read_pd_file(folder,dm,SN_fac,snap_gas,snap_high_dm,snap_stars):
    halo_pos,extent = set_center(folder)
    grid_physical_properties_data = np.load(f'./{folder}/output/{dm}/{SN_fac}/grid_physical_properties.026_galaxy0.npz')
    
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)

    gas_pos_x = grid_physical_properties_data['gas_pos_x']/1e3*LITTLEH
    gas_pos_y = grid_physical_properties_data['gas_pos_y']/1e3*LITTLEH
    gas_pos_z = grid_physical_properties_data['gas_pos_z']/1e3*LITTLEH

    x_mask = (gas_pos_x>=xmin) & (gas_pos_x<=xmax)
    y_mask = (gas_pos_y>=ymin) & (gas_pos_y<=ymax)
    z_mask = (gas_pos_z>=zmin) & (gas_pos_z<=zmax)
    pos_mask = x_mask & y_mask & z_mask

    gas_pos = np.column_stack((gas_pos_x,gas_pos_y,gas_pos_z))
    print(gas_pos[0])
    print(snap_gas[0])

    # diff_x = np.round(snap_gas[:,0],8)/np.round(gas_pos_x[0:144354],8) - 1
    # print(len(diff_x[diff_x!=0]))
    round_x = np.round(gas_pos_x,8)
    round_snap_x = np.round(snap_gas[:,0],8)
    match_x = np.isin(round_snap_x,round_x,assume_unique=False)
    print(np.sum(match_x))
    # print(len(diff_x[np.where(diff_x<1e-10)]))
    # print(gas_pos.shape)
    # gas_pos = np.where(gas_pos[:,0]==snap_gas[:,0])[0]
    # # match_pos_x = np.isin(gas_pos_x,snap_gas[:,0],assume_unique=False)
    # # print(np.sum(match_pos_x))
    # print(gas_pos)

if __name__ =='__main__':
    folder = 'N2048_L65_sd34920'
    # folder = 'N2048_L65_sd46371'
    print(folder)
    # if not os.path.exists(f'./plots/{folder}'):
    #     os.makedirs(f'./plots/{folder}')
    dm = 'cdm'
    SN_fac = 'sn_010'
    snap_gas, snap_high_dm, snap_stars = read_snap(folder,dm,SN_fac)
    read_pd_file(folder,dm,SN_fac,snap_gas,snap_high_dm,snap_stars)