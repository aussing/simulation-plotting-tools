import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py
import cmasher as cmr
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from tqdm import trange
from subhalo_count import *

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = [15,5]



LITTLEH    = 0.6688
UNITMASS   = 1.0e10
GASTYPE    = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5
# i_file     = 100
UNIT_LENGTH_FOR_PLOTS = 'kpc'

def get_sim_data(sim_directory,n_file):
    snap_fname     = f'/snapshot_{str(n_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(n_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = np.round(snap_data['Header'].attrs['Redshift'],2)
    return snap_data, haloinfo_data, z


def get_unit_len(snapshot):
    unit_length = snapshot["Parameters"].attrs['UnitLength_in_cm']
    
    return unit_length

def get_extent(halo_pos,extent):

    xmin,xmax  = halo_pos[0]-extent,halo_pos[0]+extent
    ymin,ymax  = halo_pos[1]-extent,halo_pos[1]+extent
    zmin,zmax  = halo_pos[2]-extent,halo_pos[2]+extent
    
    return xmin,xmax,ymin,ymax,zmin,zmax

def make_mask(pos_x,pos_y,pos_z,halo_pos,extent):
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)
    
    x_mask = (pos_x>=xmin) & (pos_x<=xmax)
    y_mask = (pos_y>=ymin) & (pos_y<=ymax)
    z_mask = (pos_z>=zmin) & (pos_z<=zmax)
    
    pos_mask = x_mask & y_mask & z_mask

    return pos_mask

def plot_2d_hist(i_file, folder, particle_type, dm, sn):
    sim_dir = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
    part_num_cut = 100
    num_bins = 200

    snap_data, haloinfo_data, z = get_sim_data(sim_dir,i_file)
    
    halo_pos          = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) #/ LITTLEH
    halo_mass         = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) #* UNITMASS / LITTLEH
    halo_M200c        = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS #/ LITTLEH
    halo_masstypes    = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS #/ LITTLEH
    R200c             = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) #/ LITTLEH
    group_offset_type = np.array(haloinfo_data['Group']['GroupOffsetType'], dtype=np.int64)
    group_len_type    = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.int64)

    mass_mask         = np.argsort(halo_M200c)[::-1]
    
    halo_mainID       = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
    # halo_mainID       = np.where(halo_M200c == np.max(halo_M200c))[0][0]

    target_mass_200c  = halo_M200c[mass_mask[halo_mainID]]
    target_radius     = R200c[mass_mask[halo_mainID]]
    target_offset     = group_offset_type[mass_mask[halo_mainID]]
    target_len        = group_len_type[mass_mask[halo_mainID]]

    print(f"Main halo ID     = {mass_mask[halo_mainID]}")
    print(f"Main halo pos    = {halo_pos[mass_mask[halo_mainID],:]}")
    print(f'Main halo mass   = {(target_mass_200c):.3e}  Msun/h')
    print(f"Main halo radius = {(R200c[mass_mask[halo_mainID]]):.3} Mpc/h")

    ## Subhalo data
    subhalo_groupID    = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'],dtype=np.int32)
    subhalo_len_type   = np.array(haloinfo_data['Subhalo']['SubhaloLenType'], dtype=np.int32)
    main_halo_subhalos = np.where(subhalo_groupID==[mass_mask[halo_mainID]])
    
    subhalo_len_type_main_halo   = subhalo_len_type[main_halo_subhalos]
    # subhalo_positions  = subhalo_positions[main_halo_subhalos]

    # first_particle_index = target_offset
    # target_part_type_offset = first_particle_index[particle_type]
    # target_part_type_len    = target_len[particle_type]
    # subhalo_0_len           = subhalo_len_type[0,particle_type]
    # all_subhalo_part_len    = np.sum(subhalo_len_type[:,particle_type])

    start_index         = target_offset[particle_type]
    end_sub_0_index     = target_offset[particle_type] + subhalo_len_type_main_halo[0,particle_type]
    subhalo_start_index = target_offset[particle_type] + subhalo_len_type_main_halo[0,particle_type] + 1
    subhalo_end_index   = target_offset[particle_type] + np.sum(subhalo_len_type_main_halo[:,particle_type]) -1
    start_diffuse_index = target_offset[particle_type] + np.sum(subhalo_len_type_main_halo[:,particle_type])
    end_index           = target_offset[particle_type] + target_len[particle_type]

    # start_index         = target_part_type_offset
    # end_sub_0_index     = start_index+subhalo_0_len
    # start_diffuse_index = start_index+all_subhalo_part_len
    # end_index           = start_index+target_part_type_len
    

    # subhalo_0_ID_list =  np.array(snap_data[f'PartType{particle_type}']['ParticleIDs'],dtype=np.int64)[start_index:end_sub_0_index]
    # diffuse_ID_list   =  np.array(snap_data[f'PartType{particle_type}']['ParticleIDs'],dtype=np.int64)[start_diffuse_index:end_index]
    # combined_ID_list = np.concatenate((subhalo_0_ID_list,diffuse_ID_list))

    whole_group_coords = np.array(snap_data[f'PartType{particle_type}']['Coordinates'],dtype=np.float64)[start_index:end_index]
    subhalo_0_coords = np.array(snap_data[f'PartType{particle_type}']['Coordinates'],dtype=np.float64)[start_index:end_sub_0_index]
    diffuse_0_coords = np.array(snap_data[f'PartType{particle_type}']['Coordinates'],dtype=np.float64)[start_diffuse_index:end_index]
    subhaloes_coords = np.array(snap_data[f'PartType{particle_type}']['Coordinates'],dtype=np.float64)[subhalo_start_index:subhalo_end_index]
      
    particle_coords = np.concatenate((subhalo_0_coords,diffuse_0_coords))

    if particle_type ==0:
        cmap = mpl.colormaps.get_cmap('plasma')
    elif particle_type == 1:
        cmap = mpl.colormaps.get_cmap('viridis')
    elif particle_type == 4:
        cmap = mpl.colormaps.get_cmap('gist_stern')
    grid_size = 512

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout = True)
    ax1.hexbin(whole_group_coords[:,0], whole_group_coords[:,1], bins='log', gridsize=grid_size, cmap=cmap)
    ax2.hexbin(subhalo_0_coords[:,0],   subhalo_0_coords[:,1],   bins='log', gridsize=grid_size, cmap=cmap)
    # ax3.hexbin(diffuse_0_coords[:,0],   diffuse_0_coords[:,1],   bins='log', gridsize=grid_size, cmap=cmap)
    ax3.hexbin(subhaloes_coords[:,0],   subhaloes_coords[:,1],   bins='log', gridsize=grid_size, cmap=cmap)

    # ax1.set_title(f"All group particles")
    # ax2.set_title(f"Subhalo 0 Particles")
    # ax3.set_title(f"FOF only particles")

    plt.savefig(f"./random_plots/projection_{folder}_PartType{particle_type}.png",dpi=300)
    print(f'Saved figure ./random_plots/projection_{folder}_PartType{particle_type}.png')
    # print()

if __name__ == "__main__":

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') 
        if (simulation_name.startswith("N2048_L65_sd05"))]:
        
        folder_list.append(simulation_name)
    
    folder_list = np.sort(folder_list)
    
    # n_files = np.linspace(0,26,27,dtype=int)
    i_file = 26
    particle_type = 1 # 0->gas, 1->HR_DM, 4->stars, 5->LR_DM

    dm = 'cdm' # 'wdm_3.5'
    sn = 'sn_010' # 'sn_005
    for folder in folder_list:
        print(f"\nUsing Simulation {folder}, DM = {dm}, SN = {sn}, Particle Type = PartType{particle_type}\n")
        plot_2d_hist(i_file,folder, particle_type, dm, sn)



       