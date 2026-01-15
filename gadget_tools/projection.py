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


def set_plot_len(data):
    if UNIT_LENGTH_FOR_PLOTS in ['Mpc','mpc']:
        data = data/3.085678e24
    elif UNIT_LENGTH_FOR_PLOTS in ['Kpc','kpc']:
        data = data/3.085678e21
    elif UNIT_LENGTH_FOR_PLOTS == 'pc':
        data = data/3.085678e18
    else:
        print("What units do you want?????!!! AARRHH")
        raise TypeError
    return data

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

def DM_hist(sim_directory_cdm,cdm_label,sim_directory_wdm,wdm_label,i_file,folder):
    print(f'--- DM : {i_file} ---\n')
    hist = []
    edges = []
    subhalo_pos_list = []
    part_num_cut = 100
    num_bins = 200

    for sim_directory in [sim_directory_cdm,sim_directory_wdm]:
        snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
        unit_len = get_unit_len(snap_data)
        
        halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH
        halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH
        R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
        
        mass_mask = np.argsort(halo_M200c)[::-1]

        if folder == 'N2048_L65_sd_MW_ANDR':
            # if i_file<13:
            #     halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][2]
            # else:
                # halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][1]
            halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][1]
        else:
            halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
        print(f"HALO MAIN ID = {halo_mainID}")
        print(f"Redshift = {np.round(z,2)}")
        print(f"main halo ID = {mass_mask[halo_mainID]}")
        # print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],0],halo_pos[mass_mask[halo_mainID],1],halo_pos[mass_mask[halo_mainID],2]}")
        print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],:]}")
        print(f'halo_mass = {np.round(halo_mass[mass_mask[halo_mainID]]/1e10,3)}e10 h^-1 Msun')
        print(f"R_200 = {np.round(set_plot_len(R200c[mass_mask[halo_mainID]]*unit_len),2)} {UNIT_LENGTH_FOR_PLOTS}")

        extent = 2*R200c[mass_mask[halo_mainID]] #* unit_len

        dm_part_pos = np.array(snap_data[f'PartType1']['Coordinates'],dtype=np.float64) / LITTLEH #* unit_len  

        x_pos = dm_part_pos[:,0] 
        y_pos = dm_part_pos[:,1]
        z_pos = dm_part_pos[:,2]

        mask = make_mask(x_pos,y_pos,z_pos,halo_pos[mass_mask[halo_mainID],:],extent) # *unit_len

        dm_part_mass = snap_data['Header'].attrs["MassTable"][1] * UNITMASS / LITTLEH 
        
        dm_part_pos = dm_part_pos[mask]
        # dm_part_dist_x = (set_plot_len(dm_part_pos[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        # dm_part_dist_y = (set_plot_len(dm_part_pos[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        # dm_part_dist_z = (set_plot_len(dm_part_pos[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]

        dm_part_dist_x = ((dm_part_pos[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        dm_part_dist_y = ((dm_part_pos[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        dm_part_dist_z = ((dm_part_pos[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]
        mass_weight = np.ones(dm_part_pos[:,0].shape)*dm_part_mass
        
        ## Find subhalo positions
        subhalo_positions = np.array(haloinfo_data['Subhalo']['SubhaloPos'],dtype=np.float64) * unit_len / LITTLEH
        subhalo_radius = np.array(haloinfo_data['Subhalo']['SubhaloHalfmassRad'],dtype=np.float64) * unit_len / LITTLEH
        subhalo_groupID   = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'],dtype=np.int32)
        subhalo_rank      = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.int32)
        subhalo_len_type  = np.array(haloinfo_data['Subhalo']['SubhaloLenType'], dtype=np.int32)
        main_halo_subhalos = np.where(subhalo_groupID==[mass_mask[halo_mainID]])
        
        subhalo_len_type = subhalo_len_type[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        big_haloes = np.where(np.sum(subhalo_len_type,axis=1)>part_num_cut)[0]

        subhalo_positions = subhalo_positions[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_positions = subhalo_positions[big_haloes]
        
        subhalo_radius = subhalo_radius[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_radius = set_plot_len(subhalo_radius[big_haloes]) * 8
        
        subhalo_dist_x = (set_plot_len(subhalo_positions[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        subhalo_dist_y = (set_plot_len(subhalo_positions[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        # subhalo_dist_z = (set_plot_len(subhalo_positions[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]
        dists_xy = np.column_stack((subhalo_dist_x,subhalo_dist_y,subhalo_radius))
        
        subhalo_pos_list.append(dists_xy)

        
        map, edges_x,edges_y = np.histogram2d(dm_part_dist_x,dm_part_dist_y,bins=num_bins,weights=mass_weight)#,norm=mpl.colors.LogNorm())
        edges.append(np.array((np.min(edges_x),np.max(edges_x),np.min(edges_y),np.max(edges_y))))
        map[map==0]=dm_part_mass
        hist.append(map)
    print()
    # print(np.array(subhalo_pos_list[0]).shape)
    # print()
    main_halo_only = True
    
    total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, len_type_cdm_sn_010 = subhalo_data(sim_directory_cdm,i_file,main_halo_only)
    total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, len_type_wdm_sn_005 = subhalo_data(sim_directory_wdm,i_file,main_halo_only)
    
    dm_mass_cdm              = mass_type_cdm_sn_010[np.where(total_len_cdm_sn_010>part_num_cut)[0],1]
    dm_mass_ordered_cdm      = np.argsort(dm_mass_cdm)[::-1]
    dm_mas_cdm              = dm_mass_cdm[dm_mass_ordered_cdm][np.where(dm_mass_cdm[dm_mass_ordered_cdm]>0)[0]]
    dm_mass_sum_cdm          = np.cumsum(np.ones(dm_mas_cdm.shape[0]))
    
    dm_mass_wdm              = mass_type_wdm_sn_005[np.where(total_len_wdm_sn_005>part_num_cut)[0],1]
    dm_mass_ordered_wdm      = np.argsort(dm_mass_wdm)[::-1]
    dm_mas_wdm              = dm_mass_wdm[dm_mass_ordered_wdm][np.where(dm_mass_wdm[dm_mass_ordered_wdm]>0)[0]]
    dm_mass_sum_wdm          = np.cumsum(np.ones(dm_mas_wdm.shape[0]))
    
    hist_cdm = np.array(hist[0])
    hist_wdm = np.array(hist[1])
    
    cmap = mpl.cm.get_cmap('gray_r')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout = True)
    # fig, (ax1, ax2) = plt.subplots(1, 2,constrained_layout = True)
    
    im1 = ax1.imshow(np.transpose(hist_cdm),extent=edges[0],norm=mpl.colors.LogNorm(vmin=dm_part_mass,vmax=1e10),origin='lower',cmap=cmap) # np.max(hist)
    im2 = ax2.imshow(np.transpose(hist_wdm),extent=edges[1],norm=mpl.colors.LogNorm(vmin=dm_part_mass,vmax=1e10),origin='lower',cmap=cmap) # np.max(hist)
    
    # for i in range(len(subhalo_pos_list[0])):
    #     ax1.add_patch(plt.Circle((subhalo_pos_list[0][i,0],subhalo_pos_list[0][i,1]),subhalo_pos_list[0][i,2],color='r', fill=False))
    
    ax1.set_xticks((np.linspace(-2,2,5))) # ax1.set_xticks(np.round(np.linspace(edges[0][0],edges[0][1],7),2))
    ax1.set_yticks((np.linspace(-2,2,5))) # ax1.set_yticks(np.round(np.linspace(edges[0][2],edges[0][3],7),2))
    ax1.set(xlabel=r'R/R$_{200c}$')  #   ax1.set(xlabel=f'x [{UNIT_LENGTH_FOR_PLOTS}]')
    ax1.set(ylabel=r'R/R$_{200c}$')  #   ax1.set(ylabel=f'y [{UNIT_LENGTH_FOR_PLOTS}]')
    ax1.set_title(f'{cdm_label}')

    # for i in range(len(subhalo_pos_list[1])):
        # ax2.add_patch(plt.Circle((subhalo_pos_list[1][i,0],subhalo_pos_list[1][i,1]),subhalo_pos_list[1][i,2],color='r', fill=False))
    
    ax2.set(xlabel=r'R/R$_{200c}$')      #    ax2.set(xlabel=f'x [{UNIT_LENGTH_FOR_PLOTS}]')
    ax2.set(ylabel=r'R/R$_{200c}$')      #    ax2.set(ylabel=f'x [{UNIT_LENGTH_FOR_PLOTS}]')
    ax2.set_xticks(np.linspace(-2,2,5))  #   ax2.set_xticks(np.round(np.linspace(edges[1][0],edges[1][1],7),2))
    ax2.set_yticks(np.linspace(-2,2,5))  #   ax2.set_yticks(np.round(np.linspace(edges[1][2],edges[1][3],7),2))
    ax2.set_title(f'{wdm_label}')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1,cax=cax)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2,cax=cax)
    

    ax3.plot(dm_mas_cdm,dm_mass_sum_cdm,color='blue',label=f'{cdm_label}')#,linestyle='-')
    ax3.plot(dm_mas_wdm,dm_mass_sum_wdm,color='orange',label=f'{wdm_label}')#,linestyle='-')
    ax3.set(xlabel=r'M$_{DM}$ [M$_{\odot}$]')
    ax3.set(ylabel=r'N$>$M$_{DM}$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim((5e7,1.5e11))
    ax3.set_ylim((9e-1,3e2))
    ax3.set_title(f"DM subhaloes, min particles = {part_num_cut}")
    ax3.legend()

    fig.suptitle(f'Dark Matter distribution at z={z}',fontsize=20)
    if not os.path.exists(f'./{folder}/maps/DM/'):
        os.makedirs(f'./{folder}/maps/DM/')
    fig.savefig(f"./{folder}/maps/DM/DM_xy_map_{str(i_file).zfill(3)}.png",dpi=400)
    print(f' saved figure ./{folder}/maps/DM/DM_xy_map_{str(i_file).zfill(3)}.png')
    print()

def gas_hist(sim_directory_cdm,cdm_label,sim_directory_wdm,wdm_label,i_file,folder):
    print(f'--- Gas : {i_file} ---\n')
    hist = []
    edges = []
    subhalo_pos_list = []
    part_num_cut = 100
    num_bins = 200
    halo_pos_list = []

    for sim_directory in [sim_directory_cdm,sim_directory_wdm]:
        snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
        # unit_len = get_unit_len(snap_data)
        
        halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH
        
        halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH
        R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
        
        mass_mask = np.argsort(halo_M200c)[::-1]
        if folder == 'N2048_L65_sd_MW_ANDR':
            halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][1]
        else:
            halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
        
        # print(f"Redshift = {np.round(z,2)}")
        # print(f"main halo ID = {mass_mask[halo_mainID]}")
        # print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],0],halo_pos[mass_mask[halo_mainID],1],halo_pos[mass_mask[halo_mainID],2]}")
        # print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],:]*1000}")
        
        # print(f'halo_mass = {np.round(halo_mass[mass_mask[halo_mainID]]/1e10,3)}e10 h^-1 Msun')
        # print(f"R_200 = {np.round(set_plot_len(R200c[mass_mask[halo_mainID]]*unit_len),2)} {UNIT_LENGTH_FOR_PLOTS}")
        
        # halo_pos_list.append(halo_pos[mass_mask][halo_mainID]*unit_len)
        
        extent = 1*R200c[mass_mask[halo_mainID]] #* unit_len
        axes_linspace = np.linspace(-1,1,5)
        # extent = R200c[mass_mask[halo_mainID]] #*1000 #* unit_len

        gas_part_pos = np.array(snap_data[f'PartType0']['Coordinates'],dtype=np.float64) / LITTLEH #* 1000 #* unit_len

        x_pos = gas_part_pos[:,0] 
        y_pos = gas_part_pos[:,1]
        z_pos = gas_part_pos[:,2]

        mask = make_mask(x_pos,y_pos,z_pos,halo_pos[mass_mask][halo_mainID],extent) #*unit_len *1000

        gas_part_mass = np.array(snap_data[f'PartType0']['Masses'],dtype=np.float64) * UNITMASS / LITTLEH
        
        gas_part_pos = gas_part_pos[mask]
        # gas_part_dist_x = (set_plot_len(gas_part_pos[:,0]) - halo_pos[mass_mask[halo_mainID],0]) #/ R200c[mass_mask[halo_mainID]]
        # gas_part_dist_y = (set_plot_len(gas_part_pos[:,1]) - halo_pos[mass_mask[halo_mainID],1]) #/ R200c[mass_mask[halo_mainID]]
        # gas_part_dist_z = (set_plot_len(gas_part_pos[:,2]) - halo_pos[mass_mask[halo_mainID],2]) #/ R200c[mass_mask[halo_mainID]]

        gas_part_dist_x = ((gas_part_pos[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        gas_part_dist_y = ((gas_part_pos[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        gas_part_dist_z = ((gas_part_pos[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]
        mass_weight = gas_part_mass[mask]
        
        subhalo_positions = np.array(haloinfo_data['Subhalo']['SubhaloPos'],dtype=np.float64) / LITTLEH #* unit_len
        subhalo_radius = np.array(haloinfo_data['Subhalo']['SubhaloHalfmassRad'],dtype=np.float64) / LITTLEH #* unit_len
        subhalo_groupID   = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'],dtype=np.int32)
        subhalo_rank      = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.int32)
        subhalo_len_type  = np.array(haloinfo_data['Subhalo']['SubhaloLenType'], dtype=np.int32)
        main_halo_subhalos = np.where(subhalo_groupID==[mass_mask[halo_mainID]])
        
        subhalo_len_type = subhalo_len_type[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        big_haloes = np.where(np.sum(subhalo_len_type,axis=1)>part_num_cut)[0]

        subhalo_positions = subhalo_positions[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_positions = subhalo_positions[big_haloes]
        
        subhalo_radius = subhalo_radius[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_radius = set_plot_len(subhalo_radius[big_haloes]) * 8
        
        subhalo_dist_x = (set_plot_len(subhalo_positions[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        subhalo_dist_y = (set_plot_len(subhalo_positions[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        # subhalo_dist_z = (set_plot_len(subhalo_positions[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]
        dists_xy = np.column_stack((subhalo_dist_x,subhalo_dist_y,subhalo_radius))
        
        subhalo_pos_list.append(dists_xy)

        # num_bins = 200
        # num_bins = 50
        # gas_part_pos = set_plot_len(gas_part_pos)
        map, edges_x,edges_y = np.histogram2d(gas_part_dist_x,gas_part_dist_y,bins=num_bins,weights=mass_weight)#,norm=mpl.colors.LogNorm())
        # print(edges_y)
        # raise Exception
        edges.append(np.array((np.min(edges_x),np.max(edges_x),np.min(edges_y),np.max(edges_y))))
        # map[map==0]=np.unique(gas_part_mass)
        hist.append(map)
    hist_cdm = np.array(hist[0])
    hist_wdm = np.array(hist[1])
    
    main_halo_only = True
    part_num_cut = 100
    total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, len_type_cdm_sn_010 = subhalo_data(sim_directory_cdm,i_file,main_halo_only)
    total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, len_type_wdm_sn_005 = subhalo_data(sim_directory_wdm,i_file,main_halo_only)
    
    gas_mass_cdm              = mass_type_cdm_sn_010[np.where(total_len_cdm_sn_010>part_num_cut)[0],0]
    gas_mass_ordered_cdm      = np.argsort(gas_mass_cdm)[::-1]
    gas_mas_cdm              = gas_mass_cdm[gas_mass_ordered_cdm][np.where(gas_mass_cdm[gas_mass_ordered_cdm]>0)[0]]
    gas_mass_sum_cdm          = np.cumsum(np.ones(gas_mas_cdm.shape[0]))
    
    gas_mass_wdm              = mass_type_wdm_sn_005[np.where(total_len_wdm_sn_005>part_num_cut)[0],0]
    gas_mass_ordered_wdm      = np.argsort(gas_mass_wdm)[::-1]
    gas_mas_wdm              = gas_mass_wdm[gas_mass_ordered_wdm][np.where(gas_mass_wdm[gas_mass_ordered_wdm]>0)[0]]
    gas_mass_sum_wdm          = np.cumsum(np.ones(gas_mas_wdm.shape[0]))
    halo_pos_list = set_plot_len(np.array(halo_pos_list))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True)
    # fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    cmap = cmr.ember_r
    # im1 = ax1.imshow(np.transpose(hist_cdm),extent=edges[0],norm=mpl.colors.LogNorm(vmin=np.unique(gas_part_mass),vmax=1e9),origin='lower',cmap=cmap) # np.max(hist)
    # im2 = ax2.imshow(np.transpose(hist_wdm),extent=edges[1],norm=mpl.colors.LogNorm(vmin=np.unique(gas_part_mass),vmax=1e9),origin='lower',cmap=cmap) # np.max(hist)
    im1 = ax1.imshow(np.transpose(hist_cdm),extent=edges[0],norm=mpl.colors.LogNorm(vmin=np.unique(gas_part_mass),vmax=1e9),origin='lower',cmap=cmap) # np.max(hist)
    im2 = ax2.imshow(np.transpose(hist_wdm),extent=edges[1],norm=mpl.colors.LogNorm(vmin=np.unique(gas_part_mass),vmax=1e9),origin='lower',cmap=cmap) # np.max(hist)

    # axes_linspace = np.linspace(-2,2,5)
    # for i in range(len(subhalo_pos_list[0])):
    #     ax1.add_patch(plt.Circle((subhalo_pos_list[0][i,0],subhalo_pos_list[0][i,1]),subhalo_pos_list[0][i,2],color='k', fill=False))
    ax1.set_xticks(axes_linspace) # ax1.set_xticks(np.round(np.linspace(edges[0][0],edges[0][1],7),2))
    ax1.set_yticks(axes_linspace) # ax1.set_yticks(np.round(np.linspace(edges[0][2],edges[0][3],7),2))
    # ax1.scatter(halo_pos_list[0][0],halo_pos_list[0][1],color='blue',marker='^',s=50)
    # ax1.set(xlabel=f'x [{UNIT_LENGTH_FOR_PLOTS}]') ## 
    ax1.set(xlabel=r'R/R$_{200c}$')  #   
    # ax1.set(ylabel=f'y [{UNIT_LENGTH_FOR_PLOTS}]') ## 
    ax1.set(ylabel=r'R/R$_{200c}$')  #   
    ax1.set_title(f'{cdm_label}')

    # for i in range(len(subhalo_pos_list[1])):
    #     ax2.add_patch(plt.Circle((subhalo_pos_list[1][i,0],subhalo_pos_list[1][i,1]),subhalo_pos_list[1][i,2],color='k', fill=False))
    # ax2.scatter(halo_pos_list[1][0],halo_pos_list[1][1],color='blue',marker='^',s=50)
    # ax2.set(xlabel=f'x [{UNIT_LENGTH_FOR_PLOTS}]') 
    ax2.set(xlabel=r'R/R$_{200c}$')  #   
    # ax2.set(ylabel=f'y [{UNIT_LENGTH_FOR_PLOTS}]') ## 
    ax2.set(ylabel=r'R/R$_{200c}$')  #   
    ax2.set_xticks(axes_linspace) # ax2.set_xticks(np.round(np.linspace(edges[1][0],edges[1][1],7),2))
    ax2.set_yticks(axes_linspace) # ax2.set_yticks(np.round(np.linspace(edges[1][2],edges[1][3],7),2))
    ax2.set_title(f'{wdm_label}')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1,cax=cax)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2,cax=cax)

    # print(halo_pos_list)
    ax3.plot(gas_mas_cdm,gas_mass_sum_cdm,color='orange',label=f'{cdm_label}',linestyle='-')
    ax3.plot(gas_mas_wdm,gas_mass_sum_wdm,color='blue',label=f'{wdm_label}',linestyle='-.')
    ax3.set(xlabel=r'M$_{gas}$ [M$_{\odot}$]')
    ax3.set(ylabel=r'N$>$M$_{gas}$')
    ax3.set_xlim((1e6,1.5e10))
    ax3.set_ylim((9e-1,1e1))
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    # ax3.set_xlim((5e7,1.5e11))
    # ax3.set_ylim((9e-1,3e2))
    ax3.set_title(f"Gas subhaloes, min particles = {part_num_cut}")
    ax3.legend()

    fig.suptitle(f'Gas distribution at z={z}',fontsize=20)
    if not os.path.exists(f'./{folder}/maps/gas/'):
        os.makedirs(f'./{folder}/maps/gas/')
    fig.savefig(f"./{folder}/maps/gas/gas_xy_map_{str(i_file).zfill(3)}.png",dpi=400)
    print()

def star_hist(sim_directory_cdm,cdm_label,sim_directory_wdm,wdm_label,i_file,folder):
    print(f'--- Stars : {i_file} ---\n')
    hist = []
    edges = []
    subhalo_pos_list = []
    part_num_cut = 100
    num_bins = 200
    
    for sim_directory in [sim_directory_cdm,sim_directory_wdm]:
        snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
        # unit_len = get_unit_len(snap_data)
        
        halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH
        halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH
        R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
        
        mass_mask = np.argsort(halo_M200c)[::-1]
        if folder == 'N2048_L65_sd_MW_ANDR':
            if i_file<=12:
                halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][2]
            else:
                halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][1]
        else:
            halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
        # halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
        
        print(f"Redshift = {np.round(z,2)}")
        print(f"main halo ID = {mass_mask[halo_mainID]}")
        # print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],0],halo_pos[mass_mask[halo_mainID],1],halo_pos[mass_mask[halo_mainID],2]}")
        print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],:]}")
        print(f'halo_mass = {np.round(halo_mass[mass_mask[halo_mainID]]/1e10,3)}e10 h^-1 Msun')
        # print(f"R_200 = {np.round(set_plot_len(R200c[mass_mask[halo_mainID]]*unit_len),2)} {UNIT_LENGTH_FOR_PLOTS}")

        # extent = 2*R200c[mass_mask[halo_mainID]] #* unit_len
        extent = 1*R200c[mass_mask[halo_mainID]] #* unit_len
        axes_linspace = np.linspace(-1,1,5)

        star_part_pos = np.array(snap_data[f'PartType4']['Coordinates'],dtype=np.float64) / LITTLEH #* unit_len 

        x_pos = star_part_pos[:,0] 
        y_pos = star_part_pos[:,1]
        z_pos = star_part_pos[:,2]

        mask = make_mask(x_pos,y_pos,z_pos,halo_pos[mass_mask[halo_mainID],:],extent) #*unit_len

        star_part_mass = np.array(snap_data[f'PartType4']['Masses'],dtype=np.float64) * UNITMASS / LITTLEH
        
        star_part_pos = star_part_pos[mask]
        # star_part_dist_x = (set_plot_len(star_part_pos[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        # star_part_dist_y = (set_plot_len(star_part_pos[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        # star_part_dist_z = (set_plot_len(star_part_pos[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]

        star_part_dist_x = ((star_part_pos[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        star_part_dist_y = ((star_part_pos[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        star_part_dist_z = ((star_part_pos[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]
        mass_weight = star_part_mass[mask]
        
        subhalo_positions = np.array(haloinfo_data['Subhalo']['SubhaloPos'],dtype=np.float64)  / LITTLEH #* unit_len
        subhalo_radius = np.array(haloinfo_data['Subhalo']['SubhaloHalfmassRad'],dtype=np.float64) / LITTLEH #* unit_len 
        subhalo_groupID   = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'],dtype=np.int32)
        subhalo_rank      = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.int32)
        subhalo_len_type  = np.array(haloinfo_data['Subhalo']['SubhaloLenType'], dtype=np.int32)
        main_halo_subhalos = np.where(subhalo_groupID==[mass_mask[halo_mainID]])
        
        subhalo_len_type = subhalo_len_type[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        big_haloes = np.where(np.sum(subhalo_len_type,axis=1)>part_num_cut)[0]

        subhalo_positions = subhalo_positions[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_positions = subhalo_positions[big_haloes]
        
        subhalo_radius = subhalo_radius[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_radius = set_plot_len(subhalo_radius[big_haloes]) * 8
        
        # subhalo_dist_x = (set_plot_len(subhalo_positions[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        # subhalo_dist_y = (set_plot_len(subhalo_positions[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        # subhalo_dist_z = (set_plot_len(subhalo_positions[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]

        # subhalo_dist_x = ((subhalo_positions[:,0]) - halo_pos[mass_mask[halo_mainID],0]) / R200c[mass_mask[halo_mainID]]
        # subhalo_dist_y = ((subhalo_positions[:,1]) - halo_pos[mass_mask[halo_mainID],1]) / R200c[mass_mask[halo_mainID]]
        # subhalo_dist_z = ((subhalo_positions[:,2]) - halo_pos[mass_mask[halo_mainID],2]) / R200c[mass_mask[halo_mainID]]
        # mass_weight = star_part_mass[mask]
        # dists_xy = np.column_stack((subhalo_dist_x,subhalo_dist_y,subhalo_radius))
        
        # subhalo_pos_list.append(dists_xy)
        
        num_bins = 200
        map, edges_x,edges_y = np.histogram2d(star_part_dist_x,star_part_dist_y,bins=num_bins,weights=mass_weight)#,norm=mpl.colors.LogNorm())
        edges.append(np.array((np.min(edges_x),np.max(edges_x),np.min(edges_y),np.max(edges_y))))
        # map[map==0]=np.unique(star_part_mass)
        hist.append(map)
    
    hist_cdm = np.array(hist[0])
    hist_wdm = np.array(hist[1])
    
    
    main_halo_only = True
    part_num_cut = 100
    total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, len_type_cdm_sn_010 = subhalo_data(sim_directory_cdm,i_file,main_halo_only)
    total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, len_type_wdm_sn_005 = subhalo_data(sim_directory_wdm,i_file,main_halo_only)
    
    star_mass_cdm              = mass_type_cdm_sn_010[np.where(total_len_cdm_sn_010>part_num_cut)[0],4]
    star_mass_ordered_cdm      = np.argsort(star_mass_cdm)[::-1]
    star_mas_cdm              = star_mass_cdm[star_mass_ordered_cdm][np.where(star_mass_cdm[star_mass_ordered_cdm]>0)[0]]
    star_mass_sum_cdm          = np.cumsum(np.ones(star_mas_cdm.shape[0]))
    
    star_mass_wdm              = mass_type_wdm_sn_005[np.where(total_len_wdm_sn_005>part_num_cut)[0],4]
    star_mass_ordered_wdm      = np.argsort(star_mass_wdm)[::-1]
    star_mas_wdm              = star_mass_wdm[star_mass_ordered_wdm][np.where(star_mass_wdm[star_mass_ordered_wdm]>0)[0]]
    star_mass_sum_wdm          = np.cumsum(np.ones(star_mas_wdm.shape[0]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True)
    # fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    cmap = cmr.freeze_r
    
    im1 = ax1.imshow(np.transpose(hist_cdm),extent=edges[0],norm=mpl.colors.LogNorm(vmin=np.unique(star_part_mass),vmax=1e10),origin='lower',cmap=cmap) # np.max(hist)
    im2 = ax2.imshow(np.transpose(hist_wdm),extent=edges[1],norm=mpl.colors.LogNorm(vmin=np.unique(star_part_mass),vmax=1e10),origin='lower',cmap=cmap) # np.max(hist)
    
    # for i in range(len(subhalo_pos_list[0])):
    #     ax1.add_patch(plt.Circle((subhalo_pos_list[0][i,0],subhalo_pos_list[0][i,1]),subhalo_pos_list[0][i,2],color='r', fill=False))
    # axes_linspace = np.linspace(-2,2,5)
    ax1.set_xticks(axes_linspace) # ax1.set_xticks(np.round(np.linspace(edges[0][0],edges[0][1],7),2))
    ax1.set_yticks(axes_linspace) # ax1.set_yticks(np.round(np.linspace(edges[0][2],edges[0][3],7),2))
    ax1.set(xlabel=r'R/R$_{200c}$')  #   ax1.set(xlabel=f'x [{UNIT_LENGTH_FOR_PLOTS}]')
    ax1.set(ylabel=r'R/R$_{200c}$')  #   ax1.set(ylabel=f'y [{UNIT_LENGTH_FOR_PLOTS}]')
    ax1.set_title(f'{cdm_label}')

    # for i in range(len(subhalo_pos_list[1])):
        # ax2.add_patch(plt.Circle((subhalo_pos_list[1][i,0],subhalo_pos_list[1][i,1]),subhalo_pos_list[1][i,2],color='r', fill=False))

    ax2.set(xlabel=r'R/R$_{200c}$')  #   ax2.set(xlabel=f'x [{UNIT_LENGTH_FOR_PLOTS}]')
    ax2.set(ylabel=r'R/R$_{200c}$')  #   ax2.set(ylabel=f'y [{UNIT_LENGTH_FOR_PLOTS}]')
    ax2.set_xticks(axes_linspace) # ax2.set_xticks(np.round(np.linspace(edges[1][0],edges[1][1],7),2))
    ax2.set_yticks(axes_linspace) # ax2.set_yticks(np.round(np.linspace(edges[1][2],edges[1][3],7),2))
    ax2.set_title(f'{wdm_label}')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1,cax=cax)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2,cax=cax)

    if i_file == 26:
    # if i_file == 45:
        classical_mass_ordered, mass_sum = get_obs_mass()
        ax3.scatter(classical_mass_ordered,mass_sum,marker='^',s=50,c='red',label='Observed Satellites')
    ax3.plot(star_mas_cdm,star_mass_sum_cdm,color='blue',label=f'{cdm_label}')#,linestyle='-')
    ax3.plot(star_mas_wdm,star_mass_sum_wdm,color='orange',label=f'{wdm_label}')#,linestyle='-.')
    ax3.set(xlabel=r'M$_{*}$ [M$_{\odot}$]')
    ax3.set(ylabel=r'N$>$M$_{*}$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim((2e5,5e9))
    ax3.set_ylim((9e-1,15))
    ax3.set_title(f"Satellite galaxies, min particles = {part_num_cut}")
    ax3.legend()

    # fig.colorbar(im2)
    # plt.tight_layout()
    fig.suptitle(f'Stellar distribution at z={z}',fontsize=20)
    if not os.path.exists(f'./{folder}/maps/stars/'):
        os.makedirs(f'./{folder}/maps/stars/')
    fig.savefig(f"./{folder}/maps/stars/star_xy_map_{str(i_file).zfill(3)}.png",dpi=400)
    print(f"./{folder}/maps/stars/star_xy_map_{str(i_file).zfill(3)}.png")

if __name__ == "__main__":
    
    # folder = 'N2048_L65_sd00372'
    
    # folder  = 'N2048_L65_sd17492'
    # folder  = 'N2048_L65_sd28504'
    # folder  = 'N2048_L65_sd34920'
    # folder  = 'N2048_L65_sd46371'
    # folder  = 'N2048_L65_sd57839'
    # folder  = 'N2048_L65_sd61284'
    # folder  = 'N2048_L65_sd70562'
    # folder  = 'N2048_L65_sd80325'
    # folder  = 'N2048_L65_sd93745'
    # folder = 'N2048_L65_sd_MW_ANDR'

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/') if ( simulation_name.startswith(("N2048_L65_sd0","N2048_L65_sd1")) )]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    
    n_files = np.linspace(0,26,27,dtype=int)
    # n_files = np.linspace(0,45,46,dtype=int)
    # n_files = np.linspace(10,11,1,dtype=int)
    for folder in folder_list:
        
        if not os.path.exists(f'./{folder}'):
            os.makedirs(f'./{folder}')
        if not os.path.exists(f'./{folder}/maps/'):
            os.makedirs(f'./{folder}/maps/')
        
        cdm_sn_005 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_005/'
        cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'
        
        wdm_sn_005 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
        wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'

        for i_file in n_files:
            try:
                DM_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_005,"WDM SN=5\%",i_file,folder)
                plt.close()
                gas_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_005,"WDM SN=5\%",i_file,folder)
                plt.close()
                star_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_005,"WDM SN=5\%",i_file,folder)
                
                # DM_hist(cdm_sn_005,"CDM SN=5\%",wdm_sn_005,"WDM SN=5\%",i_file,folder)
                # plt.close()
                # gas_hist(cdm_sn_005,"CDM SN=5\%",wdm_sn_005,"WDM SN=5\%",i_file,folder)
                # plt.close()
                # star_hist(cdm_sn_005,"CDM SN=5\%",wdm_sn_005,"WDM SN=5\%",i_file,folder)
                # plt.close()
            except:
                print(f"no halo in snapshot {i_file}\n")

    # DM_hist(cdm_sn_005,"CDM SN=5\%",wdm_sn_005,"WDM SN=5\%",26,folder)
    # plt.close()
    # gas_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_005,"WDM SN=5\%",26,folder)
    # plt.close()
    # star_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_005,"WDM SN=5\%",26,folder)

    # DM_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_010,"WDM SN=10\%",45,folder)
    # plt.close()
    # gas_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_010,"WDM SN=10\%",45,folder)
    # plt.close()
    # star_hist(cdm_sn_010,"CDM SN=10\%",wdm_sn_010,"WDM SN=10\%",45,folder)

    # yt_gas_proj(cdm_zoom_dir,'cdm',folder)
    # yt_gas_proj(one_kev_zoom_dir,'1Kev',folder)
    # yt_gas_proj(Agora_dir,'Agora',folder)
    # yt_gas_proj(AG_comp_dir,'SH_03',folder)
    # pynbody_gas_proj(cdm_zoom_dir,folder)
    # yt_gas_proj(sim_directory,folder)
