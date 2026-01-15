import numpy as np
import matplotlib.pyplot as plt
import h5py
import decimal
import os
from scipy.stats import bootstrap
# from astropy.io import fits

plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 14

LITTLEH    = 0.6688
UNITMASS   = 1.0e10
GASTYPE    = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5
G          = 4.3e-9# Mpc Msun^-1 (km/s)^2



def plot_total(total_mass, mass_type, total_len, part_num_cut, z, color, label, linestyle,main_halo_only=True, alpha=1):

    total_mass           = total_mass[np.where(total_len>part_num_cut)[0]]
    total_mass_ordered   = np.argsort(total_mass)[::-1]
    total_mass           = total_mass[total_mass_ordered][np.where(total_mass[total_mass_ordered]>0)[0]]
    total_mass_sum       = np.cumsum(np.ones(total_mass.shape[0]))

    plt.plot(total_mass,total_mass_sum,color=color,label=label,linestyle=linestyle,alpha=alpha)
    # if main_halo_only:
    #     plt.xlim((5e7,1.5e11))
    #     plt.ylim((9e-1,3e2))
    # else:
    #     plt.xlim((5e7,1.5e11))
    #     plt.ylim((9e-1,5e3))
    # plt.xlim((5e7,1.5e11))
    # plt.ylim((9e-1,3e2))
    # plt.ylim((9e-1,30))

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'M$_{total}$ [M$_{\odot}$]')
    # plt.ylabel(r'N$>$M$_{total}$')
    
    # # plt.title(f'Subhalo population at z={np.round(z,2)}')#, Min particles = {part_num_cut}')
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    
    # if main_halo_only:
    #     plt.title(f'Total MW halo - min part = {part_num_cut}')
    #     # plt.savefig(f'./test_plots/total_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
    #     plt.savefig(f'{folder}/subhalo/total_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
    # else:
    #     plt.title(f'Total Pop - min part = {part_num_cut}')
    #     # plt.savefig(f'./test_plots/total_mass_population_part_cut_{part_num_cut}.png',dpi=400)
    #     plt.savefig(f'{folder}/subhalo/total_mass_population_part_cut_{part_num_cut}.png',dpi=400)
    
def plot_gas(total_mass, mass_type, total_len, part_num_cut,z,color,label,linestyle,main_halo_only=True,alpha=1):
    
    gas_mass             = mass_type[np.where(total_len>part_num_cut)[0],0]
    gas_mass_ordered     = np.argsort(gas_mass)[::-1]
    gas_mass             = gas_mass[gas_mass_ordered][np.where(gas_mass[gas_mass_ordered]>0)[0]]
    gas_mass_sum         = np.cumsum(np.ones(gas_mass.shape[0]))
    # print(mass_type[np.where(total_len>part_num_cut)[0],0])
    plt.plot(gas_mass,gas_mass_sum,color=color,label=label,linestyle=linestyle,alpha=alpha)
    # if main_halo_only:
    #     plt.xlim((7.5e7,1.5e11))
    #     plt.ylim((9e-1,10))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'M$_{gas}$ [M$_{\odot}$]')
    # plt.ylabel(r'N$>$M$_{gas}$')
    # # plt.title(f'subhaloes with gas at z={np.round(z,2)}')#, Min particles = {part_num_cut}')
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # if main_halo_only:
    #     plt.savefig(f'{folder}/subhalo/gas_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
    # else:
    #     plt.savefig(f'{folder}/subhalo/gas_mass_population_part_cut_{part_num_cut}.png',dpi=400)

def plot_dm(total_mass, mass_type, total_len, part_num_cut,z,color,label,linestyle,main_halo_only=True,alpha=1):
    
    dm_mass              = mass_type[np.where(total_len>part_num_cut)[0],1]
    dm_mass_ordered      = np.argsort(dm_mass)[::-1]
    dm_mass              = dm_mass[dm_mass_ordered][np.where(dm_mass[dm_mass_ordered]>0)[0]]
    dm_mass_sum          = np.cumsum(np.ones(dm_mass.shape[0]))
    
    plt.plot(dm_mass,dm_mass_sum,color=color,label=label,linestyle=linestyle,alpha=alpha)
    # if main_halo_only:
    #     plt.xlim((5e7,1.5e11))
    #     plt.ylim((9e-1,3e2))
    # else:
    #     plt.xlim((5e7,1.5e11))
    #     plt.ylim((9e-1,5e3))
    # plt.xlim((5e7,1.5e11))
    # plt.ylim((9e-1,4e2))
    # plt.ylim((9e-1,30))

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'M$_{DM}$ [M$_{\odot}$]')
    # plt.ylabel(r'N$>$M$_{DM}$')
    # # plt.title(f'subhaloes with DM at z={np.round(z,2)}')#, Min particles = {part_num_cut}')
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # # plt.savefig(f'./test_plots/DM_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
    # if main_halo_only:
    #     plt.title(f'DM MW Halo - min part = {part_num_cut}')
    #     # plt.savefig(f'./test_plots/DM_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
    #     plt.savefig(f'{folder}/subhalo/DM_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
    # else:
    #     plt.title(f'DM Pop - min part = {part_num_cut}')
    #     # plt.savefig(f'./test_plots/DM_mass_pop_part_cut_{part_num_cut}.png',dpi=400)
    #     plt.savefig(f'{folder}/subhalo/DM_mass_population_part_cut_{part_num_cut}.png',dpi=400)

def plot_stellar(total_mass, mass_type, total_len, part_num_cut,z,color,label,linestyle,main_halo_only=True,alpha=1):

    stellar_mass         = mass_type[np.where(total_len>part_num_cut)[0],4]
    stellar_mass_ordered = np.argsort(stellar_mass)[::-1]
    stellar_mass         = stellar_mass[stellar_mass_ordered][np.where(stellar_mass[stellar_mass_ordered]>0)[0]]
    stellar_mass_sum     = np.cumsum(np.ones(stellar_mass.shape[0]))
    
    
    bins = np.geomspace(4e5,1e10,30)
    bins_middle = (bins[1:]+bins[:-1])/2
    binned_sats_count,bin_edges = np.histogram(stellar_mass,bins=bins)
    binned_cumulative_counts = np.cumsum(binned_sats_count[::-1])[::-1]
    plt.plot(bins_middle,binned_cumulative_counts,label=label,linestyle=linestyle,alpha=alpha, color=color)
    # sat_bin_membership = np.digitize(stellar_mass,bins)
    # satellites_in_bin = []
    # for i in range(len(bins)):
    #     satellites_in_bin.append(len(np.where(sat_bin_membership==i)[0]))
    #     satellites_in_bin.append(len(np.where(sat_bin_membership==i)[0])+np.sum(satellites_in_bin))
    #     # print(len(np.where(sat_bin_membership==i)[0]))
    # satellites_in_bin = np.array(satellites_in_bin)
    # print(satellites_in_bin)
    # print(np.sum(satellites_in_bin))
    # # binned_sats,bin_edges = np.histogram(satellites_in_bin,bins=bins)
    # plt.plot(bins,satellites_in_bin)
    

    # plt.plot(stellar_mass,stellar_mass_sum,label=label,linestyle=linestyle,alpha=alpha, color=color)
    return bins_middle, binned_cumulative_counts

def get_sim_data(sim_directory,i_file):
    snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = (snap_data['Header'].attrs['Redshift'])
    return snap_data, haloinfo_data, z

def get_obs_mass(normalise=False):
    classical_name =['LMC','SMC','Fornax','Leo1','Sculptor','Leo2','SagittariusdSph','Sextans(1)','Carina','Draco','UrsaMinor']
    classical_mass = 1.5e9,4.8e8,   2.0e7, 5.5e6,     2.3e6, 7.4e6,            2.1e7,       4.4e5,   3.8e5, 2.9e5 ,     2.9e5 #from Lovell 2023
    classical_mass = np.array(classical_mass,dtype=np.float64)
    sort_cls_mass = np.argsort(classical_mass)[::-1]

    distance = 50,61,149,258,86,236,18,89,107,76,102
    distance = np.array(distance,dtype=np.int32)
    srt_distance = np.argsort(distance)[::-1]

    # volume = 4/3 * np.pi * (distance/1e3)**3 
    volume = 4/3 * np.pi * (distance)**3 
    # print(f'Volume = {np.max(volume)} Mpc^3')
    MW_mass = 1e12

    mass_sum = np.cumsum(np.ones(classical_mass[sort_cls_mass].shape[0]))

    # if normalise:
    #     plt.scatter(classical_mass[sort_cls_mass]/MW_mass,mass_sum*np.max(volume[srt_distance]),marker='^',s=50,c='red',label='Observed Satellites')
    # else:
    #     plt.scatter(classical_mass[sort_cls_mass],mass_sum,marker='^',s=50,c='red',label='Observed Satellites')
    #     # plt.scatter(classical_mass[sort_cls_mass],mass_sum,c='red',label='Observed Satellites')/volume[srt_distance]
    # plt.legend()
    return classical_mass[sort_cls_mass], mass_sum

def subhalo_data(sim_directory,i_file,main_halo_only=True):
    snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
        
    halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) #* LITTLEH 
    halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH 
    halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH 
    Halo_SFR = np.array(haloinfo_data['Group']['GroupSFR'], dtype=np.float64)
    
    mass_mask = np.argsort(halo_M200c)[::-1]
    # mass_mask = mass_mask[halo_M200c[mass_mask]<3e12]

    # if folder == 'N2048_L65_sd_MW_ANDR':
    if 'N2048_L65_sd_MW_ANDR' in sim_directory:
        halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][2]
    else:
        halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
    # halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
    # print(f"Redshift = {np.round(z,2)}")
    # print(f"main halo ID = {mass_mask[halo_mainID]}")
    # print(f"Main halo coords = {halo_pos[mass_mask[halo_mainID],:]}")
    # print(f'M_200c = {np.round(halo_M200c[mass_mask[halo_mainID]]/1e12,3)}e12 Msun')
    # print(f"R_200c = {np.round(halo_R200c[mass_mask[halo_mainID]],2)} Mpc")
    # print(f"Group SFR = {Halo_SFR[mass_mask[halo_mainID]]} Msun/Year")
    subhalo_halo_num = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'], dtype=np.float64)
    # subhalo_rank = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.float64)
    
    subhalo_rank       = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.int32)
    
    subhalo_mass       = np.array(haloinfo_data['Subhalo']['SubhaloMass'],dtype=np.float64) * UNITMASS / LITTLEH
    subhalo_mass_type  = np.array(haloinfo_data['Subhalo']['SubhaloMassType'],dtype=np.float64) * UNITMASS / LITTLEH
    subhalo_len        = np.array(haloinfo_data['Subhalo']['SubhaloLen'],dtype=np.int32)
    subhalo_len_type   = np.array(haloinfo_data['Subhalo']['SubhaloLenType'],dtype=np.int32)
    subhalo_pos        = np.array(haloinfo_data['Subhalo']['SubhaloPos'], dtype=np.float64) #* LITTLEH 
    # print(subhalo_pos[subhalo_rank!=0][subhalo_mass[subhalo_rank!=0] == np.max(subhalo_mass[subhalo_rank!=0])])
    
    #To check if Nomalising by volume is something to think about? - Less than 1% for sim 4, most others should all be the same anyway
    # pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
    # highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64)
    # x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
    # y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
    # z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
    # box_volume=x_extent*y_extent*z_extent
    # print(f"volume = {box_volume}")

    if main_halo_only:
        main_halo_subhalos = np.where(subhalo_halo_num==[mass_mask[halo_mainID]])

        subhalo_mass       = subhalo_mass[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0] 
        subhalo_mass_type  = subhalo_mass_type[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_len        = subhalo_len[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        subhalo_len_type   = subhalo_len_type[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
        
        subhalo_pos = np.array(haloinfo_data['Subhalo']['SubhaloPos'], dtype=np.float64)
        subhalo_pos = subhalo_pos[main_halo_subhalos]
        subhalo_pos_dif = np.abs(subhalo_pos[0,:]-subhalo_pos[1,:])
        # print(f"subhalo rank 0 - {subhalo_pos[0,:]}")
        # print(f"subhalo rank 1 - {subhalo_pos[1,:]}\n")

        mod_subhalo_pos_dif = np.sqrt(subhalo_pos_dif[0]**2+subhalo_pos_dif[1]**2+subhalo_pos_dif[2]**2)
        if (mod_subhalo_pos_dif < 5e-2): # In case there's something weird with the main halo
            subhalo_mass = subhalo_mass[1:]
            subhalo_mass_type = subhalo_mass_type[1:] 
            subhalo_len = subhalo_len[1:] 
            subhalo_len_type = subhalo_len_type[1:] 
            print('Removed first subhalo from the list')
    else:
        subhalo_mass       = subhalo_mass[subhalo_rank!=0][subhalo_len_type[subhalo_rank!=0][:,5]==0]
        subhalo_mass_type  = subhalo_mass_type[subhalo_rank!=0][subhalo_len_type[subhalo_rank!=0][:,5]==0]
        subhalo_len        = subhalo_len[subhalo_rank!=0][subhalo_len_type[subhalo_rank!=0][:,5]==0]
        subhalo_len_type   = subhalo_len_type[subhalo_rank!=0][subhalo_len_type[subhalo_rank!=0][:,5]==0]
        print()
    return subhalo_mass, subhalo_mass_type, subhalo_len, subhalo_len_type

def mass_count_total(sim_directory,label,folder,color,i_file,linestyle,alpha=1,obs=0):
        print('\n',label)
        snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)

        if (obs==1) and (np.round(z,2)==0):
            get_obs_mass(normalise=True)

        halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) #* LITTLEH
        halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) #* LITTLEH
        halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH

        mass_mask = np.argsort(halo_mass[halo_mass<4e12])[::-1]
        halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
        print(f"Redshift = {np.round(z,2)}")
        print(f"main halo ID = {mass_mask[halo_mainID]}")
        print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],0],halo_pos[mass_mask[halo_mainID],1],halo_pos[mass_mask[halo_mainID],2]}")
        print(f'halo_mass = {np.round(halo_mass[mass_mask[halo_mainID]]/1e10,3)}e10 h^-1 Msun')
        print(f"R_200 = {np.round(halo_R200c[mass_mask[halo_mainID]],2)} Mpc")
        
        subhalo_halo_num = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'], dtype=np.int32)  
        subhalo_rank = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.float64)  
        
        
        subhalo_mass_type = np.array(haloinfo_data['Subhalo']['SubhaloMassType'],dtype=np.float64) * UNITMASS / LITTLEH

        stel_sat = np.isin(subhalo_mass_type[:,4],subhalo_mass_type[:,4][subhalo_mass_type[:,4]>0])
        non_central_subhalo = np.isin(subhalo_mass_type[:,4],subhalo_mass_type[:,4][subhalo_rank!=0])
        clean_haloes = np.isin(subhalo_mass_type[:,4],subhalo_mass_type[:,4][subhalo_mass_type[:,5]==0])
        
        mask = stel_sat & non_central_subhalo & clean_haloes

        stel_mass = subhalo_mass_type[mask,4]
        
        sat_group_nr = subhalo_halo_num[mask]
        
        host_mass = halo_M200c[sat_group_nr]
        print(np.where(halo_M200c[sat_group_nr] ==np.min(halo_M200c)))
        print(len(stel_mass[host_mass==0]))
        stel_frac = stel_mass/host_mass
        
        
        # Need to normalise by volume, too hard to determine in final snap, just use cubic volume defined in t=0/z=127 (first) snapshot
        pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
        highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64)
        x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
        y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
        z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
        box_volume=x_extent*y_extent*z_extent
        # box_volume=(x_extent*y_extent*z_extent)/1e9

        stel_frac_ordered = np.argsort(stel_frac)[::-1]
        subhalo_frac_sum = np.cumsum(np.ones(stel_frac[stel_frac_ordered].shape[0]))

        # num_clean_halos = len(halo_M200c[np.where(halo_masstypes[:,5] == 0)])
        # plt.plot(stel_frac[stel_frac_ordered],subhalo_frac_sum/len(halo_M200c[np.where(subhalo_mass_type[:,5]==0)]),color=color,label=label)

        norm_volume = subhalo_frac_sum/(box_volume)
        norm_num = subhalo_frac_sum/ np.max(subhalo_frac_sum)
        # plt.plot(stel_frac[stel_frac_ordered],norm_volume,color=color,label=label,linestyle=linestyle,alpha=alpha)
        # plt.plot(stel_frac[stel_frac_ordered],norm_num,color=color,label=label,linestyle=linestyle,alpha=alpha)
        plt.plot(stel_frac[stel_frac_ordered],subhalo_frac_sum,color=color,label=label,linestyle=linestyle,alpha=alpha)

        
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel(r'$M_{*}$/$M_{total, host}$')
        plt.ylabel(r'N$>$M$_{*}$ [Mpc$^{-3}$ h$^3$]')
        # plt.title(f'Satellite population at Redshift = {np.round(z,2)}')
        plt.legend()
        
        
        plt.savefig(f'{folder}/sub_mass_bound_total.png',dpi=400)
        # plt.savefig(f'./sub_mass_bound_total.png',dpi=400)
        
def safe_fig(part_num_cut, dm, plot_type, ax, main_halo_only=False):
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.xlim(2e5,1e10)
    plt.ylim(9e-1,2.7e1)
    print(dm)
    # print(f'test/{dm}_{plot_type}_main_halo_part_cut_{part_num_cut}.png')
    if main_halo_only:

        # if dm.split('_')[0] == str('cdm'):
        #     if dm.split('_')[-1] == str('005'):
        #         plt.text(0.95,0.95,"CDM SN=5\% ",ha='right', va='top',transform=ax.transAxes,fontsize=50.0,fontweight='heavy')
        #     else:
        #         plt.text(0.95,0.95,"CDM SN=10\% ",ha='right', va='top',transform=ax.transAxes,fontsize=50.0,fontweight='heavy')
        # else:# dm.split('_')[0:1] == str('wdm_3.5'):
        #     if dm.split('_')[-1] == str('005'):
        #         plt.text(0.95,0.95,"WDM SN=5\%",ha='right', va='top',transform=ax.transAxes,fontsize=50.0,fontweight='heavy')
        #     else:
        #         plt.text(0.95,0.95,"WDM SN=10\%",ha='right', va='top',transform=ax.transAxes,fontsize=50.0,fontweight='heavy')

        # plt.title(f'{dm} {plot_type}; n = {part_num_cut}')
        # plt.savefig(f'./test_plots/total_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
        plt.savefig(f'combined_plots/{dm}_{plot_type}_main_halo_part_cut_{part_num_cut}.png',dpi=400)
        print(f'combined_plots/{dm}_{plot_type}_main_halo_part_cut_{part_num_cut}.png')
    else:
        plt.title(f'{dm} {plot_type}; n = {part_num_cut}')
        # plt.savefig(f'./test_plots/total_mass_population_part_cut_{part_num_cut}.png',dpi=400)
        plt.savefig(f'combined_plots/{dm}_{plot_type}_population_part_cut_{part_num_cut}.png',dpi=400)
        
def main_halo_plots(folder_list,i_file,part_num_cut):
    # snap_data, haloinfo_data, z = get_sim_data(cdm_sn_010,i_file)
    main_halo_only = True
    # good_list_dust_mass = ['N2048_L65_sd17492', 'N2048_L65_sd28504', 'N2048_L65_sd46371', 'N2048_L65_sd61284', 'N2048_L65_sd70562']#
    # good_list_gini = ['N2048_L65_sd17492', 'N2048_L65_sd46371', 'N2048_L65_sd70562']#
    sn_list = ['sn_005','sn_010']
    dm_list = ['cdm','wdm_3.5']
    for dm in dm_list :
        for sn in sn_list:
            # print(dm, dm.type)
            fig, ax = plt.subplots()
            all_binned_sat_counts = []
            for folder in folder_list:
                snapshot_loc = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
                total_mass, mass_type, total_len, len_type = subhalo_data(snapshot_loc,i_file,main_halo_only)
                # if folder in good_list_dust_mass:
                #     if dm =='cdm':
                #         bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'lime',  'CDM SN=\%' , '-', main_halo_only, alpha=1)
                #     else:
                #         bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'red',  'WDM SN=5\%' , '-', main_halo_only, alpha=1)
                # if folder in good_list_gini:
                #     if dm =='cdm':
                #         bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'blue',  'CDM SN=\%' , '-', main_halo_only, alpha=1)
                #     else:
                #         bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'orange',  'WDM SN=5\%' , '-', main_halo_only, alpha=1)
                if dm == 'cdm':
                    if sn == 'sn_005':
                        bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'blue',  'CDM SN=5\%' , '-', main_halo_only, alpha=0.3)    
                    else :
                        bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'blue',  'CDM SN=10\%' , '-', main_halo_only, alpha=0.3)    
                else:
                    if sn == 'sn_005':   
                        bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'orange',  'WDM SN=5\%' , '-', main_halo_only, alpha=0.3)    
                    else:
                        bins_middle, sat_counts = plot_stellar(total_mass, mass_type, total_len, part_num_cut, 0, 'orange',  'WDM SN=10\%' , '-', main_halo_only, alpha=0.3)    
                all_binned_sat_counts.append(sat_counts)
            
            all_binned_sat_counts = np.array(all_binned_sat_counts)
            # print(all_binned_sat_counts,'\n')
            # median_binned_sats = np.median(all_binned_sat_counts,axis=0)
            # plt.plot(bins_middle,median_binned_sats,c='k')

            mean_binned_sats = np.mean(all_binned_sat_counts,axis=0)
            plt.plot(bins_middle,mean_binned_sats,c='k')

            binned_data_btstrp_low = []
            binned_data_btstrp_high = []
            # print(all_binned_sat_counts[:,0])
            for i in range(all_binned_sat_counts.shape[1]):
                data = (all_binned_sat_counts[:,i],)
                # print(data)
                # btstrp_data = bootstrap(data,np.median,confidence_level=0.99,random_state=1,method='percentile')
                btstrp_data = bootstrap(data,np.mean,confidence_level=0.99,random_state=1,method='percentile')
                # plt.figure()
                # plt.hist(btstrp_data.bootstrap_distribution,bins=30)
                # plt.axvline(btstrp_data.confidence_interval[0], ls='--')
                # plt.axvline(btstrp_data.confidence_interval[1], ls='--')
                # plt.savefig(f'./test_plots/{dm}_bin_median_{i}.png',dpi=300)
                ci_low,ci_high = btstrp_data.confidence_interval
                binned_data_btstrp_low.append(ci_low)
                binned_data_btstrp_high.append(ci_high)
                # print(ci_low,ci_high)
                # print(np.array(binned_data_btstrp_high))
            if dm == 'cdm':
                plt.fill_between(bins_middle,binned_data_btstrp_low, binned_data_btstrp_high, color='blue',alpha=0.3,ls='None')
            else:
                plt.fill_between(bins_middle,binned_data_btstrp_low, binned_data_btstrp_high, color='orange',alpha=0.3,ls='None')
        
            
            # mean_binned_sats = np.mean(all_binned_sat_counts,axis=0)
            # plt.plot(bins_middle,mean_binned_sats,c='k')

            classical_mass_odered, mass_sum = get_obs_mass()
            ###
            plt.scatter(classical_mass_odered,mass_sum,marker='^',s=400,c='red',label='Observed Satellites')
            plt.xlabel(r'M$_{*}$ [M$_{\odot}$]')
            plt.ylabel(r'N$>$M$_{*}$')
            safe_fig(part_num_cut,f'{dm}_{sn}', 'stellar', ax, main_halo_only)
            plt.close()
            ###

            # for folder in folder_list:
            #     cdm_sn_005 = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
            #     total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, len_type_cdm_sn_005 = subhalo_data(cdm_sn_005,i_file,main_halo_only)
            #     plot_total(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, 0, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)    
            # plt.xlabel(r'M$_{*}$ [M$_{\odot}$]')
            # plt.ylabel(r'N$>$M$_{*}$')
            # safe_fig(20,f'{dm}_{sn}', 'total', main_halo_only)
            # plt.close()

            # for folder in folder_list:
            #     cdm_sn_005 = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
            #     total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, len_type_cdm_sn_005 = subhalo_data(cdm_sn_005,i_file,main_halo_only)
            #     plot_gas(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, 0, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)    
            # plt.xlabel(r'M$_{*}$ [M$_{\odot}$]')
            # plt.ylabel(r'N$>$M$_{*}$')
            # safe_fig(20,f'{dm}_{sn}', 'gas', main_halo_only)
            # plt.close()

            # for folder in folder_list:
            #     cdm_sn_005 = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
            #     total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, len_type_cdm_sn_005 = subhalo_data(cdm_sn_005,i_file,main_halo_only)
            #     plot_dm(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, 0, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)    
            # plt.xlabel(r'M$_{*}$ [M$_{\odot}$]')
            # plt.ylabel(r'N$>$M$_{*}$')
            # safe_fig(20,f'{dm}_{sn}', 'dm', main_halo_only)
            plt.close()

    # plot_total(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_gas(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_dm(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'

    # wdm_sn_005 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
    # wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'
        # print("--- cdm_sn_010 ---")
        # total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, len_type_cdm_sn_010 = subhalo_data(cdm_sn_010,i_file,main_halo_only)
        # print("--- wdm_sn_005 ---")
        # total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, len_type_wdm_sn_005 = subhalo_data(wdm_sn_005,i_file,main_halo_only)
        # print("--- wdm_sn_010 ---")
        # total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, len_type_wdm_sn_010 = subhalo_data(wdm_sn_010,i_file,main_halo_only)
        # print()
    
    
    
    # print(f'Minimum particle limit = {part_num_cut}\n')
    # plt.close()
    # print('plotting main halo subhalo total mass function')
    # plot_total(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_total(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' , main_halo_only, alpha=1)
    # plot_total(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_total(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' , main_halo_only, alpha=1)
    
    # plt.close()
    # print('plotting main halo subhalo gas mass function')
    # plot_gas(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_gas(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' , main_halo_only, alpha=1)
    # plot_gas(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_gas(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' , main_halo_only, alpha=1)
    
    # plt.close()
    # print('plotting main halo subhalo DM mass function')
    # plot_dm(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_dm(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' , main_halo_only, alpha=1)
    # plot_dm(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_dm(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' , main_halo_only, alpha=1)
    
    # plt.close()
    # fig,ax = plt.subplots()
    # classical_mass_odered, mass_sum = get_obs_mass()
    # # plt.scatter(classical_mass_odered,mass_sum,marker='^',s=400,c='red',label='Observed Satellites')
  
    # # print('plotting main halo subhalo stellar mass function')
    # plot_stellar(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', fig, ax, main_halo_only, alpha=1)
    # plot_stellar(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' , fig, ax, main_halo_only, alpha=1)
    # plot_stellar(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.', fig, ax, main_halo_only, alpha=1)
    # plot_stellar(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' , fig, ax, main_halo_only, alpha=1)
    # plt.scatter(classical_mass_odered,mass_sum,marker='^',s=400,c='red',label='Observed Satellites')
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # plt.savefig(f'./stellar_mass_main_halo_part_cut_{part_num_cut}.png',dpi=400)
    # plt.close()

def population_plots(folder,i_file,part_num_cut):
    cdm_sn_005 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_005/'
    cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'

    wdm_sn_005 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
    wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'

    main_halo_only = False
    print("--- cdm_sn_005 ---")
    total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, len_type_cdm_sn_005 = subhalo_data(cdm_sn_005,i_file,main_halo_only)
    print("--- cdm_sn_010 ---")
    total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, len_type_cdm_sn_010 = subhalo_data(cdm_sn_010,i_file,main_halo_only)
    print("--- wdm_sn_005 ---")
    total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, len_type_wdm_sn_005 = subhalo_data(wdm_sn_005,i_file,main_halo_only)
    print("--- wdm_sn_010 ---")
    total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, len_type_wdm_sn_010 = subhalo_data(wdm_sn_010,i_file,main_halo_only)
    
    snap_data, haloinfo_data, z = get_sim_data(cdm_sn_010,i_file)
    
    print(f'Minimum particle limit = {part_num_cut}')
    plt.close()
    print('plotting population subhalo total mass function')
    # plot_total(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_total(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' , main_halo_only, alpha=1)
    # plot_total(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.',main_halo_only, alpha=1)
    # plot_total(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' ,main_halo_only, alpha=1)
    
    # plt.close()
    # print('plotting population subhalo gas mass function')
    # plot_gas(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_gas(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' , main_halo_only, alpha=1)
    # plot_gas(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.',main_halo_only, alpha=1)
    # plot_gas(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' ,main_halo_only, alpha=1)
    
    # plt.close()
    # print('plotting population subhalo DM mass function')
    # plot_dm(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.', main_halo_only, alpha=1)
    # plot_dm(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' , main_halo_only, alpha=1)
    # plot_dm(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.',main_halo_only, alpha=1)
    # plot_dm(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' ,main_halo_only, alpha=1)
    
    plt.close()
    fig, ax = plt.subplots()
    print('plotting population subhalo stellar mass function')
    plot_stellar(total_mass_cdm_sn_005, mass_type_cdm_sn_005, total_len_cdm_sn_005, part_num_cut, z, 'blue',  'CDM SN=5\%' , '-.',  fig, ax, main_halo_only, alpha=1)
    plot_stellar(total_mass_cdm_sn_010, mass_type_cdm_sn_010, total_len_cdm_sn_010, part_num_cut, z, 'blue',  'CDM SN=10\%', '-' ,  fig, ax, main_halo_only, alpha=1)
    plot_stellar(total_mass_wdm_sn_005, mass_type_wdm_sn_005, total_len_wdm_sn_005, part_num_cut, z, 'orange','WDM SN=5\%' , '-.', fig, ax, main_halo_only, alpha=1)
    plot_stellar(total_mass_wdm_sn_010, mass_type_wdm_sn_010, total_len_wdm_sn_010, part_num_cut, z, 'orange','WDM SN=10\%', '-' , fig, ax, main_halo_only, alpha=1)
    
    plt.close()


if __name__ == "__main__":

    # folder  = 'N2048_L65_sd17492'
    # folder  = 'N2048_L65_sd28504'
    # folder  = 'N2048_L65_sd34920'
    # folder  = 'N2048_L65_sd46371'
    # folder  = 'N2048_L65_sd57839'
    # folder  = 'N2048_L65_sd61284'
    # folder  = 'N2048_L65_sd70562'
    # folder  = 'N2048_L65_sd80325'
    # folder  = 'N2048_L65_sd93745'
    i_file  = 26

    # folder  = 'N2048_L65_sd_MW_ANDR'
    # i_file  = 45

    # folder_list = ['N2048_L65_sd17492', 'N2048_L65_sd28504', 'N2048_L65_sd34920', 'N2048_L65_sd46371', 'N2048_L65_sd57839', 'N2048_L65_sd61284', 'N2048_L65_sd70562', 'N2048_L65_sd80325', 'N2048_L65_sd93745']
    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/') if ( simulation_name.startswith("N2048_L65_sd"))]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    plt.close()

    part_num_cut=[20,100,200]
    for num,part_num in enumerate(part_num_cut):
        main_halo_plots(folder_list, i_file, part_num)
    
    # population_plots(folder_list,i_file)
    


