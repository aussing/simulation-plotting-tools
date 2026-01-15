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

def subhalo_data(sim_directory,i_file):
    snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
    
    halo_R200c     = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH 
    halo_M200c     = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH 
    Halo_SFR       = np.array(haloinfo_data['Group']['GroupSFR'], dtype=np.float64)
    
    mass_mask = np.argsort(halo_M200c)[::-1]

    halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
    
    subhalo_halo_num   = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'], dtype=np.float64)
    subhalo_rank       = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.int32)
    
    subhalo_mass       = np.array(haloinfo_data['Subhalo']['SubhaloMass'],dtype=np.float64) * UNITMASS / LITTLEH
    subhalo_mass_type  = np.array(haloinfo_data['Subhalo']['SubhaloMassType'],dtype=np.float64) * UNITMASS / LITTLEH
    subhalo_len        = np.array(haloinfo_data['Subhalo']['SubhaloLen'],dtype=np.int32)
    subhalo_len_type   = np.array(haloinfo_data['Subhalo']['SubhaloLenType'],dtype=np.int32)
    subhalo_pos        = np.array(haloinfo_data['Subhalo']['SubhaloPos'], dtype=np.float64) #* LITTLEH 


    main_halo_subhalos = np.where(subhalo_halo_num==[mass_mask[halo_mainID]])

    subhalo_mass       = subhalo_mass[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0] 
    subhalo_mass_type  = subhalo_mass_type[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
    subhalo_len        = subhalo_len[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
    subhalo_len_type   = subhalo_len_type[main_halo_subhalos][subhalo_rank[main_halo_subhalos]!=0]
    

    return halo_M200c[mass_mask[halo_mainID]], halo_R200c[mass_mask[halo_mainID]], halo_masstypes[mass_mask[halo_mainID],4], Halo_SFR[mass_mask[halo_mainID]], subhalo_mass, subhalo_mass_type, subhalo_len_type

def write_props(folder_list,part_num_cut, file_name):
    sn_list = ['sn_005','sn_010']
    dm_list = ['cdm','wdm_3.5']
    for folder in folder_list:
        # file_name = f'./All_sim_MW_properties_test.txt'
        with open(file_name, "a") as properties_file:
            properties_file.writelines(f'Simulation -> {folder} \n')
            properties_file.close()
        for dm in dm_list :
            for sn in sn_list:
                print(f'{dm}, {sn}')
                if dm=='cdm' and sn=='sn_005':
                    continue
                elif dm=='wdm_3.5' and sn=='sn_010':
                    continue
                snapshot_loc = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
                try:
                    m_200c, r_200c, stellar_mass, sfr, subhalo_mass, subhalo_mass_type, subhalo_len_type = subhalo_data(snapshot_loc,26)
                    total_subs_gtr_min_num_part = np.where( np.sum( subhalo_len_type, axis=1 ) >= part_num_cut )[0]
                    sats_gtr_min_num_part = np.where(subhalo_mass_type[total_subs_gtr_min_num_part, 4] > 0)[0]
                    subhalo_mass_fraction = np.sum(subhalo_mass)/m_200c * 100.0
                    # file_name = f'./All_sim_MW_properties.txt'
                    # file_name = f'./{folder}/All_sim_MW_properties.txt'
                    with open(file_name, "a") as properties_file:
                        if dm =='cdm' and sn=='sn_005':
                            # properties_file.writelines(f'Simulation -> {folder} \n')
                            # dust_mass_file.writelines(f'\n')
                            properties_file.writelines(f'CDM SN 5%\n')
                        elif dm == 'cdm' and sn=='sn_010':                        
                            properties_file.writelines(f'CDM SN 10%\n')
                        elif dm == 'wdm_3.5' and sn=='sn_005':
                            properties_file.writelines(f'WDM SN 5%\n')
                        elif dm == 'wdm_3.5' and sn=='sn_010':
                            properties_file.writelines(f'WDM SN 10%\n')
                        
                        properties_file.writelines((f"M_200c        -> {np.round(m_200c/1e12,3)}e12 Msun\n"))
                        properties_file.writelines((f"M_200_stellar -> {np.round(stellar_mass/1e10,3)}e10 Msun\n"))
                        if stellar_mass<1e10:
                            print(f'Simulation - {folder}, DM - {dm}, SN - {sn}')
                        # print((f"M_200_stellar -> {np.round(stellar_mass/1e10,3)}e10 Msun\n"))
                        properties_file.writelines((f"R_200c        -> {np.round(r_200c*1000,3)} kpc\n"))
                        properties_file.writelines((f"SFR           -> {np.round(sfr,3)} Msun/yr\n"))
                        properties_file.writelines((f"Num sats      -> {len(subhalo_mass_type[subhalo_mass_type[:,4]>0])}\n"))
                        properties_file.writelines((f"Num sats > {part_num_cut} -> {len(sats_gtr_min_num_part)}\n"))
                        properties_file.writelines((f"SBH Mass Frac -> {np.round(subhalo_mass_fraction,3)}% \n"))
                        properties_file.writelines(f'\n')
                except:
                    print(f'Missing file for {snapshot_loc}')
                # print(len(total_subhalo_len))
            
if __name__ == "__main__":

    i_file  = 26
    
    folder_list = []
    # for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if ( simulation_name.startswith("N2048_L65_sd") )]:
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/') if ( simulation_name.startswith("N2048_L65_sd") )]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)

    file_name = f'./All_sim_MW_properties.txt'
    with open(file_name, "w") as prop_file:
        prop_file.close()
    part_num_cut = 20
    write_props(folder_list, part_num_cut, file_name)    



