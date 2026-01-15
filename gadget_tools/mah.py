import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.ticker import AutoMinorLocator
import os
# from scipy.stats import bootstrap

plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 11
LITTLEH = 0.6688
UNITMASS = 1e10

def get_sim_data(sim_directory,n_file):
    snap_fname     = f'/snapshot_{str(n_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(n_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = (snap_data['Header'].attrs['Redshift'])
    return snap_data, haloinfo_data, z

def check_low_res(sim_directory,I_FILE):

    sn0 = h5py.File(sim_directory+'snapshot_000.hdf5','r')
    sn_I_FILE = h5py.File(sim_directory+f'snapshot_{str(I_FILE).zfill(3)}.hdf5','r')
    sub_I_FILE = h5py.File(sim_directory+f'fof_subhalo_tab_{str(I_FILE).zfill(3)}.hdf5','r')

    M200c = np.array(sub_I_FILE['Group']['Group_M_Crit200'], dtype=np.float64)
    R200c = np.array(sub_I_FILE['Group']['Group_R_Crit200'], dtype=np.float64)
    GrpPos = np.array(sub_I_FILE['Group']['GroupPos'], dtype=np.float64)

    box_size = sn0['Header'].attrs['BoxSize']
    h = sn0['Parameters'].attrs['HubbleParam']
    # massInterval = [(0.5e12/1.0e10)*h, (2e12/1.0e10)*h]
    massInterval = [(1.4e12/1.0e10)*h, (2e12/1.0e10)*h]
    i_mass_select = np.where( (M200c>massInterval[0]) & (M200c<massInterval[1]) )[0]

    # ISOLATION CRITERION
    minDist = 9.0*h
    # minDist = 9e3*h
    minMassDist = 0.5*massInterval[0]
    i_perturber = np.where( (M200c>minMassDist))[0]
    i_isolated = []
    for i_candidate in i_mass_select:
        dist2 = ( GrpPos[i_candidate,0] - GrpPos[i_perturber, 0] )**2 + \
        ( GrpPos[i_candidate,1] - GrpPos[i_perturber, 1] )**2 + \
        ( GrpPos[i_candidate,2] - GrpPos[i_perturber, 2] )**2
        i_ngb = np.where(dist2 < minDist**2)[0]
        if len(i_ngb) < 2:   ## always should contain itself
            i_isolated.append(i_candidate)
        
            
    # SELECTION OF THE MOST CENTRAL GALAXY (CLOSEST TO CENTER OF BOX)
    BoxHalf = box_size/2
    dists = []
    for i_candidate in i_isolated:
        dist2 = ( GrpPos[i_candidate, 0] - BoxHalf)**2 + \
        ( GrpPos[i_candidate, 1] - BoxHalf)**2 + \
        ( GrpPos[i_candidate, 2] - BoxHalf)**2
        dists.append(np.sqrt(dist2))
    dists = np.array(dists)

    i_select = np.where(dists == np.min(dists))[0][0]
    i_halo = i_isolated[i_select]
    pos = GrpPos[i_halo, :]
    print('Selected halo %d \nPosition %g %g %g'%(i_halo, pos[0], pos[1], pos[2]))

    print(f'Mass of selected halo = {np.round(M200c[i_halo],4)} ^12 M_sun/h')    
    mass_target = M200c[i_halo]
    return mass_target, pos

def check_zoom(sim_directory,I_FILE):

    snap_data, haloinfo_data, z = get_sim_data(sim_directory,I_FILE)
    halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) 
    halo_mass  = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH

    mass_mask = np.argsort(halo_mass)[::-1]
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.int32) * UNITMASS / LITTLEH
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0]
    # print('Halo pos = ',halo_pos[mass_mask[halo_mainID]])
    main_halo_mass = halo_mass[mass_mask[halo_mainID]]
    main_halo_pos = halo_pos[mass_mask[halo_mainID]]
    
    return main_halo_mass, main_halo_pos

def compare_tree_and_plot(tree_dir, target_mass, target_pos, label,fig,ax,color,ls='-',alpha=0.3,normalisation=True, plot=True):
    tree_data = h5py.File(f'{tree_dir}/trees.hdf5', 'r')
    Group_m_200c = np.array(tree_data['TreeHalos/Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    # Group_mass = np.array(tree_data['TreeHalos/Group_Mass'], dtype=np.float64) * UNITMASS / LITTLEH
    subhalo_mass = np.array(tree_data['TreeHalos/SubhaloMass'], dtype=np.float64) * UNITMASS / LITTLEH
    Main_prog = np.array(tree_data['TreeHalos/TreeMainProgenitor'])
    snap_num = np.array(tree_data['TreeHalos/SnapNum'])
    scale_factor = np.array(tree_data['TreeTimes/Time'])
    subhalo_pos = np.array(tree_data['TreeHalos/SubhaloPos'], dtype=np.float64)
    TreeId = np.array(tree_data['TreeHalos/TreeID'])
    redshift = np.array(tree_data['TreeTimes/Redshift'])
    first_halo_in_FOF_group = np.array(tree_data['TreeHalos/TreeFirstHaloInFOFgroup'])
    eps = 1e-5
    a = np.unique(np.where(np.abs(subhalo_pos-target_pos)<eps)[0])
    b = np.where(np.abs(subhalo_mass-target_mass)<10*UNITMASS)[0]
    c = np.where(snap_num==np.max(snap_num))
    
    # index = int(a[np.isin(a,b,c)][0])
    index = int(a[np.isin(a,c)][0])
    # if len(a[np.isin(a,c)]) > 1:
    #     raise Exception
    # print(a,index)
    tree_ID = np.where(TreeId==TreeId[index])[0]
    
    tree_group_mass = Group_m_200c[tree_ID]
    # tree_group_mass = subhalo_mass[tree_ID]
    
    tree_snapnumber = snap_num[tree_ID]

    tree_main_prog = Main_prog[tree_ID]
    FHIFG = first_halo_in_FOF_group[tree_ID]
    # print(FHIFG)
    # print(FHIFG.shape)
    snapnum_list = np.zeros(len(scale_factor))
    mah = np.zeros(len(scale_factor))#* [0,]
    i = 0
    # print(tree_main_prog.shape)
    # print(tree_main_prog.dtype)
    while tree_main_prog[i]!=-1:
        # print(tree_snapnumber[i])
        mah[tree_snapnumber[i]] = tree_group_mass[i]
        # print(i, tree_group_mass[i]/UNITMASS)
        snapnum_list[tree_snapnumber[i]]=tree_snapnumber[i]
        # print(tree_snapnumber[i])
        # print('FHIFG     = ',FHIFG[tree_snapnumber[i]])
        # print('Main prog = ',tree_main_prog[tree_snapnumber[i]],'\n')
        if tree_main_prog[FHIFG[i]] != tree_main_prog[i]:
            i = FHIFG[i]
            print('aaaaa')
        else:
            i = tree_main_prog[i]
            # print(tree_snapnumber[i])
    if normalisation:
        mah = mah/mah[-1]
    else:
        mah = mah/UNITMASS
    # print(mah)
    # print(scale_factor.shape)
    



    # plt.plot(scale_factor[1::],mah[1::],label=label,color=color,ls=ls)
    if plot == True:
        plt.plot(scale_factor,mah,label=label,color=color,ls=ls,alpha=alpha)
    # plt.plot(np.round(redshift[mah>0],2),mah[mah>0],label=label)

    z_labels = np.array([0,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,4])
    new_x_ax = 1/(1+z_labels)
    ax.set_xticks(new_x_ax, z_labels)
    # x_ax = scale_factor[2::3]
    # x_label = np.round(1/scale_factor-1,2)[2::3]
    # plt.xticks(x_ax,x_label )
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    return scale_factor, mah

if __name__ == "__main__":

    i_file = 26
    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/') if ( simulation_name.startswith("N2048_L65_sd") )]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(folder_list)

#####
# Plot individual MAH's
    # for folder in folder_list:

    #     if not os.path.exists(f'./{folder}'):
    #         os.makedirs(f'./{folder}')
    #     sn_list = ['sn_005','sn_010']
    #     dm_list = ['cdm','wdm_3.5']
    #     fig, ax = plt.subplots()
    #     for dm in dm_list :
    #         for sn in sn_list:
    #             snap_location = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
    #             mass, position = check_zoom(snap_location,i_file)
    #             if dm =='cdm':
    #                 if sn == 'sn_005':
    #                     scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "CDM SN=5\%" , fig, ax, 'blue', ls='-.',   alpha=1, normalisation=False)
    #                 else:  
    #                     scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "CDM SN=10\%", fig, ax, 'blue', ls='-',    alpha=1, normalisation=False)
    #             else:
    #                 if sn == 'sn_005':
    #                     scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "WDM SN=5\%" , fig, ax, 'orange', ls='-.', alpha=1, normalisation=False)
    #                 else:
    #                     scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "WDM SN=10\%", fig, ax, 'orange', ls='-',  alpha=1, normalisation=False)
    #     plt.legend()
    #     plt.xlabel('Redshift')
    #     # plt.yscale('log')
    #     plt.ylabel(r'$M_{200c}$ [10$^{10}$ $M_{\odot}$]')
    #     plt.tight_layout()
    #     plt.gca().invert_xaxis()

    #     plt.savefig(f'./{folder}/mass_accretion_history.png', dpi=250)
    #     plt.close()
    #     print(f'Saved figure ./{folder}/mass_accretion_history.png')
## Plot individual MAH's
#####

#####
## Plot all MAH's
    good_list_dust_mass = ['N2048_L65_sd00372' , 'N2048_L65_sd03157','N2048_L65_sd01829','N2048_L65_sd05839' 'N2048_L65_sd17492','N2048_L65_sd28504', 'N2048_L65_sd46371', 'N2048_L65_sd61284', 'N2048_L65_sd70562']#
    bad_list_dust_mass = ['N2048_L65_sd34920','N2048_L65_sd04721']# 
    # good_list_gini = ['N2048_L65_sd17492']#, 'N2048_L65_sd46371', 'N2048_L65_sd70562']#
    sn_list = ['sn_005','sn_010']
    dm_list = ['cdm','wdm_3.5']

    for dm in dm_list :
        # print(dm)
        for sn in sn_list:
            if dm =='cdm' and sn == 'sn_010':
                continue
            # print(sn)
            fig, ax = plt.subplots()
            mah_list = []
            for folder in folder_list:
                # print(folder)
                snap_location = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
                snap_location_cdm_10 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'
                
                mass, position = check_zoom(snap_location,i_file)
                mass_cdm_10, position_cdm_10 = check_zoom(snap_location_cdm_10,i_file)

                scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "CDM SN=5%" , fig, ax, 'blue',ls='-', alpha=1,plot=False)
                scale_factor_cdm_10, mah_cdm_10 = compare_tree_and_plot(snap_location_cdm_10, mass_cdm_10, position_cdm_10, "CDM SN=10%" , fig, ax, 'orange',ls='-', alpha=1,plot=False)
                # print(mah/mah_cdm_10)
                mah_comp = np.nan_to_num(mah[4::]/mah_cdm_10[4::],copy=False,nan=0.0)
                # print(mah_comp)
                mah_list.append(mah_comp)
                plt.plot(scale_factor[4::],mah_comp*100-100, color='gray', alpha = 0.3)
            mah_list = np.array(mah_list)
            median_mah = np.median(mah_list,axis=0)
            plt.plot(scale_factor[4::],median_mah*100-100, color='k', alpha = 1)
            plt.axhline(0,ls=':',c='k')
            plt.ylim(-10,10)
            plt.xlabel('Redshift')            
            plt.gca().invert_xaxis()
            plt.ylabel(r'Target $M_{200c}$ / Reference $M_{200c, z=0}$')
            plt.tight_layout()
            plt.savefig(f'./combined_plots/AA_MAH_{dm}_{sn}_cdm_10_ref.png',dpi=300)

    plt.close() 
    exit()           
    for dm in dm_list :
        # print(dm)
        for sn in sn_list:
            # print(sn)
            fig, ax = plt.subplots()
            mah_list = []
            for folder in folder_list:
                # print(folder)
                snap_location = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn}/'
                mass, position = check_zoom(snap_location,i_file)
                # if folder not in good_list_dust_mass:
                #     if dm =='cdm':
                #         scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "CDM SN=5%" , fig, ax, 'blue',ls='-', alpha=0.5)
                #         formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                #         # plt.axvline(formation_time, ymax=0.2, color='k', ls='--', alpha=0.5)
                #     else:
                #         scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "" , fig, ax, 'orange',ls='-', alpha=0.5)
                #         formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                        # plt.axvline(formation_time, ymax=0.2, color='k', ls='--', alpha=0.5)
                # if folder in good_list_gini:
                #     if dm =='cdm':
                #         scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "CDM SN=5%" , fig, ax, 'blue',ls='-', alpha=1)
                #         # formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                #         # plt.axvline(formation_time, ymax=0.2, color='k', ls='--', alpha=0.5)
                #     else:
                #         scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "" , fig, ax, 'orange',ls='-', alpha=1)
                #         # formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                #         # plt.axvline(formation_time, ymax=0.2, color='k', ls='--', alpha=0.5)
                # elif dm == 'cdm':
                alpha = 1
                # alpha = 0.3
                # if folder in good_list_dust_mass: ### =='N2048_L65_sd46371':
                #     alpha = 1
                if dm == 'cdm':
                    if sn == 'sn_005':                        
                        scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "CDM SN=5%" , fig, ax, 'blue',ls='-', alpha=alpha)
                        formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                        # plt.axvline(formation_time, ymax=0.2,color='k', ls='--', alpha=0.5)
                    else :
                        scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "CDM SN=10%" , fig, ax, 'blue',ls='-', alpha=alpha)
                        formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                        # plt.axvline(formation_time, ymax=0.2, color='k', ls='--', alpha=0.5)
                else:
                    if sn == 'sn_005':   
                        scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "WDM SN=5%" , fig, ax, 'orange',ls='-', alpha=alpha)
                        formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                        # plt.axvline(formation_time, ymax=0.2, color='k', ls='--', alpha=0.5)
                    else:
                        scale_factor, mah = compare_tree_and_plot(snap_location, mass, position, "WDM SN=10%" , fig, ax, 'orange',ls='-', alpha=alpha) 
                        formation_time = scale_factor[np.argmin(np.abs(mah-0.5))]
                        # plt.axvline(formation_time, ymax=0.2, color='k', ls='--', alpha=0.5)
                mah_list.append(mah)
                # compare_tree_and_plot(cdm_sn_005, cdm_sn_005_mass, cdm_sn_005_pos, "CDM SN=5%" , fig, ax, 'blue',ls='-.')
                # compare_tree_and_plot(cdm_sn_010, cdm_sn_010_mass, cdm_sn_010_pos, "CDM SN=10%", fig, ax, 'blue')
                # compare_tree_and_plot(wdm_sn_005, wdm_sn_005_mass, wdm_sn_005_pos, "WDM SN=5%" , fig, ax, 'orange',ls='-.')
                # compare_tree_and_plot(wdm_sn_010, wdm_sn_010_mass, wdm_sn_010_pos, "WDM SN=10%", fig, ax, 'orange')
            mah_list = np.array(mah_list)
            median_mah = np.median(mah_list,axis=0)
            # print(mah_list.shape)
            file_name = f'./mah_merger_history.txt'
            redshift = np.round(1/scale_factor-1,2)
            # with open(file_name, "a") as properties_file:
            #     if dm =='cdm' and sn=='sn_005':
            #         properties_file.writelines(f'\nCDM SN 5%\n')
            #         for i in range(mah_list.shape[0]):
            #             properties_file.writelines(f'\nSimulation = {folder_list[i]}\n')
            #             for j in range(mah_list.shape[1]):
            #                 if j !=0 :
            #                     if ((mah_list[i,j]/mah_list[i,j-1]*100)-100) > 30 : 
            #                         properties_file.writelines(f'MAJOR MERGER: z = {redshift[j]}, mass ratio = {np.round(((mah_list[i,j]/mah_list[i,j-1]*100)-100),4)}%\n')
            #                         plt.scatter(scale_factor[j],mah_list[i,j]/mah_list[i,-1],c='r')
                
            #     elif dm == 'cdm' and sn=='sn_010':
            #         properties_file.writelines(f'\nCDM SN 10%\n')
            #         for i in range(mah_list.shape[0]):
            #             properties_file.writelines(f'\nSimulation = {folder_list[i]}\n\n')
            #             for j in range(mah_list.shape[1]):
            #                 if j !=0 :
            #                     if ((mah_list[i,j]/mah_list[i,j-1]*100)-100) > 30 : 
            #                         properties_file.writelines(f'MAJOR MERGER: z = {redshift[j]}, mass ratio = {np.round(((mah_list[i,j]/mah_list[i,j-1]*100)-100),4)}%\n')
            #                         plt.scatter(scale_factor[j],mah_list[i,j]/mah_list[i,-1],c='r')
                
            #     elif dm == 'wdm_3.5' and sn=='sn_005':
            #         properties_file.writelines(f'\nWDM SN 5%\n')
            #         for i in range(mah_list.shape[0]):
            #             properties_file.writelines(f'\nSimulation = {folder_list[i]}\n\n')
            #             for j in range(mah_list.shape[1]):
            #                 if j !=0 :
            #                     if ((mah_list[i,j]/mah_list[i,j-1]*100)-100) > 30 : 
            #                         properties_file.writelines(f'MAJOR MERGER: z = {redshift[j]}, mass ratio = {np.round(((mah_list[i,j]/mah_list[i,j-1]*100)-100),4)}%\n')
            #                         plt.scatter(scale_factor[j],mah_list[i,j]/mah_list[i,-1],c='r')
                
            #     elif dm == 'wdm_3.5' and sn=='sn_010':
            #         properties_file.writelines(f'\nWDM SN 10%\n')
            #         for i in range(mah_list.shape[0]):
            #             properties_file.writelines(f'\nSimulation = {folder_list[i]}\n\n')
            #             for j in range(mah_list.shape[1]):
            #                 if j !=0 :
            #                     if ((mah_list[i,j]/mah_list[i,j-1]*100)-100) > 30 : 
            #                         properties_file.writelines(f'MAJOR MERGER: z = {redshift[j]}, mass ratio = {np.round(((mah_list[i,j]/mah_list[i,j-1]*100)-100),4)}%\n')
            #                         plt.scatter(scale_factor[j],mah_list[i,j]/mah_list[i,-1],c='r')
            
            plt.xlabel('Redshift')
            # plt.xscale('log')
            # plt.yscale('log')
            plt.gca().invert_xaxis()

            # plt.ylabel(r'Halo Mass [x10$^{10}$ $M_{\odot}h^{-1}$]')
            plt.ylabel(r'$M_{200c}$/$M_{200c, z=0}$')
            # plt.axhline(0.5, c='k', ls='--', alpha=0.5)
            # plt.title('Mass Accretion History')
            # plt.legend()
            plt.tight_layout()
            plt.savefig(f'./combined_plots/good_Mass_accretion_history_{dm}_{sn}.png',dpi=400)
            plt.close()
## Plot all MAH's
#####

    # target_mass, target_pos = check_low_res(low_res_sim_directory,i_file)
    # compare_tree_and_plot(low_res_sim_directory,target_mass, target_pos, "Coarse",fig,ax,'k',':')


    # compare_tree_and_plot(cdm_sn_010,cdm_sn_010_mass, cdm_sn_010_pos, "CDM",fig,ax,'blue')
    # compare_tree_and_plot(wdm_sn_005,wdm_sn_005_mass, wdm_sn_005_pos, "WDM",fig,ax,'orange')#,ls='-.'
