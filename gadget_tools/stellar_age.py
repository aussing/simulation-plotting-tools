import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import astropy
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck13
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from scipy.stats import bootstrap


plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 14

LITTLEH    = 1#0.703
UNITMASS   = 1.0e10
GASTYPE    = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5


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

def SFR_hist(sim_directory,label,folder,cosmo,i_file):

    print(f'\nStellar age for {label}')
    snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
    # print(sim_directory)
    halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH
    group_len_type = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.int32)
    group_offset_type = np.array(haloinfo_data['Group']['GroupOffsetType'], dtype=np.int32)

    halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) 
    halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64)
    halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    group_len = np.array(haloinfo_data['Group']['GroupLen'], dtype=np.int32)
    

    mass_mask = np.argsort(halo_mass)[::-1]
    if folder == 'N2048_L65_sd_MW_ANDR':
        halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][1]
    else:
        halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]

    print(halo_R200c[mass_mask[halo_mainID]])
    print(halo_pos[mass_mask[halo_mainID]])
    
    star_len =  group_len_type[mass_mask[halo_mainID],STARTYPE]
    star_end = star_len-1
    print(group_offset_type[mass_mask[halo_mainID],STARTYPE],group_offset_type[mass_mask[halo_mainID],STARTYPE]+star_end)
    stellar_mass =  np.array(snap_data[f'PartType{STARTYPE}']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH
    
    stellar_mass = stellar_mass[group_offset_type[mass_mask[halo_mainID],STARTYPE]:group_offset_type[mass_mask[halo_mainID],STARTYPE]+star_end]
    
    stellar_part_mass = np.unique(stellar_mass) ## should be a single value
    print(f'Star particle mass = {stellar_part_mass[:]} Msun')
    star_form_time_a = np.array(snap_data[f'PartType{STARTYPE}']['StellarFormationTime'], dtype=np.float64)
    star_form_time_a = star_form_time_a[group_offset_type[mass_mask[halo_mainID],STARTYPE]:group_offset_type[mass_mask[halo_mainID],STARTYPE]+star_end]
    star_form_time_z = 1/(star_form_time_a)-1

    star_form_time_age = cosmo.age(star_form_time_z).value

    print(f"First star formed at z={np.round(np.max(star_form_time_z),2)}, Last star formed at z={np.round(np.min(star_form_time_z),2)}")
    print(f"First star formed at T={np.round(np.min(star_form_time_age),2)}Gyr, Last star formed at T={np.round(np.max(star_form_time_age),2)}Gyr")

    # bin_width = 0.2
    # num_of_bins = np.int32((np.max(star_form_time_age+1e-4) - np.min(star_form_time_age-1e-4))/bin_width)
    # print(f"bin number = {num_of_bins}")
    num_of_bins = 100
    bin_width = (np.max(star_form_time_age+1e-4) - np.min(star_form_time_age-1e-4))/num_of_bins
    print(f"bin width = {np.round(bin_width,2)} Gyr")
    bin_range = np.linspace(np.min(star_form_time_age-1e-4),np.max(star_form_time_age+1e-4),num_of_bins)
    
    sfr, bins_func = np.histogram(star_form_time_age,bin_range)
    

    for i in range(num_of_bins-1): #Change from bin edges to bin centres, remove last value in bins_func when plotting
        bins_func[i]=bins_func[i]+(bins_func[i+1]-bins_func[i])/2
    bins_func=bins_func[:-1]
    bin_width = bin_width*1e9
    
    sfr = (sfr*stellar_part_mass)/bin_width
    # print(cosmo.age(0))
    # print(np.min(star_form_time_age))

    # plt.hist(star_form_time_age,bins=50,histtype='stepfilled',label=label,alpha=0.6)
    plt.plot(bins_func,sfr,label=label)
    plt.xlabel('Formation time [Gyr]')
    plt.ylabel('SFR [Msun/yr]')
    plt.yscale('log')
    plt.xlim((0,14))
    # ax = plt.gca
    # ax.secondary_xaxis('top',star_form_time_z)
    plt.legend()
    plt.savefig(f'{folder}/SFR_history.png',dpi=400)
    return bins_func,sfr


def plot_dif(bins,SFR_hist_cdm,SFR_hist,label,folder):
    print(f"plotting difference for {label}")
    diff = SFR_hist/SFR_hist_cdm-1
    # plt.tight_layout()
    plt.plot(bins,diff*100,label=f"{label}/CDM")
    plt.axhline(0,0,14,ls='--',color='k',alpha=0.5)
    plt.xlabel("Formation time [Gyr]")
    plt.ylabel("% Difference to CDM")
    plt.ylim((-150,600))
    plt.legend()
    plt.savefig(f'{folder}/SFR_history_diff.png',dpi=400)

def group_sfr(sim_directory,label,folder, fig, ax, alpha, color):
    print('-- ',label,' --\n')
    redshift_list = np.linspace(0,26,27,dtype=int)
    # redshift_list = np.linspace(0,45,46,dtype=int)
    main_halo_sfr = []
    z_list = []
    a_list = []
    age = []
    for redshift in redshift_list:
        try:
            snap_data, haloinfo_data, z = get_sim_data(sim_directory,redshift)
            halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) #* LITTLEH
            halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
            halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH 
            mass_mask = np.argsort(halo_mass)[::-1]
            if folder == 'N2048_L65_sd_MW_ANDR':
                halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][1]
            else:
                halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
            
            # print()
            # print(np.round(z,2),redshift)
            # print(f"main halo ID = {mass_mask[halo_mainID]}")
            # print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID]]}")
            # print(f"Main halo mass = {halo_mass[mass_mask[halo_mainID]]/1e10}")
            # print()
            halo_sfr = np.array(haloinfo_data['Group']['GroupSFR'], dtype=np.float64)
            # print(f"Group SFR = {halo_sfr[mass_mask[halo_mainID]]} Msun/Year, z={np.round(z,2)}")
            main_halo_sfr.append(halo_sfr[mass_mask[halo_mainID]])
            age.append(cosmo.age(z).value)
            z_list.append(np.round(z,2))
            a_list.append(np.round(snap_data['Header'].attrs['Time'],2))
        except:
            snap_data, haloinfo_data, z = get_sim_data(sim_directory,redshift)
            main_halo_sfr.append(0)
            z_list.append(np.round(z,2))
            age.append(cosmo.age(z).value)
            a_list.append(np.round(snap_data['Header'].attrs['Time'],2))
            # print(f"No halo at z={z}")
    print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID]]}")
    # print(label)
    print(f"Group SFR = {np.round(halo_sfr[mass_mask[halo_mainID]],5)} Msun/Year, z={np.round(z,2)}\n")
    
    # fig, ax = plt.subplots()
    # plt.plot(1/(1+np.array(z_list)),main_halo_sfr)
    # plt.plot(a_list,main_halo_sfr)
    # ax.plot(age,main_halo_sfr,label=label, alpha=0.3, c=color)

    ax.plot(z_list,main_halo_sfr,label=label, alpha=alpha, c=color)

    # plt.xticks(np.array(a_list)[::4],np.array(z_list)[::4])
    # # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # # ax.xaxis.set_minor_locator(FixedLocator(np.array(a_list)))
    plt.yscale('log')
    # plt.xlabel('Formation time [Gyr]')
    # plt.ylabel('SFR [Msun/yr]')
    # # plt.xscale('log')
    # plt.legend()
    # plt.savefig(f'{folder}/group_sfr.png',dpi=400)
    return a_list, z_list, main_halo_sfr, age
    
if __name__ == "__main__":

    cosmo = FlatLambdaCDM(H0=68.8,  Om0= 0.321, Ob0=0.045, Tcmb0=2.725) 
    i_file     = 26
    # folder_list = ['N2048_L65_sd17492']#, 'N2048_L65_sd28504', 'N2048_L65_sd34920', 'N2048_L65_sd46371', 'N2048_L65_sd57839', 'N2048_L65_sd61284', 'N2048_L65_sd70562', 'N2048_L65_sd80325', 'N2048_L65_sd93745',]
    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if ( simulation_name.startswith("N2048_L65_sd") )]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    good_list_dust_mass = ['N2048_L65_sd00372' , 'N2048_L65_sd03157','N2048_L65_sd01829','N2048_L65_sd05839' 'N2048_L65_sd17492','N2048_L65_sd28504', 'N2048_L65_sd46371', 'N2048_L65_sd61284', 'N2048_L65_sd70562']#
    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(25, 10))
    fig, (ax2) = plt.subplots()
    
    sfr_list_cdm = []
    sfr_list_wdm = []
    ages = []
    all_z_list = []
    all_a_list = []

    for folder in folder_list:
        # fig, ax = plt.subplots()
        print(folder)
        cdm_sn_005 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_005/'
        cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'

        wdm_sn_005 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
        wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'

        alpha=0.3
        # if folder in good_list_dust_mass:
        #     alpha=1
        # else :
        #     alpha=0.3

        # a_list, z_list, sfr_hist_cdm_005 = group_sfr(cdm_sn_005,'cdm_sn_005',folder)
        a_list, z_list, sfr_hist_cdm_010, age = group_sfr(cdm_sn_010,'cdm_sn_010',folder,fig,ax2,alpha,color='blue')
        a_list, z_list, sfr_hist_wdm_005, age = group_sfr(wdm_sn_005,'wdm_sn_005',folder,fig,ax2, alpha, color='orange')
        # a_list, z_list, sfr_hist_wdm_010 = group_sfr(wdm_sn_010,'wdm_sn_010',folder)
        sfr_list_cdm.append(sfr_hist_cdm_010)
        sfr_list_wdm.append(sfr_hist_wdm_005)
        ages.append(age)
        all_z_list.append(z_list)
        all_a_list.append(a_list)

        # plt.xticks(np.array(a_list)[::4],np.array(z_list)[::4])
        # # plt.gca().invert_xaxis()
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.xaxis.set_minor_locator(FixedLocator(np.array(a_list)))
        # plt.yscale('log')
        # plt.xlabel('Formation time [Gyr]')
        # plt.ylabel('SFR [Msun/yr]')
        # plt.xscale('log')
        # plt.legend()
        # plt.savefig(f'{folder}/group_sfr.png',dpi=400)
        # print(f'{folder}/group_sfr.png')

    all_z_list = np.array(all_z_list)[0,:]
    all_a_list = np.array(all_a_list)[0,:]
    
    # ax1.set_xticks(np.array(all_a_list)[::4],np.array(all_z_list)[::4])
    # ax1.xaxis.set_minor_locator(AutoMinorLocator())
    # ax1.xaxis.set_minor_locator(FixedLocator(np.array(all_a_list)))

    # ax2.set_xticks(np.array(all_a_list)[2::4],np.array(all_z_list)[2::4])
    # ax2.xaxis.set_minor_locator(AutoMinorLocator())
    # ax2.xaxis.set_minor_locator(FixedLocator(np.array(all_a_list)))
    
    # ax1.set_ylim(1e-4,7e2)
    # ax2.set_ylim(1e-4,7e2)
    
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')

    # ax1.set_xlabel('Formation time [Gyr]')
    # ax1.set_xlabel('Redshift')
    ax2.set_xlabel('Redshift')
    # ax2.set_xlabel('Formation time [Gyr]')

    # ax1.set_ylabel(r'SFR [M$_{\odot}$/yr]')
    ax2.set_ylabel(r'SFR [M$_{\odot}$/yr]')
    


    # cdm_btstrp_low = []
    # cdm_btstrp_high = []
    # for i in range(np.array(sfr_list_cdm).shape[1]):
    #     data = (np.array(sfr_list_cdm)[:,i],)
    #     btstrp_data = bootstrap(data,np.median,confidence_level=0.99,random_state=1,method='percentile')
    #     cdm_btstrp_low.append(btstrp_data.confidence_interval[0])
    #     cdm_btstrp_high.append(btstrp_data.confidence_interval[1])

    # wdm_btstrp_low = []
    # wdm_btstrp_high = []
    # for i in range(np.array(sfr_list_wdm).shape[1]):
    #     data = (np.array(sfr_list_wdm)[:,i],)
    #     btstrp_data = bootstrap(data,np.median,confidence_level=0.99,random_state=1,method='percentile')
    #     wdm_btstrp_low.append(btstrp_data.confidence_interval[0])
    #     wdm_btstrp_high.append(btstrp_data.confidence_interval[1])

    cdm_mean_sfr = np.median(np.array(sfr_list_cdm),axis=0)
    wdm_mean_sfr = np.median(np.array(sfr_list_wdm),axis=0)

    # ax1.plot(np.array(ages).T,cdm_mean_sfr,c='k')
    # ax2.plot(np.array(ages).T,wdm_mean_sfr,c='k')all_z_list

    # ax2.plot(all_a_list,cdm_mean_sfr,c='k')
    # ax2.plot(all_z_list,cdm_mean_sfr,c='k')
    # ax2.fill_between(all_a_list,cdm_btstrp_low,cdm_btstrp_high,color='gray',alpha=0.3)
    
    ax2.plot(all_a_list,wdm_mean_sfr,c='k')
    # ax2.fill_between(all_a_list,wdm_btstrp_low,wdm_btstrp_high,color='gray',alpha=0.3)
    
    # ax2.plot(all_a_list,(np.divide(wdm_mean_sfr,cdm_mean_sfr)-1)*100,c='orange')
    # ax2.plot(all_a_list,(np.divide(cdm_mean_sfr,cdm_mean_sfr)-1)*100,c='blue',ls='--')
    
    
    
    # plt.gca().invert_xaxis()
    plt.xlim(0,4)
    plt.tight_layout()
    plt.savefig(f'combined_plots/SFR_test_wdm.png',dpi=300)
        
    plt.close()

    # for folder in folder_list:
    #     cdm_sn_005 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_005/'
    #     cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'

    #     wdm_sn_005 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
    #     wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'

    #     bins_cdm_05,SFR_hist_cdm_05 = SFR_hist(cdm_sn_005,"CDM SN=5%",folder,cosmo,i_file)
    #     bins_cdm_10,SFR_hist_cdm_10 = SFR_hist(cdm_sn_010,"CDM SN=10%",folder,cosmo,i_file)
        
    #     bins_wdm_05,SFR_hist_wdm_05 = SFR_hist(wdm_sn_005,"WDM SN=5%",folder,cosmo,i_file)
    #     bins_wdm_10,SFR_hist_wdm_10 = SFR_hist(wdm_sn_010,"WDM SN=10%",folder,cosmo,i_file)
        
    #     # plt.xticks(np.array(a_list)[::4],np.array(z_list)[::4])
    #     # plt.gca().invert_xaxis()
    #     # ax.xaxis.set_minor_locator(AutoMinorLocator())
    #     # ax.xaxis.set_minor_locator(FixedLocator(np.array(a_list)))
    #     plt.yscale('log')
    #     plt.xlabel('Formation time [Gyr]')
    #     plt.ylabel('SFR [Msun/yr]')
    #     # plt.xscale('log')
    #     # plt.legend()
    #     plt.savefig(f'{folder}/SFR_history.png',dpi=400)
    #     print(f'{folder}/SFR_history.png')
    #     plt.close()
    #     # plt.close()

    # # bins_wdm_15,SFR_hist_wdm_15 = SFR_hist(wdm_sn_015,"WDM SN=15%",folder,cosmo,i_file)
    # # bins_cdm_20,SFR_hist_cdm_20 = SFR_hist(cdm_sn_020,"CDM SN=20%",folder,cosmo,i_file)
    # plt.close()
    # plot_dif(bins_cdm_05,SFR_hist_cdm_10,SFR_hist_cdm_05,"CDM SN=5%",folder)
    # plot_dif(bins_wdm_05,SFR_hist_cdm_10,SFR_hist_wdm_05,"WDM SN=5%",folder)
    # plot_dif(bins_wdm_10,SFR_hist_cdm_10,SFR_hist_wdm_10,"WDM SN=10%",folder)


    # plot_dif(bins_cdm_20,SFR_hist_cdm_10,SFR_hist_cdm_20,"SN=20%",folder)
    # stellar_age(wdm_sn_005,"WDM SN=5\%",folder,cosmo)
    # stellar_age(wdm_sn_010,"WDM SN=10\%",folder,cosmo)
    # stellar_age(wdm_sn_015,"WDM SN=15\%",folder,cosmo)
    # stellar_age(wdm_sn_020,"WDM SN=20\%",folder,cosmo)

    # plot_dif(bins_2k,SFR_hist_cdm,SFR_hist_2k,"2KeV",folder)
    # plot_dif(bins_3k,SFR_hist_cdm,SFR_hist_3k,"3.5KeV",folder)
    # plot_dif(bins_7k,SFR_hist_cdm,SFR_hist_7k,"7KeV",folder)
    # plot_dif(bins_11k,SFR_hist_cdm,SFR_hist_11k,"11KeV",folder)