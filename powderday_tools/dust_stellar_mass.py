import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import inf
from numpy import nan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from tqdm import trange
import astropy.units as u
import h5py
import os
import cmasher as cmr
import fileinput
import os
import shutil

plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['font.size'] = 11

LITTLEH = 0.6688
UNITMASS = 1e10
SOLMASSINGRAMS = 1.989e+33

UNIT_LENGTH_FOR_PLOTS = 'kpc'

def get_sim_data(folder,dm,SN_fac,n_file):
    snap_fname     = f'/snapshot_{str(n_file).zfill(3)}.hdf5'
    snap_directory = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/' + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(n_file).zfill(3)}.hdf5'
    haloinfo_directory = f'/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/' + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = np.round(snap_data['Header'].attrs['Redshift'],2)
    return snap_data, haloinfo_data, z

def get_unit_len(snapshot):
    unit_length = snapshot["Parameters"].attrs['UnitLength_in_cm']
    
    return unit_length

def set_plot_len(data, unit=UNIT_LENGTH_FOR_PLOTS):
    if unit in ['Mpc','mpc']:
        data = data/3.085678e24
    elif unit in ['Kpc','kpc']:
        data = data/3.085678e21
    elif unit == 'pc':
        data = data/3.085678e18
    else:
        print("What units do you want?????!!! AARRHH")
        raise TypeError
    return data

def get_snapshot(folder,snap_num,dm='cdm',SN_fac='sn_010'):

    snapshot = h5py.File(f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/snapshot_{str(snap_num).zfill(3)}.hdf5")
    
    return snapshot

def load_grid_data(folder,method,dm,SN_fac,snap_num):
    data = np.load(f'./{folder}/{method}/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')

    return data

def set_center(folder,dm,SN_fac,snap_num):
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')

    halo_pos   = np.array(halo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH
    halo_M200c = np.array(halo_data['Group']['Group_M_Crit200'], dtype=np.float64) / LITTLEH * UNITMASS
    halo_masstypes = np.array(halo_data['Group']['GroupMassType'], dtype=np.float64) / LITTLEH  * UNITMASS
    R200c = np.array(halo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
    Group_SFR = np.array(halo_data['Group']['GroupSFR'], dtype=np.float64) 
    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0] #Select largest non-contaminated halo / resimulation target
    
    # print(f'halo mass = {halo_M200c[mass_mask][halo_mainID]/UNITMASS*LITTLEH}e10')
    snapshot = get_snapshot(folder,snap_num,dm,SN_fac)
    unit_length = get_unit_len(snapshot)

    halo_pos = np.array(halo_pos[mass_mask][halo_mainID]) * unit_length
    halo_rad = R200c[mass_mask][halo_mainID] * unit_length
    stellar_mass = halo_masstypes[mass_mask][halo_mainID][4]
    sfr = Group_SFR[mass_mask][halo_mainID] 
    print(f"stellar mass = {stellar_mass:.3e} Msun")
    print(f"SFR          = {np.round(sfr,4)} Msun/yr")
    # print(f"Halo Rad = {halo_rad}, halo pos = {halo_pos}\n")
    # extent = halo_rad*2
    extent = halo_rad


    return halo_pos,extent, stellar_mass, sfr

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

def dust_mass(dm,SN_fac,label,folder,method,num_bins,snap_num):
    print(dm,'-',SN_fac)
    
    halo_pos, extent, stellar_mass, sfr = set_center(folder,dm,SN_fac,snap_num)
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')
    a = halo_data['Header'].attrs['Time']
    z = halo_data['Header'].attrs['Redshift']
    # print(halo_pos)
    # method = out_folder.split('/')[-2]

    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)
    
    pos_x = grid_physical_properties_data['gas_pos_x'] * 3.085678e18 / a# convert parsecs to cm, convert back later
    pos_y = grid_physical_properties_data['gas_pos_y'] * 3.085678e18 / a
    pos_z = grid_physical_properties_data['gas_pos_z'] * 3.085678e18 / a
    # print(np.min(grid_physical_properties_data['gas_pos_x'])/1e6*LITTLEH, np.max(grid_physical_properties_data['gas_pos_x'])/1e6*LITTLEH)
    # print((np.max(grid_physical_properties_data['gas_pos_x']) - np.min(grid_physical_properties_data['gas_pos_x']))/2e6*LITTLEH)
    extent = np.float32(extent)
    extent_r200 = extent

    stellar_mass_array = np.geomspace(1e9,5e11,50)

    # M_0 = 3.98e10
    # alpha, beta, gamma = 0.14, 0.39, 0.10 
    # R_bar_late = gamma * (stellar_mass)**alpha * (1+stellar_mass/M_0)**(beta-alpha)
    # R_bar_late_array = gamma * (stellar_mass_array)**alpha * (1+stellar_mass_array/M_0)**(beta-alpha)

    # print(f'R_bar_late        = {R_bar_late} kpc')
    # print(f'extent/R_bar_late = {set_plot_len(extent)/R_bar_late} \n')

    # a, b = 0.56, 2.88e-6
    # R_bar_early = b * (stellar_mass)**a
    # R_bar_early_array = b * (stellar_mass_array)**a
    # print(f'R_bar_early        = {R_bar_early} kpc')
    # print(f'extent/R_bar_early = {set_plot_len(extent)/R_bar_early} \n')

    # fig,ax = plt.subplots()
    # plt.loglog(stellar_mass_array, R_bar_early_array, label='Early')
    # plt.loglog(stellar_mass_array, R_bar_late_array, label='Late')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./R_50_profiles.png',dpi=250)
    # plt.close()
    
    
    # extent = extent * 1.0

    pos_mask = make_mask(pos_x,pos_y,pos_z,halo_pos,extent)
    pos_mask_rad200 = make_mask(pos_x,pos_y,pos_z,halo_pos,extent_r200)

    extent_25kpc = 25 * 3.085678e21
    pos_mask_25kpc = make_mask(pos_x,pos_y,pos_z,halo_pos,extent_25kpc)
    
    extent_15kpc = 15 * 3.085678e21
    pos_mask_15kpc = make_mask(pos_x,pos_y,pos_z,halo_pos,extent_15kpc)
    
    gas_mass         = grid_physical_properties_data['particle_gas_mass'][pos_mask_rad200] / SOLMASSINGRAMS
    gas_mass_15kpc   = grid_physical_properties_data['particle_gas_mass'][pos_mask_15kpc] / SOLMASSINGRAMS
    # gas_mass         = grid_physical_properties_data['particle_gas_mass'][pos_mask_rad200] / SOLMASSINGRAMS

    dust_mass        = grid_physical_properties_data['particle_dustmass']#[pos_mask]
    dust_mass_rad200 = grid_physical_properties_data['particle_dustmass'][pos_mask_rad200]
    dust_mass_25kpc  = grid_physical_properties_data['particle_dustmass'][pos_mask_25kpc]
    dust_mass_15kpc  = grid_physical_properties_data['particle_dustmass'][pos_mask_15kpc]
    
    
    return dust_mass, np.sum(dust_mass_rad200), stellar_mass, sfr


if __name__ =='__main__':

    # out_folder = f'plots/{folder}/dtm/'
    # out_folder = f'plots/{folder}/rr/'
    # out_folder = f'plots/{folder}/li_bf/'
    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if (simulation_name.startswith("N2048_L65_sd"))]: #and not simulation_name.startswith("N2048_L65_sd0"))]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(folder_list)

    sn_list = ['sn_005','sn_010']
    dm_list = ['cdm','wdm_3.5']
    
            
    mean_dust_mass      = []
    method_list = ['dtm','rr','li_bf']
    cdm_sn = 'sn_010'
    wdm_sn = 'sn_005'
    for method in method_list:     
        # for sn in sn_list:
            # for dm in dm_list :          
        cdm_stel_dust_mass, cdm_sfr_dust_mass = [], []
        wdm_stel_dust_mass, wdm_sfr_dust_mass = [], []
        fig=plt.subplots()
        for folder in folder_list:
            print()
            print(f'Simulation -> {folder}')
            
            snap_num = 26
            num_bins = 80

            cdm_dust_mass_total, cdm_dust_mass_rad200, cdm_stellar_mass, cdm_sfr = dust_mass('cdm',cdm_sn,'',folder,method,num_bins,snap_num)
            wdm_dust_mass_total, wdm_dust_mass_rad200, wdm_stellar_mass, wdm_sfr = dust_mass('wdm_3.5',wdm_sn,'',folder,method,num_bins,snap_num)

            cdm_stel_dust_mass.append((cdm_stellar_mass,cdm_dust_mass_rad200))
            cdm_sfr_dust_mass.append((cdm_sfr,cdm_dust_mass_rad200))
            
            wdm_stel_dust_mass.append((wdm_stellar_mass,wdm_dust_mass_rad200))
            wdm_sfr_dust_mass.append((wdm_sfr,wdm_dust_mass_rad200))


        cdm_stel_dust_mass = np.array(cdm_stel_dust_mass)
        wdm_stel_dust_mass = np.array(wdm_stel_dust_mass)
        
        
        plt.scatter(np.log10(cdm_stel_dust_mass[:,0]),np.log10(cdm_stel_dust_mass[:,1]),c='blue', label='CDM SN=10\%')
        plt.scatter(np.log10(wdm_stel_dust_mass[:,0]),np.log10(wdm_stel_dust_mass[:,1]),c='orange', label='WDM SN=5\%')
        
        log_stellar_mass = np.geomspace(10.5,11.5,50)+0.23                
        peeples_14_fit = 0.86*(log_stellar_mass) - 1.31
        skibba_11_bar_fit  = log_stellar_mass - 2.84
        skibba_11_nobar_fit  = log_stellar_mass - 3.06
        plt.plot(log_stellar_mass,peeples_14_fit,c='k',ls='--',label='Peeples+14')
        plt.plot(log_stellar_mass,skibba_11_bar_fit,c='k',ls=':',label='Skibba+11 barred')
        plt.plot(log_stellar_mass,skibba_11_nobar_fit,c='k',ls='-.',label='Skibba+11 no bar')
        plt.xlabel(r'M$_{*}$ [M$_{\odot}$]')
        plt.ylabel(r'M$_{dust}$ [M$_{\odot}$]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./plots/post_sub_tests/stellar_dust_mass_{method}.png',dpi=250)
        
        cdm_sfr_dust_mass  = np.array(cdm_sfr_dust_mass)
        wdm_sfr_dust_mass  = np.array(wdm_sfr_dust_mass)

        fig=plt.subplots()
        sfr_space = np.logspace(-1,0.8)
        da_cunha_10_dust_mass = 1.28e7 * sfr_space**1.11
        plt.scatter(np.log10(cdm_sfr_dust_mass[:,0]),(cdm_sfr_dust_mass[:,1]),c='blue', label="CDM SN=10\%")
        plt.scatter(np.log10(wdm_sfr_dust_mass[:,0]),(wdm_sfr_dust_mass[:,1]),c='orange', label='WDM SN=5\%')
        plt.plot(sfr_space,da_cunha_10_dust_mass,c='k',ls='--', label='Da Cunha+10')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("SFR")
        plt.ylabel("Dust Mass")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./plots/post_sub_tests/SFR_dust_mass_{method}.png',dpi=250)                

            # # plt.xscale('log')
            # plt.yscale('log')
            # plt.savefig(f'stellar_dust_mass_{dm}_{sn}.png',dpi=300)
            


                    # dust_mass_total, dust_mass_rad200 = dust_mass(dm,sn,'CDM SN=10\%',folder,out_folder,num_bins,snap_num)
                    
            
                    # dust_mass_total, dust_mass_rad200 = dust_mass(dm,sn,'WDM SN=5\%',folder,out_folder,num_bins,snap_num)
                    # dust_mass_total, dust_mass_rad200 = dust_mass(dm,sn,'WDM SN=10\%',folder,out_folder,num_bins,snap_num)
                        
                    
                        