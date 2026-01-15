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

plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['font.size'] = 11

LITTLEH = 0.6688
UNITMASS = 1e10
SOLMASSINGRAMS = 1.989e+33

UNIT_LENGTH_FOR_PLOTS = 'pc'

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

def check_converge(folder,dm,SN_fac,snap_num):
    snapshot = get_snapshot(folder,snap_num,dm,SN_fac)
    unit_len = get_unit_len(snapshot)
    halo_pos,extent = set_center(folder,dm,SN_fac,snap_num)
    extent = set_plot_len(extent/2,'kpc')
    halo_pos = halo_pos * LITTLEH
    # print(set_plot_len(halo_pos),set_plot_len(extent))
    r_conv_test = np.geomspace(0.1,extent,50,dtype=np.float64) * 3.085678e21 # Kpc

    gas_part_coords  = np.array(snapshot['PartType0']['Coordinates'],dtype=np.float64) * unit_len
    gas_part_mass    = np.array(snapshot['PartType0']['Masses'],dtype=np.float64) * UNITMASS

    dm_part_coords   = np.array(snapshot['PartType1']['Coordinates'],dtype=np.float64) * unit_len
    dm_mass          = snapshot["Header"].attrs['MassTable'][1] * UNITMASS 

    star_part_coords = np.array(snapshot['PartType4']['Coordinates'],dtype=np.float64) * unit_len
    star_part_mass   = np.array(snapshot['PartType4']['Masses'],dtype=np.float64) * UNITMASS

    kappa_r = [] 
    crit_dens = 2.7754e11 # h^2 M_sol Mpc^-3
    for i in trange(len(r_conv_test)):
        gas_pos_mask  = make_mask(gas_part_coords[:,0],  gas_part_coords[:,1],  gas_part_coords[:,2], halo_pos, r_conv_test[i])
        dm_pos_mask   = make_mask(dm_part_coords[:,0],   dm_part_coords[:,1],   dm_part_coords[:,2],  halo_pos, r_conv_test[i])
        star_pos_mask = make_mask(star_part_coords[:,0], star_part_coords[:,1], star_part_coords[:,2],halo_pos, r_conv_test[i])

        total_part = np.sum(gas_pos_mask) + np.sum(dm_pos_mask) + np.sum(star_pos_mask) 
        # print(f'total particles for radius = {np.round(set_plot_len(test_rad),2)} = {total_part}')

        volume = 4/3 * np.pi * (set_plot_len(r_conv_test[i],'mpc'))**3
        total_mass = np.sum(gas_part_mass[gas_pos_mask]) + np.sum(dm_pos_mask)*dm_mass + np.sum(star_part_mass[star_pos_mask])
        # print(f'total mass for radius = {np.round(set_plot_len(test_rad),2)} = {total_mass/UNITMASS}e10 M_sol\n')

        density = total_mass/volume
        kappa = np.sqrt(200) / 8 * total_part / np.log(total_part) * (density/crit_dens)**(-1/2) # Power et al 2003 eq 20 relaxtion 
        kappa_r.append(kappa)
    kappa_r = np.array(kappa_r)
    converg_rad = r_conv_test[np.abs(kappa_r-1).argmin()]
    print(f"convergence radius = {np.round(set_plot_len(converg_rad),4)} {UNIT_LENGTH_FOR_PLOTS}")

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

    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0] #Select largest non-contaminated halo / resimulation target
    
    # print(f'halo mass = {halo_M200c[mass_mask][halo_mainID]/UNITMASS*LITTLEH}e10')
    snapshot = get_snapshot(folder,snap_num,dm,SN_fac)
    unit_length = get_unit_len(snapshot)

    halo_pos = np.array(halo_pos[mass_mask][halo_mainID]) * unit_length
    halo_rad = R200c[mass_mask][halo_mainID] * unit_length
    
    # print(f"Halo Rad = {halo_rad}, halo pos = {halo_pos}\n")
    # extent = halo_rad*2
    extent = halo_rad

    return halo_pos,extent

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

def obs_surface_dens():
    data = np.loadtxt('./M31_dust_profile.txt',delimiter=',')
    rad = data[:,0]
    surface_dens = data[:,1]/(1e3)**2 
    # plt.plot(rad[1:],surface_dens[1:],'o',c='r',label='Draine+ (2014)')
    plt.plot(rad[2:],surface_dens[2:],'o',c='r',label='Draine+ (2014)', linewidth=7)

def calc_rad_dust(grid_physical_properties_data,halo_pos,extent,num_bins=[]):
    
    pos_x = grid_physical_properties_data['gas_pos_x']  * 3.085678e18 # convert parsecs to cm, convert back later
    pos_y = grid_physical_properties_data['gas_pos_y']  * 3.085678e18
    pos_z = grid_physical_properties_data['gas_pos_z']  * 3.085678e18
    
    # print("extent = ",extent)
    # extent = 25 * 3.085678e21 ## turn R200 into 25kpc
    
    pos_mask = make_mask(pos_x,pos_y,pos_z,halo_pos,extent)

    dust_mass = grid_physical_properties_data['particle_dustmass'][pos_mask] #in units Msun

    rad_dist_x = np.abs(pos_x[pos_mask]-halo_pos[0])
    rad_dist_y = np.abs(pos_y[pos_mask]-halo_pos[1])
    rad_dist_z = np.abs(pos_z[pos_mask]-halo_pos[2])

    rad_mod = np.sqrt(rad_dist_x**2+rad_dist_y**2+rad_dist_z**2)

    # bin_width = 5 * 3.085678e21 # 5 Kpc to cm
    bin_width = extent/50
    
    if num_bins==[]:
        num_bins = int((extent+0.05*extent)/bin_width) # width of extent plus an extra 5% to cover the edges
    else: 
        num_bins=num_bins
    bins_rad = np.linspace(bin_width,extent+0.05*extent,num_bins)
    dig = np.digitize(rad_mod,bins_rad,right=True)
    
    mass_in_bin = []
    density = []
    for ibin in range(len(bins_rad)):
        mass_in_bin.append(np.sum(dust_mass[dig == (ibin) ]))
        
        if ibin == 0:
            volume = 4/3 * np.pi * (bins_rad[ibin])**3
            # print(bins_rad[ibin])
            # print(set_plot_len(bins_rad[ibin]))
        else :
            volume = 4/3 * np.pi * (bins_rad[ibin]-bins_rad[ibin-1])**3
            # print(bins_rad[ibin]-bins_rad[ibin-1])
            # print(set_plot_len(bins_rad[ibin]-bins_rad[ibin-1]))
        
        density.append(mass_in_bin[ibin]/set_plot_len(set_plot_len(set_plot_len(volume))))
    # print(np.max(bins_rad))    
    # print(bins_rad/(extent/2))
    return np.array(density), bins_rad

def calc_rad_dust_surf_dens(grid_physical_properties_data,halo_pos,extent,num_bins=[]):
    
    pos_x = grid_physical_properties_data['gas_pos_x']  * 3.085678e18 # convert parsecs to cm, convert back later
    pos_y = grid_physical_properties_data['gas_pos_y']  * 3.085678e18
    pos_z = grid_physical_properties_data['gas_pos_z']  * 3.085678e18
    
    # print("extent = ",extent)
    # extent = 25000 * 3.085678e21 ## turn R200 into 25kpc
    
    pos_mask = make_mask(pos_x,pos_y,pos_z,halo_pos,extent)

    dust_mass = grid_physical_properties_data['particle_dustmass'][pos_mask] #in units Msun

    rad_dist_x = np.abs(pos_x[pos_mask]-halo_pos[0])
    rad_dist_y = np.abs(pos_y[pos_mask]-halo_pos[1])
    rad_dist_z = np.abs(pos_z[pos_mask]-halo_pos[2])

    rad_mod = np.sqrt(rad_dist_x**2+rad_dist_y**2+rad_dist_z**2)

    # bin_width = 5 * 3.085678e21 # 5 Kpc to cm
    # bin_width = extent/50
    
    if num_bins==[]:
        bin_width = extent/50
        num_bins = int((extent+0.05*extent)/bin_width) # width of extent plus an extra 5% to cover the edges
    else: 
        num_bins=num_bins
        bin_width = extent/num_bins
    start = (0.5*2.8) * 3.085678e21
    bins_rad = np.geomspace(start,extent+0.05*extent,num_bins)
    dig = np.digitize(rad_mod,bins_rad,right=True)
    
    mass_in_bin = []
    density = []
    for ibin in range(len(bins_rad)):
        mass_in_bin.append(np.sum(dust_mass[dig == (ibin) ]))
        
        if ibin == 0:
            volume = 4/3 * np.pi * (bins_rad[ibin])**3
            area = np.pi * (bins_rad[ibin])**2
            # print(bins_rad[ibin])
            # print(set_plot_len(bins_rad[ibin]))
        else :
            volume = 4/3 * np.pi * (bins_rad[ibin]-bins_rad[ibin-1])**3
            area = np.pi * (bins_rad[ibin]-bins_rad[ibin-1])**2
            # print(bins_rad[ibin]-bins_rad[ibin-1])
            # print(set_plot_len(bins_rad[ibin]-bins_rad[ibin-1]))
        
        # density.append(mass_in_bin[ibin]/set_plot_len(set_plot_len(set_plot_len(volume))))
        density.append(mass_in_bin[ibin]/set_plot_len(set_plot_len(area)))
    # print(np.max(bins_rad))    
    # print(bins_rad/(extent/2))
    return np.array(density), bins_rad

def rad_dust_surf_dens(dm,SN_fac,label,folder,out_folder,snap_num,color,linestyle="-"):
    print(f"Radial plot for {dm} {SN_fac}")
    halo_pos,extent = set_center(folder,dm,SN_fac,snap_num)
    # extent = 100 * 3.085678e21
    method = out_folder.split('/')[-2]
    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)
    # density, bins_rad = calc_rad_dust(grid_physical_properties_data,halo_pos,extent)
    num_bins = 100
    extent = extent/2
    # extent = extent/10
    density, bins_rad = calc_rad_dust_surf_dens(grid_physical_properties_data,halo_pos,extent,num_bins)
    bins_rad = set_plot_len(bins_rad)/1000
    # print(f'R_200c = {set_plot_len(extent)/2} {UNIT_LENGTH_FOR_PLOTS}')
    # fig, ax = plt.subplots()
    plt.plot(bins_rad,density,label=label,color=color,ls=linestyle)
    if label=='WDM SN=10\%':    
        scaling_a = 1e-3 #* LITTLEH
        scaling_b = 3e-2 #* LITTLEH
        dust_rads = (bins_rad)#/1000
        obs_surface_dens()
        # print(bins_rad)
        ax = plt.gca()
        
        # import matplotlib.patches as mpatches
        # arr = mpatches.FancyArrowPatch((0.5*2.8, 10), (0.5*2.8, 100), arrowstyle='->,head_width=.5', mutation_scale=20,color='red',lw=4)
        # ax.add_patch(arr)

        rad_10_arg = np.argmin(np.abs(bins_rad-10))
        # plt.plot(bins_rad[rad_10_arg:],scaling_a*dust_rads[rad_10_arg:]**(-0.8),color='k',ls=':')#,label='Mernard+ (2010)')
        # t = plt.text(0.95,0.22,r"$\lambda$=1$\times 10^{-3}$",ha='right', va='bottom',transform=ax.transAxes,fontsize=30.0,weight='bold',color='black')
        plt.plot(bins_rad[rad_10_arg:],scaling_b*dust_rads[rad_10_arg:]**(-0.8),color='k',ls=':',label='Mernard+ (2010)',linewidth=7)
        t = plt.text(0.95,0.42,r"$\lambda$=3$\times 10^{-2}$",ha='right', va='bottom',transform=ax.transAxes,fontsize=30.0,weight='bold',color='black')
        # plt.plot(bins_rad,scaling_b*dust_rads**(-0.8),color='gray',ls=':')
        
        
    # plt.axvline(0.5/LITTLEH,ls=':')
    # plt.axvline(0.5,ls=':')
    # plt.axvline(set_plot_len(extent/2)/1000,ls=':')
    # plt.plot(bins_rad/(extent/2),density,label=label,color=color,ls=linestyle)
    
    # plt.xlim((0.5*2.8,bins_rad[-1]))
    plt.vlines(15/LITTLEH,1e-6,1e2,ls='--',color='k',alpha=0.7)
    # plt.arrow(0.5*2.8, 1e1, 0,78,width=0.05,color='red')
    
    
    # plt.ylim((1e-6,1e2))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'R [kpc]')
    
    # plt.ylabel(r'Dust Mass density [M$_{\odot}$ kpc$^{-2}$]')
    plt.ylabel(r'$\Sigma_{dust}$ [M$_{\odot}$ pc$^{-2}$]')
    plt.legend(loc='lower left')
    # plt.title(f"{method}")
    plt.tight_layout()
    plt.savefig(f'{out_folder}/rad_plot_dust_surf_dens_{method}.png',dpi=400)
    print(f"saving fig {out_folder}/rad_plot_dust_surf_dens_{method}.png")

def radial_dist_dust(dm,SN_fac,label,folder,snap_num,color,linestyle="-"):
    print(f"Radial plot for {dm} {SN_fac}")
    halo_pos,extent = set_center(folder,dm,SN_fac,snap_num)
    method = out_folder.split('/')[-2]
    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)
    density, bins_rad = calc_rad_dust(grid_physical_properties_data,halo_pos,extent)
    
    print(f'R_200c = {set_plot_len(extent)/2} {UNIT_LENGTH_FOR_PLOTS}')
    plt.plot(bins_rad/(extent/2),density,label=label,color=color,ls=linestyle)
    plt.yscale('log')
    # plt.xlim([0,1])
    # plt.ylim(1e-3,1e6)
    plt.xlabel(r'R/R200$_c$')
    # plt.ylabel(f'Dust Mass density [Msun/{UNIT_LENGTH_FOR_PLOTS}^3]')
    plt.ylabel(r'Dust Mass density [M$_{\odot}$ kpc$^{-3}$]')
    # plt.title(f"Radial Dust Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_folder}/rad_plot_dust.png',dpi=400)
    print(f"saving fig {out_folder}/rad_plot_dust.png")

def radial_dist_dust_dif(folder, out_folder):
    print(f"Radial difference plots")
    
    snap_num = 26
    
    halo_pos_cdm_05,extent_cdm_05 = set_center(folder,'cdm','sn_005',snap_num)
    halo_pos_cdm_10,extent_cdm_10 = set_center(folder,'cdm','sn_010',snap_num)
    halo_pos_wdm_05,extent_wdm_05 = set_center(folder,'wdm_3.5','sn_005',snap_num)
    halo_pos_wdm_10,extent_wdm_10 = set_center(folder,'wdm_3.5','sn_010',snap_num)

    bin_width = 5 * 3.085678e21 # 5 Kpc to cm
    avg_extent = np.average((extent_cdm_05,extent_cdm_10,extent_wdm_05,extent_wdm_10))
    num_bins = int((avg_extent+0.05*avg_extent)/bin_width)
    # print(num_bins)
    method = out_folder.split('/')[-2]
    
    grid_physical_properties_data_cdm_05 = load_grid_data(folder,method,'cdm','sn_005',snap_num)
    density_cdm_05, bins_rad_cdm_05 = calc_rad_dust(grid_physical_properties_data_cdm_05,halo_pos_cdm_05,extent_cdm_05,num_bins)

    
    grid_physical_properties_data_cdm_10 = load_grid_data(folder,method,'cdm','sn_010',snap_num)
    density_cdm_10, bins_rad_cdm_10 = calc_rad_dust(grid_physical_properties_data_cdm_10,halo_pos_cdm_10,extent_cdm_10,num_bins)

    
    grid_physical_properties_data_wdm_05 = load_grid_data(folder,method,'wdm_3.5','sn_005',snap_num)
    density_wdm_05, bins_rad_wdm_05 = calc_rad_dust(grid_physical_properties_data_wdm_05,halo_pos_wdm_05,extent_wdm_05,num_bins)

    
    grid_physical_properties_data_wdm_10 = load_grid_data(folder,method,'wdm_3.5','sn_010',snap_num)
    density_wdm_10, bins_rad_wdm_10 = calc_rad_dust(grid_physical_properties_data_wdm_10,halo_pos_wdm_10,extent_wdm_10,num_bins)

    # print(density_cdm_05.shape)
    # print(density_cdm_10.shape)
    # print(density_wdm_05.shape)
    # print(density_wdm_10.shape)

    plt.plot(bins_rad_cdm_10/(extent_cdm_10/2),density_cdm_05/density_cdm_10,label='CDM SN=5%',color='k')
    plt.plot(bins_rad_cdm_10/(extent_cdm_10/2),density_wdm_05/density_cdm_10,label='WDM SN=5%',color='gold')
    plt.plot(bins_rad_cdm_10/(extent_cdm_10/2),density_wdm_10/density_cdm_10,label='WDM SN=10%',color='cyan')
    plt.axhline(1,color='k',ls='--',alpha=1)
    plt.yscale('log')
    plt.ylim(1e-3,1e6)
    plt.xlabel(f'R200_c')
    plt.ylabel(f'Percent Difference')
    plt.legend()
    plt.savefig(f'{out_folder}/rad_plot_dust_dif.png',dpi=400)
      
def radial_dist_dust_cumulative(dm,SN_fac,label,folder,out_folder,halo_pos,extent):
    print(f"Cumulative radial plot for {dm} {SN_fac}")

    method = out_folder.split('/')[-2]
    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)

    pos_x = grid_physical_properties_data['gas_pos_x'] * LITTLEH * 3.085678e18 # convert parsecs to cm, convert back later
    pos_y = grid_physical_properties_data['gas_pos_y'] * LITTLEH * 3.085678e18
    pos_z = grid_physical_properties_data['gas_pos_z'] * LITTLEH * 3.085678e18
    
    pos_mask = make_mask(pos_x,pos_y,pos_z,halo_pos,extent)

    dust_mass = grid_physical_properties_data['particle_dustmass'][pos_mask]

    rad_dist_x = np.abs(pos_x[pos_mask]-halo_pos[0])
    rad_dist_y = np.abs(pos_y[pos_mask]-halo_pos[1])
    rad_dist_z = np.abs(pos_z[pos_mask]-halo_pos[2])

    rad_mod = np.sqrt(rad_dist_x**2+rad_dist_y**2+rad_dist_z**2)
    
    bin_width = 5 * 3.085678e21 # 5 Kpc to cm
    num_bins = int((extent+0.05*extent)/bin_width) # width of extent plus an extra 5% to cover the edges
    
    bins_rad = np.geomspace(1e-5,extent+0.05*extent,num_bins)
    dig = np.digitize(rad_mod,bins_rad,right=True)

    mass_in_bin = []
    density = []
    
    for ibin in range(len(bins_rad)):

        if ibin == 0:
            mass_in_bin.append(np.sum(dust_mass[dig == (ibin+1) ]))
            volume = 4/3 * np.pi * (bins_rad[ibin])**3
        else :
            mass_in_bin.append(np.sum(dust_mass[dig == (ibin+1) ])+mass_in_bin[ibin-1])
            volume = 4/3 * np.pi * (bins_rad[ibin]-bins_rad[ibin-1])**3
        density.append(mass_in_bin[ibin]/volume)
        # print(mass_in_bin)

    plt.plot(set_plot_len(bins_rad),mass_in_bin,label=label)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(f'Radius [{UNIT_LENGTH_FOR_PLOTS}]')
    plt.ylabel('Dust Mass [Msun]')
    # plt.ylabel('Dust Mass in bin [Msun/Kpc^3]')
    plt.legend()


if __name__ =='__main__':

    # folder  = 'N2048_L65_sd17492'
    # folder  = 'N2048_L65_sd28504'
    # folder  = 'N2048_L65_sd34920'
    # folder  = 'N2048_L65_sd46371'
    # folder  = 'N2048_L65_sd57839'
    folder  = 'N2048_L65_sd61284'
    # folder  = 'N2048_L65_sd70562'
    # folder  = 'N2048_L65_sd80325'
    # folder  = 'N2048_L65_sd93745'

    # folder_list = ['N2048_L65_sd17492', 'N2048_L65_sd28504', 'N2048_L65_sd34920', 'N2048_L65_sd46371', 'N2048_L65_sd57839', 'N2048_L65_sd61284', 'N2048_L65_sd70562','N2048_L65_sd80325', 'N2048_L65_sd93745']
    folder_list = ['N2048_L65_sd80325']

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if (simulation_name.startswith("N2048_L65_sd"))]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(folder_list)
    
    snap_num = 26
    num_bins = 80
    
    for folder in folder_list:
        print()
        print(f'Simulation -> {folder}')
        out_list = [f'./plots/{folder}/dtm/',f'./plots/{folder}/rr/',f'./plots/{folder}/li_bf/']    
        for out_folder in out_list:
            method = out_folder.split('/')[-2]
            print(f"method = {method}")
            if not os.path.exists(f'./{out_folder}'):
                os.makedirs(f'./{out_folder}')        
            rad_dust_surf_dens('cdm','sn_005','CDM SN=5\%',folder,out_folder,snap_num,'blue',"-.")
            rad_dust_surf_dens('cdm','sn_010','CDM SN=10\%',folder,out_folder,snap_num,'blue')
            
            rad_dust_surf_dens('wdm_3.5','sn_005','WDM SN=5\%',folder,out_folder,snap_num,'orange',"-.")
            rad_dust_surf_dens('wdm_3.5','sn_010','WDM SN=10\%',folder,out_folder,snap_num,'orange')
    
            plt.close()
        print(f'Finished plotting for simulation {folder}')