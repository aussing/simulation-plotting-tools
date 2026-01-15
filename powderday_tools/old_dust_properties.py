import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import inf
from numpy import nan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d 
import astropy.units as u
import h5py
import os
import cmasher as cmr

plt.rcParams['font.size'] = 11

LITTLEH = 0.6688
UNITMASS = 1e10

def set_center(folder,sn_fac,snap_num):
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')

    halo_pos   = np.array(halo_data['Group']['GroupPos'], dtype=np.float64) 
    halo_M200c = np.array(halo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS
    halo_masstypes = np.array(halo_data['Group']['GroupMassType'], dtype=np.float64)

    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0] #Select largest non-contaminated halo / resimulation target

    halo_pos = [halo_pos[mass_mask][halo_mainID,0],halo_pos[mass_mask][halo_mainID,1],halo_pos[mass_mask][halo_mainID,2]]
    R200c = np.array(halo_data['Group']['Group_R_Crit200'], dtype=np.float64)

    halo_rad = R200c[mass_mask][halo_mainID]
    # print(f"Halo Rad = {halo_rad}, halo pos = {halo_pos}\n")
    extent = halo_rad*2
    # extent = 0.5
    return halo_pos,extent

def get_extent(halo_pos,extent):

    xmin,xmax  = halo_pos[0]-extent,halo_pos[0]+extent
    ymin,ymax  = halo_pos[1]-extent,halo_pos[1]+extent
    zmin,zmax  = halo_pos[2]-extent,halo_pos[2]+extent
    
    return xmin,xmax,ymin,ymax,zmin,zmax

def load_grid_data(folder,dm,SN_fac):
    # data = np.load(f'./{folder}/rr_out/{dm}/{SN_fac}/grid_physical_properties.026_galaxy0.npz')
    # data = np.load(f'./{folder}/rr_output_test/grid_physical_properties.026_galaxy0.npz')
    data = np.load(f'./{folder}/output/{dm}/{SN_fac}/grid_physical_properties.026_galaxy0.npz')
    return data

def make_mask(pos_x,pos_y,pos_z,halo_pos,extent):
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)

    x_mask = (pos_x>=xmin) & (pos_x<=xmax)
    y_mask = (pos_y>=ymin) & (pos_y<=ymax)
    z_mask = (pos_z>=zmin) & (pos_z<=zmax)
    
    pos_mask = x_mask & y_mask & z_mask

    return pos_mask

def dust_mass_hist(dm,SN_fac,label,folder,out_folder,num_bins,halo_pos,extent):
    print(dm,'-',SN_fac)
    
    grid_physical_properties_data = load_grid_data(folder,dm,SN_fac)
    
    dust_pos_x = grid_physical_properties_data['dust_pos_x']/1e6*LITTLEH
    dust_pos_y = grid_physical_properties_data['dust_pos_y']/1e6*LITTLEH
    dust_pos_z = grid_physical_properties_data['dust_pos_z']/1e6*LITTLEH

    pos_mask = make_mask(dust_pos_x,dust_pos_y,dust_pos_z,halo_pos,extent)

    
    plot_scatter(dust_pos_x,dust_pos_y,dust_pos_z,pos_mask,dm,SN_fac,folder,out_folder)#,gas_pos_real,gas_len)

    dust_mass = grid_physical_properties_data['particle_dustmass'][pos_mask]
    print(f'Total dust mass in the halo region = {np.round(np.sum(dust_mass)/UNITMASS*1e3,2)}e7 Msun')
    bins=np.empty((num_bins,3))

    fig = plt.figure()
    ax = fig.add_subplot()
    cmap = cmr.torch
    data = plt.hist2d(dust_pos_x[pos_mask],dust_pos_y[pos_mask],bins=num_bins,data=dust_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4),cmap=cmap)
    # print(data)
    # data,x_edge,y_edge = np.histogram2d(gas_pos_x[pos_mask],gas_pos_y[pos_mask],bins=[bins[:,0],bins[:,1]],data=dust_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    # plt.imshow(data[0])
    # print(data)
    plt.colorbar(label='Dust Mass [Msun]')
    plt.title(f'{label}')
    plt.xlabel('x [Mpc]')
    plt.ylabel('y [Mpc]')
    fig.savefig(f'./plots/{out_folder}/dust_hist_{dm}_{SN_fac}.png',dpi=400)
    plt.close()
    return data

def plot_hist_dif(data_comp,model_name,data_base,base_name,folder,out_folder,dm,SN_fac,num_bins,halo_pos,extent):
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)
    
    dif = (data_comp[0]/data_base[0]-1)*100
    dif[dif==inf]=100
    dif[dif==-inf]=-100
    
    # cmap = cmr.prinsenvlag
    cmap = cmr.fusion_r
    vmax_dif = np.max(dif)
    vmax_dif = 150
    
    fig = plt.figure()
    ax = fig.add_subplot()
    dif = np.transpose(dif)
    h = ax.imshow(dif, extent=[xmin,xmax,ymin,ymax],origin='lower',cmap=cmap,vmin=-vmax_dif,vmax=vmax_dif)
    cs = ax.contour(np.transpose(data_comp[0]),levels=[50,100,1000],extent=[xmin,xmax,ymin,ymax],colors=['r','b','k'],alpha=0.6,linestyles='dashed')
    lb = ax.contour(np.transpose(data_base[0]),levels=[50,100,1000],extent=[xmin,xmax,ymin,ymax],colors=['r','b','k'],alpha=0.6)
    # ax.clabel(cs,"WDM",inline=True)
    # ax.clabel(lb,"CDM",inline=True)
    plt.colorbar(h, label='more CDM dust <- [%] difference -> more WDM dust')
    # plt.hist2d(data[0][:,0]/data_comp[0][:,0],bins=[data_comp[1],data_comp[2]])#,data=data[0]/data_comp[0])
    ax.set_xlabel('x [Mpc]')
    ax.set_ylabel('y [Mpc]')
    plt.title(f'{model_name} / {base_name} ')
    plt.legend(["dashed - WDM","Solid - CDM"])
    fig.savefig(f'./plots/{out_folder}/dust_hist_diff_{dm}_{SN_fac}.png',dpi=400)
    plt.close()

def radial_dist_dust(dm,SN_fac,label,folder,out_folder,halo_pos,extent):
    print(f"Radial plot for {dm} {SN_fac}")
    
    grid_physical_properties_data = load_grid_data(folder,dm,SN_fac)

    gas_pos_x = grid_physical_properties_data['gas_pos_x']/1e6*LITTLEH
    gas_pos_y = grid_physical_properties_data['gas_pos_y']/1e6*LITTLEH
    gas_pos_z = grid_physical_properties_data['gas_pos_z']/1e6*LITTLEH

    pos_mask = make_mask(gas_pos_x,gas_pos_y,gas_pos_z,halo_pos,extent)

    dust_mass = grid_physical_properties_data['particle_dustmass'][pos_mask]

    rad_dist_x = np.abs(gas_pos_x[pos_mask]-halo_pos[0])
    rad_dist_y = np.abs(gas_pos_y[pos_mask]-halo_pos[1])
    rad_dist_z = np.abs(gas_pos_z[pos_mask]-halo_pos[2])

    rad_mod = np.sqrt(rad_dist_x**2+rad_dist_y**2+rad_dist_z**2)
    bin_width = 5 #Kpc
    num_bins = int((extent*1000+10)/bin_width)
    bins_rad = np.linspace(1e-3,extent+0.02,num_bins)
    dig = np.digitize(rad_mod,bins_rad,right=True)
    mass_in_bin = []
    density = []
    for ibin in range(len(bins_rad)):
        mass_in_bin.append(np.sum(dust_mass[dig == (ibin+1) ]))
        
        if ibin == 0:
            volume = 4/3 * np.pi * (bins_rad[ibin]*1000)**3
        else :
            volume = 4/3 * np.pi * (bins_rad[ibin]*1000-bins_rad[ibin-1]*1000)**3
        density.append(mass_in_bin[ibin]/volume)


    plt.plot(bins_rad*1000,density,label=label)
    plt.yscale('log')
    # plt.ylim(1e-5,1e1)
    # plt.xscale('log')
    plt.xlabel('Radius [Kpc]')
    # plt.ylabel('Dust Mass [Msun]')
    plt.ylabel('Dust Mass in bin [Msun/Kpc^3]')
    plt.legend()
    
def radial_dist_dust_cumulative(dm,SN_fac,label,folder,out_folder,halo_pos,extent):
    print(f"Cumulative radial plot for {dm} {SN_fac}")

    grid_physical_properties_data = load_grid_data(folder,dm,SN_fac)

    dust_pos_x = grid_physical_properties_data['dust_pos_x']/1e6*LITTLEH
    dust_pos_y = grid_physical_properties_data['dust_pos_y']/1e6*LITTLEH
    dust_pos_z = grid_physical_properties_data['dust_pos_z']/1e6*LITTLEH
    
    pos_mask = make_mask(dust_pos_x,dust_pos_y,dust_pos_z,halo_pos,extent)

    dust_mass = grid_physical_properties_data['particle_dustmass'][pos_mask]

    rad_dist_x = np.abs(dust_pos_x[pos_mask]-halo_pos[0])
    rad_dist_y = np.abs(dust_pos_y[pos_mask]-halo_pos[1])
    rad_dist_z = np.abs(dust_pos_z[pos_mask]-halo_pos[2])

    rad_mod = np.sqrt(rad_dist_x**2+rad_dist_y**2+rad_dist_z**2)
    bin_width = 5 #Kpc
    num_bins = int((extent*1000+10)/bin_width)
    bins_rad = np.geomspace(1e-5,extent+0.02,num_bins)
    dig = np.digitize(rad_mod,bins_rad,right=True)
    mass_in_bin = []
    density = []
    for ibin in range(len(bins_rad)):
        if ibin == 0:
            mass_in_bin.append(np.sum(dust_mass[dig == (ibin+1) ]))
            volume = 4/3 * np.pi * (bins_rad[ibin]*1000)**3
        else :
            mass_in_bin.append(np.sum(dust_mass[dig == (ibin+1) ])+mass_in_bin[ibin-1])
            volume = 4/3 * np.pi * (bins_rad[ibin]*1000-bins_rad[ibin-1]*1000)**3
        density.append(mass_in_bin[ibin]/volume)
        # print(mass_in_bin)

    plt.plot(bins_rad*1000,mass_in_bin,label=label)
    # plt.plot(bins_rad,mass_in_bin,label=label)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Radius [Kpc]')
    plt.ylabel('Dust Mass [Msun]')
    # plt.ylabel('Dust Mass in bin [Msun/Kpc^3]')
    plt.legend()
        
def radial_dist_gas(dm,SN_fac,label,folder,out_folder,halo_pos,extent):
    grid_physical_properties_data = load_grid_data(folder,dm,SN_fac)

    gas_pos_x = grid_physical_properties_data['gas_pos_x']/1e6*LITTLEH
    gas_pos_y = grid_physical_properties_data['gas_pos_y']/1e6*LITTLEH
    gas_pos_z = grid_physical_properties_data['gas_pos_z']/1e6*LITTLEH
    
    pos_mask = make_mask(gas_pos_x,gas_pos_y,gas_pos_z,halo_pos,extent)

    gas_mass = grid_physical_properties_data['particle_gas_mass'][pos_mask]/1.989e33 #M_SUN in grams

    rad_dist_x = np.abs(gas_pos_x[pos_mask]-halo_pos[0])
    rad_dist_y = np.abs(gas_pos_y[pos_mask]-halo_pos[1])
    rad_dist_z = np.abs(gas_pos_z[pos_mask]-halo_pos[2])

    rad_mod = np.sqrt(rad_dist_x**2+rad_dist_y**2+rad_dist_z**2)
    bin_width = 10 #Kpc
    num_bins = np.int32((extent*1000+20-10)/bin_width)
    bins_rad = np.geomspace(0.01,extent+0.02,num_bins)
    dig = np.digitize(rad_mod,bins_rad,right=True)
    mass_in_bin = []
    
    for ibin in range(len(bins_rad)):
        mass_in_bin.append(np.sum(gas_mass[dig == (ibin+1) ]))
    
    plt.plot(bins_rad*1000,mass_in_bin,label=label)
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Radius [Kpc]')
    plt.ylabel('Gas Mass [Msun]')
    plt.legend()
    
def radial_dist_stars(dm,SN_fac,label,folder,out_folder,halo_pos,extent):
    print(SN_fac)
    grid_physical_properties_data = load_grid_data(folder,dm,SN_fac)
    snapshot = h5py.File(f"./{folder}/{dm}/{SN_fac}/snapshot_026.hdf5")
    
    star_pos = np.array(snapshot['PartType4']['Coordinates'],dtype=np.float32)/1e3

    star_pos_x = star_pos[:,0]
    star_pos_y = star_pos[:,1]
    star_pos_z = star_pos[:,2]

    pos_mask = make_mask(star_pos_x,star_pos_y,star_pos_z,halo_pos,extent)

    star_mass = np.array(snapshot['PartType4']['Masses'],dtype=np.float32)[pos_mask]*UNITMASS

    rad_dist_x = np.abs(star_pos_x[pos_mask]-halo_pos[0])
    rad_dist_y = np.abs(star_pos_y[pos_mask]-halo_pos[1])
    rad_dist_z = np.abs(star_pos_z[pos_mask]-halo_pos[2])
    
    rad_mod = np.sqrt(rad_dist_x**2+rad_dist_y**2+rad_dist_z**2)
    bin_width = 5 #Kpc
    num_bins = int((extent*1000+10)/bin_width)
    bins_rad = np.linspace(0.0,extent+0.02,num_bins)
    print(num_bins)
    dig = np.digitize(rad_mod,bins_rad,right=True)
    # print(np.unique(dig).shape)
    mass_in_bin = []
 
    for ibin in range(len(bins_rad)):
        mass_in_bin.append(np.sum(star_mass[dig == (ibin+1) ]))
        
        if ibin == 0:
            volume = 4/3 * np.pi * (bins_rad[ibin]*1000)**3
        else :
            volume = 4/3 * np.pi * (bins_rad[ibin]*1000-bins_rad[ibin-1]*1000)**3
        density = mass_in_bin/volume

    print(volume)
    # print(volume.shape)
    print(len(volume))
    print(len(mass_in_bin))
    plt.plot(bins_rad*1000,mass_in_bin,label=label)
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Radius [Kpc]')
    plt.ylabel('Star Mass in bin [Msun/bin_size]')
    plt.legend()
    
def plot_scatter(gas_pos_x,gas_pos_y,gas_pos_z,pos_mask,dm,SN_fac,folder,out_folder):#,gas_pos_real,gas_len):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    # plt.scatter(skirt_gas_data[0:gas_len,0][pos_mask[0:gas_len]]/1e6*h,skirt_gas_data[0:gas_len,1][pos_mask[0:gas_len]]/1e6*h,s=0.01)
    ax.scatter(gas_pos_x[pos_mask],gas_pos_y[pos_mask],gas_pos_z[pos_mask],s=0.01,alpha=0.6,zorder=2)
    # ax.scatter(gas_pos_real[0:gas_len,0],gas_pos_real[0:gas_len,1],gas_pos_real[0:gas_len,2],color='r',s=0.01,zorder=1)
    ax.set_aspect('auto')
    # fig.tight_layout()
    fig.savefig(f'./plots/{out_folder}/part_scatter_{dm}_{SN_fac}.png',dpi=400)
    plt.close()


if __name__ =='__main__':

    # folder = 'N2048_L65_sd3490'
    # out_folder = 'N2048_L65_sd234920/rr_out'
    folder = 'N2048_L65_sd46371'
    out_folder = 'tests/norm/'
    # out_folder = 'N2048_L65_sd46371/rr_out_test'
    print(folder)
    if not os.path.exists(f'./plots/{out_folder}'):
        os.makedirs(f'./plots/{out_folder}')
    
    halo_pos,extent = set_center(folder,sn_fac,snap_num)

    num_bins=80
    dm='cdm'
    # data_cdm_sn_05 = dust_mass_hist(dm,'sn_005','CDM SN=5%',folder,out_folder,num_bins,halo_pos,extent)
    data_cdm_sn_10 = dust_mass_hist(dm,'sn_010','CDM SN=10%',folder,out_folder,num_bins,halo_pos,extent)
    # data_cdm_sn_15 = dust_mass_hist(dm,'sn_015','CDM SN=15%',folder,out_folder,num_bins,halo_pos,extent)
    # # data_cdm_sn_20 = dust_mass_hist(dm,'sn_020','CDM SN=20%',folder,num_bins,halo_pos,extent)
    dm='wdm_3.5'
    # data_wdm_sn_05 = dust_mass_hist(dm,'sn_005','WDM SN=5%',folder,out_folder,num_bins,halo_pos,extent)
    # data_wdm_sn_10 = dust_mass_hist(dm,'sn_010','WDM SN=10%',folder,out_folder,num_bins,halo_pos,extent)
    # data_wdm_sn_15 = dust_mass_hist(dm,'sn_015','WDM SN=15%',folder,out_folder,num_bins,halo_pos,extent)

    # plot_hist_dif(data_cdm_sn_05,'CDM SN 5%',data_cdm_sn_10,'CDM SN 10%',folder,out_folder,'cdm','sn_005',num_bins,halo_pos,extent)
    # plot_hist_dif(data_cdm_sn_15,'CDM SN 15%',data_cdm_sn_10,'CDM SN 10%',folder,out_folder,'cdm','sn_015',num_bins,halo_pos,extent)

    
    # plot_hist_dif(data_wdm_sn_05,'WDM SN 5%',data_wdm_sn_10,'WDM SN 10%',folder,out_folder,'wdm','sn_005',num_bins,halo_pos,extent)
    # plot_hist_dif(data_wdm_sn_15,'WDM SN 15%',data_wdm_sn_10,'WDM SN 10%',folder,out_folder,'wdm','sn_015',num_bins,halo_pos,extent)


    # plot_hist_dif(data_wdm_sn_05,'WDM SN 5% ',data_cdm_sn_10,'CDM SN 10%',folder,out_folder,'wdm_cdm','05_10',num_bins,halo_pos,extent)
    # plt.close()

    # radial_dist_dust('cdm','sn_005','CDM SN=5%',folder,out_folder,halo_pos,extent)
    radial_dist_dust('cdm','sn_010','CDM SN=10%',folder,out_folder,halo_pos,extent)
    # # radial_dist_dust('cdm','sn_015','CDM SN=15%',folder,out_folder,halo_pos,extent)
    # # # radial_dist_dust('cdm','sn_020','CDM SN=20%',folder)
    # radial_dist_dust('wdm_3.5','sn_005','WDM SN=5%',folder,out_folder,halo_pos,extent)
    # radial_dist_dust('wdm_3.5','sn_010','WDM SN=10%',folder,out_folder,halo_pos,extent)
    # # radial_dist_dust('wdm_3.5','sn_015','WDM SN=15%',folder,out_folder,halo_pos,extent)
    plt.savefig(f'./plots/{out_folder}/rad_plot_dust.png',dpi=400)
    
    plt.close()

    # radial_dist_dust_cumulative('cdm','sn_005','CDM SN=5%',folder,out_folder,halo_pos,extent)
    radial_dist_dust_cumulative('cdm','sn_010','CDM SN=10%',folder,out_folder,halo_pos,extent)
    # radial_dist_dust_cumulative('cdm','sn_015','CDM SN=15%',folder,out_folder,halo_pos,extent)
    # # # radial_dist_dust_cumulative('cdm','sn_020','CDM SN=20%',folder)
    # radial_dist_dust_cumulative('wdm_3.5','sn_005','WDM SN=5%',folder,out_folder,halo_pos,extent)
    # radial_dist_dust_cumulative('wdm_3.5','sn_010','WDM SN=10%',folder,out_folder,halo_pos,extent)
    # radial_dist_dust_cumulative('wdm_3.5','sn_015','WDM SN=15%',folder,out_folder,halo_pos,extent)
    plt.savefig(f'./plots/{out_folder}/rad_cumul_plot_dust.png',dpi=400)

    plt.close()
    # radial_dist_gas('cdm','sn_005','CDM SN=5%',folder,halo_pos,extent)
    # radial_dist_gas('cdm','sn_010','CDM SN=10%',folder,halo_pos,extent)
    # # radial_dist_gas('cdm','sn_015','CDM SN=15%',folder,halo_pos,extent)
    # # radial_dist_gas('cdm','sn_020','CDM SN=20%',folder)
    # radial_dist_gas('wdm_3.5','sn_005','WDM SN=5%',folder,halo_pos,extent)
    # radial_dist_gas('wdm_3.5','sn_010','WDM SN=10%',folder,halo_pos,extent)
    # plt.savefig(f'./plots/{folder}/rad_plot_gas.png',dpi=400)
    plt.close()

    # radial_dist_stars(dm,SN_fac,label,folder,out_folder,halo_pos,extent)
    # radial_dist_stars('cdm','sn_005','CDM SN=5%',folder,out_folder,halo_pos,extent)
    # radial_dist_stars('cdm','sn_010','CDM SN=10%',folder,halo_pos,extent)
    # # radial_dist_stars('cdm','sn_015','CDM SN=15%',folder,halo_pos,extent)
    # # radial_dist_stars('cdm','sn_020','CDM SN=20%',folder)
    # radial_dist_stars('wdm_3.5','sn_005','WDM SN=5%',folder,halo_pos,extent)
    # radial_dist_stars('wdm_3.5','sn_010','WDM SN=10%',folder,halo_pos,extent)
    # plt.savefig(f'./plots/{out_folder}/rad_plot_stars.png',dpi=400)
    print(f'Finished plotting for simulation {folder}')