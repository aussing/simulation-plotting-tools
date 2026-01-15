import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import tqdm
from tqdm import trange
from scipy.spatial.transform import Rotation as R


plt.rcParams['font.size'] = 11

LITTLEH = 0.6688
UNITMASS = 1e10

def set_center(folder,snap_num,dm,sn_fac):
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
    print(f"Halo Rad = {halo_rad}, halo pos = {halo_pos}\n")
    extent = halo_rad*2
    # extent = 0.5
    return halo_pos,extent

def get_extent(halo_pos,extent):

    xmin,xmax  = halo_pos[0]-extent,halo_pos[0]+extent
    ymin,ymax  = halo_pos[1]-extent,halo_pos[1]+extent
    zmin,zmax  = halo_pos[2]-extent,halo_pos[2]+extent
    
    return xmin,xmax,ymin,ymax,zmin,zmax

def gini(data):

    flattened = np.sort(np.ravel(data))
    npix = np.size(flattened)
    normalization = np.abs(np.mean(flattened)) * npix * (npix - 1)
    kernel = (2.0 * np.arange(1, npix + 1) - npix - 1) * np.abs(flattened)

    return np.sum(kernel) / normalization

def dust_mass_hist(dm,SN_fac,folder,num_bins,halo_pos,extent,snap_num):
    # print(dm,'-',SN_fac, " Snap - ", snap_num)
    grid_physical_properties_data = np.load(f'./{folder}/rr_out/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)
    
    dust_pos_x = grid_physical_properties_data['gas_pos_x']/1e6*LITTLEH
    dust_pos_y = grid_physical_properties_data['gas_pos_y']/1e6*LITTLEH
    dust_pos_z = grid_physical_properties_data['gas_pos_z']/1e6*LITTLEH
    
    x_mask = (dust_pos_x>=xmin) & (dust_pos_x<=xmax)
    y_mask = (dust_pos_y>=ymin) & (dust_pos_y<=ymax)
    z_mask = (dust_pos_z>=zmin) & (dust_pos_z<=zmax)

    pos_mask = x_mask & y_mask & z_mask
    
    dust_mass = grid_physical_properties_data['particle_dustmass']*UNITMASS
    
    
    bins=np.empty((num_bins,3))
    bins[:,0] = np.linspace(xmin,xmax,num_bins)
    bins[:,1] = np.linspace(ymin,ymax,num_bins)
    bins[:,2] = np.linspace(zmin,zmax,num_bins)
    # print(bins.shape)
    data_xyz,edges = np.histogramdd((dust_pos_x[pos_mask],dust_pos_y[pos_mask],dust_pos_z[pos_mask]),bins=num_bins,weights=dust_mass[pos_mask])
    # print(data_xyz)
    plt.close()
    data_xy = plt.hist2d(dust_pos_x[pos_mask],dust_pos_y[pos_mask],bins=[bins[:,0],bins[:,1]],data=dust_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    plt.colorbar()
    plt.savefig(f"./test_plots/dust_hist_snap_{snap_num}",dpi=400)
    plt.close()
    data_xz = plt.hist2d(dust_pos_x[pos_mask],dust_pos_z[pos_mask],bins=[bins[:,0],bins[:,2]],data=dust_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_yz = plt.hist2d(dust_pos_y[pos_mask],dust_pos_z[pos_mask],bins=[bins[:,1],bins[:,2]],data=dust_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    # plt.close()
    return data_xy, data_xz, data_yz, data_xyz

def gas_mass_hist(dm,SN_fac,folder,num_bins,halo_pos,extent,snap_num):
    grid_physical_properties_data = np.load(f'./{folder}/rr_out/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)
    
    gas_pos_x = grid_physical_properties_data['gas_pos_x']/1e6*LITTLEH
    gas_pos_y = grid_physical_properties_data['gas_pos_y']/1e6*LITTLEH
    gas_pos_z = grid_physical_properties_data['gas_pos_z']/1e6*LITTLEH
    
    x_mask = (gas_pos_x>=xmin) & (gas_pos_x<=xmax)
    y_mask = (gas_pos_y>=ymin) & (gas_pos_y<=ymax)
    z_mask = (gas_pos_z>=zmin) & (gas_pos_z<=zmax)

    pos_mask = x_mask & y_mask & z_mask
    
    gas_mass = grid_physical_properties_data['particle_gas_mass'][pos_mask]/1.989e33
    
    bins=np.empty((num_bins,3))
    bins[:,0] = np.linspace(xmin,xmax,num_bins)
    bins[:,1] = np.linspace(ymin,ymax,num_bins)
    bins[:,2] = np.linspace(zmin,zmax,num_bins)
    
    data_xy = plt.hist2d(gas_pos_x[pos_mask],gas_pos_y[pos_mask],bins=[bins[:,0],bins[:,1]],data=gas_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_xz = plt.hist2d(gas_pos_x[pos_mask],gas_pos_z[pos_mask],bins=[bins[:,0],bins[:,2]],data=gas_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_yz = plt.hist2d(gas_pos_y[pos_mask],gas_pos_z[pos_mask],bins=[bins[:,1],bins[:,2]],data=gas_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    plt.colorbar()
    plt.savefig(f"./test_plots/gas_hist_snap_{snap_num}",dpi=400)
    plt.close()
    return data_xy,data_xz,data_yz

def star_mass_hist(dm,SN_fac,folder,num_bins,halo_pos,extent,snap_num):
    # print(dm,'-',SN_fac, " Snap - ", snap_num)
    snapshot = h5py.File(f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/snapshot_{str(snap_num).zfill(3)}.hdf5")
    # print(f"./{folder}/{dm}/{SN_fac}/snapshot_{str(snap_num).zfill(3)}.hdf5")
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)
    
    star_pos = np.array(snapshot['PartType4']['Coordinates'],dtype=np.float32)

    star_pos_x = star_pos[:,0]
    star_pos_y = star_pos[:,1]
    star_pos_z = star_pos[:,2]
    
    x_mask = (star_pos_x>=xmin) & (star_pos_x<=xmax)
    y_mask = (star_pos_y>=ymin) & (star_pos_y<=ymax)
    z_mask = (star_pos_z>=zmin) & (star_pos_z<=zmax)

    pos_mask = x_mask & y_mask & z_mask

    star_mass = np.array(snapshot['PartType4']['Masses'],dtype=np.float32)[pos_mask]*UNITMASS
    
    bins=np.empty((num_bins,3))
    bins[:,0] = np.linspace(xmin,xmax,num_bins)
    bins[:,1] = np.linspace(ymin,ymax,num_bins)
    bins[:,2] = np.linspace(zmin,zmax,num_bins)

    
    data_xy = plt.hist2d(star_pos_x[pos_mask],star_pos_y[pos_mask],bins=[bins[:,0],bins[:,1]],data=star_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_xz = plt.hist2d(star_pos_x[pos_mask],star_pos_z[pos_mask],bins=[bins[:,0],bins[:,2]],data=star_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_yz = plt.hist2d(star_pos_y[pos_mask],star_pos_z[pos_mask],bins=[bins[:,1],bins[:,2]],data=star_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    plt.colorbar()
    plt.savefig(f"./test_plots/star_hist_snap_{snap_num}",dpi=400)
    plt.close()
    return data_xy,data_xz,data_yz
    
def gini_dust(folder,snap_list,a_list,num_bins,out_folder):
    print("Calculating dust gini coefficient")
    z_list = []
    gini_score_cdm_sn_05_xy, gini_score_cdm_sn_10_xy, gini_score_wdm_sn_05_xy, gini_score_wdm_sn_10_xy = [],[],[],[]

    gini_score_cdm_sn_05_xz, gini_score_cdm_sn_10_xz, gini_score_wdm_sn_05_xz, gini_score_wdm_sn_10_xz = [],[],[],[]
    
    gini_score_cdm_sn_05_yz, gini_score_cdm_sn_10_yz, gini_score_wdm_sn_05_yz, gini_score_wdm_sn_10_yz = [],[],[],[]

    gini_score_cdm_sn_05_xyz, gini_score_cdm_sn_10_xyz, gini_score_wdm_sn_05_xyz, gini_score_wdm_sn_10_xyz = [],[],[],[]
    
    for i_snap in trange(len(snap_list)):
        snap_num = snap_list[i_snap]
        a_fac = a_list[i_snap]
        halo_pos,extent = set_center(folder,snap_num,'cdm','sn_010')
        halo_pos=np.array(halo_pos)*a_fac
        # extent = extent*a_fac
        # print(f"halo pos = {halo_pos}, extent = {extent}")
        z = 1/a_fac-1
        z_list.append(z)
        
        dm='cdm'
        data_cdm_sn_05_xy,data_cdm_sn_05_xz,data_cdm_sn_05_yz, data_cdm_sn_05_xyz = dust_mass_hist(dm,'sn_005',folder,num_bins,halo_pos,extent,snap_num)
        data_cdm_sn_10_xy,data_cdm_sn_10_xz,data_cdm_sn_10_yz, data_cdm_sn_10_xyz = dust_mass_hist(dm,'sn_010',folder,num_bins,halo_pos,extent,snap_num)
        # print(data_cdm_sn_05_xy)
        # print(data_cdm_sn_05_xyz)
        dm='wdm_3.5'
        data_wdm_sn_05_xy,data_wdm_sn_05_xz,data_wdm_sn_05_yz, data_wdm_sn_05_xyz = dust_mass_hist(dm,'sn_005',folder,num_bins,halo_pos,extent,snap_num)
        data_wdm_sn_10_xy,data_wdm_sn_10_xz,data_wdm_sn_10_yz, data_wdm_sn_10_xyz = dust_mass_hist(dm,'sn_010',folder,num_bins,halo_pos,extent,snap_num)
        
        score_cdm_sn_05_xy = gini(data_cdm_sn_05_xy[0])
        score_cdm_sn_10_xy = gini(data_cdm_sn_10_xy[0])
        score_wdm_sn_05_xy = gini(data_wdm_sn_05_xy[0])
        score_wdm_sn_10_xy = gini(data_wdm_sn_10_xy[0])

        score_cdm_sn_05_xz = gini(data_cdm_sn_05_xz[0])
        score_cdm_sn_10_xz = gini(data_cdm_sn_10_xz[0])
        score_wdm_sn_05_xz = gini(data_wdm_sn_05_xz[0])
        score_wdm_sn_10_xz = gini(data_wdm_sn_10_xz[0])
        
        score_cdm_sn_05_yz = gini(data_cdm_sn_05_yz[0])
        score_cdm_sn_10_yz = gini(data_cdm_sn_10_yz[0])
        score_wdm_sn_05_yz = gini(data_wdm_sn_05_yz[0])
        score_wdm_sn_10_yz = gini(data_wdm_sn_10_yz[0])

        score_cdm_sn_05_xyz = gini(data_cdm_sn_05_xyz)
        score_cdm_sn_10_xyz = gini(data_cdm_sn_10_xyz)
        score_wdm_sn_05_xyz = gini(data_wdm_sn_05_xyz)
        score_wdm_sn_10_xyz = gini(data_wdm_sn_10_xyz)
        
        gini_score_cdm_sn_05_xy.append(score_cdm_sn_05_xy)
        gini_score_cdm_sn_10_xy.append(score_cdm_sn_10_xy)
        gini_score_wdm_sn_05_xy.append(score_wdm_sn_05_xy)
        gini_score_wdm_sn_10_xy.append(score_wdm_sn_10_xy)

        gini_score_cdm_sn_05_xz.append(score_cdm_sn_05_xz)
        gini_score_cdm_sn_10_xz.append(score_cdm_sn_10_xz)
        gini_score_wdm_sn_05_xz.append(score_wdm_sn_05_xz)
        gini_score_wdm_sn_10_xz.append(score_wdm_sn_10_xz)

        gini_score_cdm_sn_05_yz.append(score_cdm_sn_05_yz)
        gini_score_cdm_sn_10_yz.append(score_cdm_sn_10_yz)
        gini_score_wdm_sn_05_yz.append(score_wdm_sn_05_yz)
        gini_score_wdm_sn_10_yz.append(score_wdm_sn_10_yz)

        gini_score_cdm_sn_05_xyz.append(score_cdm_sn_05_xyz)
        gini_score_cdm_sn_10_xyz.append(score_cdm_sn_10_xyz)
        gini_score_wdm_sn_05_xyz.append(score_wdm_sn_05_xyz)
        gini_score_wdm_sn_10_xyz.append(score_wdm_sn_10_xyz)

    print(gini_score_cdm_sn_05_xyz)
    print()
    print(gini_score_cdm_sn_05_xy)

    fig = plt.figure()
    # plt.plot(z_list,gini_score_cdm_sn_05_xy,label="CDM 5%",color='C0')
    # plt.plot(z_list,gini_score_cdm_sn_10_xy,label="CDM 10%",color='C1')
    # plt.plot(z_list,gini_score_wdm_sn_05_xy,label="WDM 5%",color='C2')
    # plt.plot(z_list,gini_score_wdm_sn_10_xy,label="WDM 10%",color='C3')
    
    # plt.plot(z_list,gini_score_cdm_sn_05_xz,color='C0',label="CDM 5%" )#,alpha=0.6)#,ls='--')
    # plt.plot(z_list,gini_score_cdm_sn_10_xz,color='C1',label="CDM 10%")#,alpha=0.6)#,ls='--')
    # plt.plot(z_list,gini_score_wdm_sn_05_xz,color='C2',label="WDM 5%" )#,alpha=0.6)#,ls='--')
    # plt.plot(z_list,gini_score_wdm_sn_10_xz,color='C3',label="WDM 10%")#,alpha=0.6)#,ls='--')
    
    # plt.plot(z_list,gini_score_cdm_sn_05_yz,color='C0',label="CDM 5%" )#,alpha=0.3)#,ls='-.')
    # plt.plot(z_list,gini_score_cdm_sn_10_yz,color='C1',label="CDM 10%")#,alpha=0.3)#,ls='-.')
    # plt.plot(z_list,gini_score_wdm_sn_05_yz,color='C2',label="WDM 5%" )#,alpha=0.3)#,ls='-.')
    # plt.plot(z_list,gini_score_wdm_sn_10_yz,color='C3',label="WDM 10%")#,alpha=0.3)#,ls='-.')

    gini_cdm_05 = np.column_stack((gini_score_cdm_sn_05_xy,gini_score_cdm_sn_05_xz))#,gini_score_cdm_sn_05_yz))
    gini_cdm_10 = np.column_stack((gini_score_cdm_sn_10_xy,gini_score_cdm_sn_10_xz))#,gini_score_cdm_sn_10_yz))
    gini_wdm_05 = np.column_stack((gini_score_wdm_sn_05_xy,gini_score_wdm_sn_05_xz))#,gini_score_wdm_sn_05_yz))
    gini_wdm_10 = np.column_stack((gini_score_wdm_sn_10_xy,gini_score_wdm_sn_10_xz))#,gini_score_wdm_sn_10_yz))

    # dif_c_05_c_10 = (np.array(gini_score_cdm_sn_05_xyz)/np.array(gini_score_cdm_sn_10_xyz) -1 ) #* 100
    # dif_w_05_c_10 = (np.array(gini_score_wdm_sn_05_xyz)/np.array(gini_score_cdm_sn_10_xyz) -1 ) #* 100
    # dif_w_10_c_10 = (np.array(gini_score_wdm_sn_10_xyz)/np.array(gini_score_cdm_sn_10_xyz) -1 ) #* 100

    # plt.plot(z_list, dif_c_05_c_10,label="CDM 5%",color='C0')
    # plt.plot(z_list, dif_w_05_c_10,label="WDM 5%",color='C1')
    # plt.plot(z_list, dif_w_10_c_10,label="WDM 10%",color='C2')
    # plt.axhline(0,c='k',ls='--',alpha=0.5)
    
    plt.plot(z_list,np.mean(gini_cdm_05,axis=1),label="CDM 5%",color='C0')
    plt.plot(z_list,np.mean(gini_cdm_10,axis=1),label="CDM 10%",color='C1')#,alpha=0.5,)
    plt.plot(z_list,np.mean(gini_wdm_05,axis=1),label="WDM 5%",color='C2')
    plt.plot(z_list,np.mean(gini_wdm_10,axis=1),label="WDM 10%",color='C3')

    # plt.plot(z_list,gini_score_cdm_sn_05_xyz,color='C0',label="CDM 5%")
    # plt.plot(z_list,gini_score_cdm_sn_10_xyz,color='C1',label="CDM 10%")
    # plt.plot(z_list,gini_score_wdm_sn_05_xyz,color='C2',label="WDM 5%")
    # plt.plot(z_list,gini_score_wdm_sn_10_xyz,color='C3',label="WDM 10%")
    
    # plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel("Redshift")
    plt.ylabel("Gini score")
    fig.savefig(f"{out_folder}/gini_dust_history_test.png",dpi=400)
    plt.close()

def gini_gas(folder,snap_list,a_list,num_bins):
    print("Calculating gas gini coefficient")
    z_list = []
    gini_score_cdm_sn_05_xy, gini_score_cdm_sn_10_xy, gini_score_wdm_sn_05_xy, gini_score_wdm_sn_10_xy = [],[],[],[]

    gini_score_cdm_sn_05_xz, gini_score_cdm_sn_10_xz, gini_score_wdm_sn_05_xz, gini_score_wdm_sn_10_xz = [],[],[],[]
    
    gini_score_cdm_sn_05_yz, gini_score_cdm_sn_10_yz, gini_score_wdm_sn_05_yz, gini_score_wdm_sn_10_yz = [],[],[],[]
    
    for i_snap in trange(len(snap_list)):
        snap_num = snap_list[i_snap]
        a_fac = a_list[i_snap]
        halo_pos,extent = set_center(folder,snap_num,'cdm','sn_010')
        halo_pos=np.array(halo_pos)*a_fac
        z = 1/a_fac-1
        z_list.append(z)
        
        dm='cdm'
        data_cdm_sn_05_xy,data_cdm_sn_05_xz,data_cdm_sn_05_yz = dust_mass_hist(dm,'sn_005',folder,num_bins,halo_pos,extent,snap_num)
        data_cdm_sn_10_xy,data_cdm_sn_10_xz,data_cdm_sn_10_yz = dust_mass_hist(dm,'sn_010',folder,num_bins,halo_pos,extent,snap_num)

        dm='wdm_3.5'
        data_wdm_sn_05_xy,data_wdm_sn_05_xz,data_wdm_sn_05_yz = dust_mass_hist(dm,'sn_005',folder,num_bins,halo_pos,extent,snap_num)
        data_wdm_sn_10_xy,data_wdm_sn_10_xz,data_wdm_sn_10_yz = dust_mass_hist(dm,'sn_010',folder,num_bins,halo_pos,extent,snap_num)
        
        score_cdm_sn_05_xy = gini(data_cdm_sn_05_xy)
        score_cdm_sn_10_xy = gini(data_cdm_sn_10_xy)
        score_wdm_sn_05_xy = gini(data_wdm_sn_05_xy)
        score_wdm_sn_10_xy = gini(data_wdm_sn_10_xy)

        score_cdm_sn_05_xz = gini(data_cdm_sn_05_xz)
        score_cdm_sn_10_xz = gini(data_cdm_sn_10_xz)
        score_wdm_sn_05_xz = gini(data_wdm_sn_05_xz)
        score_wdm_sn_10_xz = gini(data_wdm_sn_10_xz)
        
        score_cdm_sn_05_yz = gini(data_cdm_sn_05_yz)
        score_cdm_sn_10_yz = gini(data_cdm_sn_10_yz)
        score_wdm_sn_05_yz = gini(data_wdm_sn_05_yz)
        score_wdm_sn_10_yz = gini(data_wdm_sn_10_yz)
        
        gini_score_cdm_sn_05_xy.append(score_cdm_sn_05_xy)
        gini_score_cdm_sn_10_xy.append(score_cdm_sn_10_xy)
        gini_score_wdm_sn_05_xy.append(score_wdm_sn_05_xy)
        gini_score_wdm_sn_10_xy.append(score_wdm_sn_10_xy)

        gini_score_cdm_sn_05_xz.append(score_cdm_sn_05_xz)
        gini_score_cdm_sn_10_xz.append(score_cdm_sn_10_xz)
        gini_score_wdm_sn_05_xz.append(score_wdm_sn_05_xz)
        gini_score_wdm_sn_10_xz.append(score_wdm_sn_10_xz)

        gini_score_cdm_sn_05_yz.append(score_cdm_sn_05_yz)
        gini_score_cdm_sn_10_yz.append(score_cdm_sn_10_yz)
        gini_score_wdm_sn_05_yz.append(score_wdm_sn_05_yz)
        gini_score_wdm_sn_10_yz.append(score_wdm_sn_10_yz)

    # print(z_list)

    fig = plt.figure()
    plt.plot(z_list,gini_score_cdm_sn_05_xy,label="CDM 5%")
    plt.plot(z_list,gini_score_cdm_sn_10_xy,label="CDM 10%")
    plt.plot(z_list,gini_score_wdm_sn_05_xy,label="WDM 5%")
    plt.plot(z_list,gini_score_wdm_sn_10_xy,label="WDM 10%")
    plt.xscale('log')
    # plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel("Redshift")
    plt.ylabel("Gini score")
    fig.savefig("./gini_gas_history.png",dpi=400)
    plt.close()

def gini_star(folder,snap_list,a_list,num_bins):
    print("Calculating star gini coefficient")
    z_list = []
    gini_score_cdm_sn_05_xy, gini_score_cdm_sn_10_xy, gini_score_wdm_sn_05_xy, gini_score_wdm_sn_10_xy = [],[],[],[]

    gini_score_cdm_sn_05_xz, gini_score_cdm_sn_10_xz, gini_score_wdm_sn_05_xz, gini_score_wdm_sn_10_xz = [],[],[],[]
    
    gini_score_cdm_sn_05_yz, gini_score_cdm_sn_10_yz, gini_score_wdm_sn_05_yz, gini_score_wdm_sn_10_yz = [],[],[],[]
     
    for i_snap in trange(len(snap_list)):
        snap_num = snap_list[i_snap]
        a_fac = a_list[i_snap]
        halo_pos,extent = set_center(folder,snap_num,'cdm','sn_010')
        halo_pos=np.array(halo_pos)*a_fac
        z = 1/a_fac-1
        z_list.append(z)
        
        dm='cdm'
        data_cdm_sn_05_xy,data_cdm_sn_05_xz,data_cdm_sn_05_yz = star_mass_hist(dm,'sn_005',folder,num_bins,halo_pos,extent,snap_num)
        data_cdm_sn_10_xy,data_cdm_sn_10_xz,data_cdm_sn_10_yz = star_mass_hist(dm,'sn_010',folder,num_bins,halo_pos,extent,snap_num)

        dm='wdm_3.5'
        data_wdm_sn_05_xy,data_wdm_sn_05_xz,data_wdm_sn_05_yz = star_mass_hist(dm,'sn_005',folder,num_bins,halo_pos,extent,snap_num)
        data_wdm_sn_10_xy,data_wdm_sn_10_xz,data_wdm_sn_10_yz = star_mass_hist(dm,'sn_010',folder,num_bins,halo_pos,extent,snap_num)
        
        score_cdm_sn_05_xy = gini(data_cdm_sn_05_xy)
        score_cdm_sn_10_xy = gini(data_cdm_sn_10_xy)
        score_wdm_sn_05_xy = gini(data_wdm_sn_05_xy)
        score_wdm_sn_10_xy = gini(data_wdm_sn_10_xy)

        score_cdm_sn_05_xz = gini(data_cdm_sn_05_xz)
        score_cdm_sn_10_xz = gini(data_cdm_sn_10_xz)
        score_wdm_sn_05_xz = gini(data_wdm_sn_05_xz)
        score_wdm_sn_10_xz = gini(data_wdm_sn_10_xz)
        
        score_cdm_sn_05_yz = gini(data_cdm_sn_05_yz)
        score_cdm_sn_10_yz = gini(data_cdm_sn_10_yz)
        score_wdm_sn_05_yz = gini(data_wdm_sn_05_yz)
        score_wdm_sn_10_yz = gini(data_wdm_sn_10_yz)
        
        gini_score_cdm_sn_05_xy.append(score_cdm_sn_05_xy)
        gini_score_cdm_sn_10_xy.append(score_cdm_sn_10_xy)
        gini_score_wdm_sn_05_xy.append(score_wdm_sn_05_xy)
        gini_score_wdm_sn_10_xy.append(score_wdm_sn_10_xy)

        gini_score_cdm_sn_05_xz.append(score_cdm_sn_05_xz)
        gini_score_cdm_sn_10_xz.append(score_cdm_sn_10_xz)
        gini_score_wdm_sn_05_xz.append(score_wdm_sn_05_xz)
        gini_score_wdm_sn_10_xz.append(score_wdm_sn_10_xz)

        gini_score_cdm_sn_05_yz.append(score_cdm_sn_05_yz)
        gini_score_cdm_sn_10_yz.append(score_cdm_sn_10_yz)
        gini_score_wdm_sn_05_yz.append(score_wdm_sn_05_yz)
        gini_score_wdm_sn_10_yz.append(score_wdm_sn_10_yz)

    # print(z_list)

    fig = plt.figure()
    plt.plot(z_list,gini_score_cdm_sn_05_xy,label="CDM 5%")
    plt.plot(z_list,gini_score_cdm_sn_10_xy,label="CDM 10%")
    plt.plot(z_list,gini_score_wdm_sn_05_xy,label="WDM 5%")
    plt.plot(z_list,gini_score_wdm_sn_10_xy,label="WDM 10%")
    plt.xscale('log')
    # plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel("Redshift")
    plt.ylabel("Gini score")
    fig.savefig("./gini_star_history.png",dpi=400)
    plt.close()

def gini_history(folder,out_folder):
    snap_list = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    a_list = [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1.0]
    
    num_bins=50
    
    gini_dust(folder,snap_list,a_list,num_bins,out_folder)
    # gini_gas(folder,snap_list,a_list,num_bins)
    # gini_star(folder,snap_list,a_list,num_bins)

if __name__ =='__main__':
    
    # folder = 'N2048_L65_sd3490'
    # out_folder = 'N2048_L65_sd234920/rr_out'
    folder = 'N2048_L65_sd46371'
    out_folder = './plots/gini/N2048_L65_sd46371'
    print(f"Plotting Gini score for {folder}\n")
    if not os.path.exists(f'{out_folder}'):
        os.makedirs(f'{out_folder}')
    
    gini_history(folder,out_folder)
    print()