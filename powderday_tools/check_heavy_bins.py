import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
from tqdm import trange
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from scipy.stats import bootstrap
from matplotlib import gridspec

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = [16,5]

LITTLEH = 0.6688
UNITMASS = 1e10
UNIT_LENGTH_FOR_PLOTS = 'Kpc'

def get_unit_len(snapshot):
    unit_length = snapshot["Parameters"].attrs['UnitLength_in_cm']
    
    return unit_length

def set_plot_len(data):
    if UNIT_LENGTH_FOR_PLOTS == 'Mpc':
        data = data/3.085678e24
    elif UNIT_LENGTH_FOR_PLOTS == 'Kpc':
        data = data/3.085678e21
    elif UNIT_LENGTH_FOR_PLOTS == 'pc':
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
    # if method =='rr_out':
    #     data = np.load(f'./{folder}/rr_out/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    # elif method =='li_bf':    
    #     data = np.load(f'./{folder}/li_bf_out/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    # elif method =='dtm':
    #     data = np.load(f'./{folder}/dtm/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    return data

def set_center(folder,snap_num,dm,sn_fac):
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')

    halo_pos   = np.array(halo_data['Group']['GroupPos'], dtype=np.float64) #/LITTLEH
    halo_M200c = np.array(halo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS#/ LITTLEH
    halo_masstypes = np.array(halo_data['Group']['GroupMassType'], dtype=np.float64) #/ LITTLEH

    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0] #Select largest non-contaminated halo / resimulation target

    # halo_pos = [halo_pos[mass_mask][halo_mainID,0],halo_pos[mass_mask][halo_mainID,1],halo_pos[mass_mask][halo_mainID,2]]
    R200c = np.array(halo_data['Group']['Group_R_Crit200'], dtype=np.float64) #/ LITTLEH

    snapshot = get_snapshot(folder,snap_num,dm,sn_fac)
    unit_length = get_unit_len(snapshot)

    halo_pos = np.array(halo_pos[mass_mask][halo_mainID]) * unit_length
    halo_rad = R200c[mass_mask][halo_mainID] * unit_length
    
    # extent = halo_rad*2
    extent = halo_rad
    # extent = 25 * 3.085678e21
    return halo_pos,extent

def get_extent(halo_pos,extent):

    xmin,xmax  = halo_pos[0]-extent,halo_pos[0]+extent
    ymin,ymax  = halo_pos[1]-extent,halo_pos[1]+extent
    zmin,zmax  = halo_pos[2]-extent,halo_pos[2]+extent
    
    return xmin,xmax,ymin,ymax,zmin,zmax

def gini(input_data):
    # print(input_data.shape)
    flattened = np.sort(np.ravel(input_data,order='C'))
    flattened = flattened[0:-10]
    # print(f"flattened = {flattened.shape}")
    npix = np.size(flattened)
    normalization = np.abs(np.mean(flattened)) * npix * (npix - 1)
    kernel = (2.0 * np.arange(1, npix + 1) - npix - 1) * np.abs(flattened)

    return np.sum(kernel) / normalization

def rotation_matrix(phi,omega,theta):
    u_x = np.cos(theta) * np.cos(omega)
    u_y = np.cos(theta) * np.sin(omega)
    u_z = -np.sin(theta)
    
    v_x = -np.cos(phi) * np.sin(omega) + np.sin(phi) * np.sin(theta) * np.cos(omega)
    v_y = np.cos(phi) * np.cos(omega) + np.sin(phi) * np.sin(theta) * np.sin(omega)
    v_z = np.sin(phi) * np.cos(theta)

    w_x = np.sin(phi) * np.sin(omega) + np.cos(phi) * np.sin(theta) * np.cos(omega)
    w_y = -np.sin(phi) * np.cos(omega) + np.cos(phi) * np.sin(theta) * np.sin(omega)
    w_z = np.cos(phi) * np.cos(theta)

    rot_mat = [[u_x,v_x,w_x],
               [u_y,v_y,w_y],
               [u_z,v_z,w_z]]

    return np.array(rot_mat)

def dust_mass_hist(dm,SN_fac,folder,out_folder,num_bins,snap_num,a_fac,rot_mat_x,rot_mat_y,rot_mat_z,i):
    # print(dm,'-',SN_fac, " Snap - ", snap_num)
    method = out_folder.split('/')[-2]
    print(f'File - ./{folder}/{method}/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)
    
    halo_pos,extent = set_center(folder,snap_num,dm,SN_fac)
    halo_pos = np.array(halo_pos) * a_fac
    extent = extent * a_fac
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)
    
    dust_pos_x = grid_physical_properties_data['gas_pos_x'] * LITTLEH * 3.085678e18
    dust_pos_y = grid_physical_properties_data['gas_pos_y'] * LITTLEH * 3.085678e18
    dust_pos_z = grid_physical_properties_data['gas_pos_z'] * LITTLEH * 3.085678e18

    x_mask = (dust_pos_x>=xmin) & (dust_pos_x<=xmax)
    y_mask = (dust_pos_y>=ymin) & (dust_pos_y<=ymax)
    z_mask = (dust_pos_z>=zmin) & (dust_pos_z<=zmax)

    pos_mask = x_mask & y_mask & z_mask
    rad_pos_x = dust_pos_x[pos_mask]-halo_pos[0]
    rad_pos_y = dust_pos_y[pos_mask]-halo_pos[1]
    rad_pos_z = dust_pos_z[pos_mask]-halo_pos[2]

    rad_pos = np.column_stack((rad_pos_x,rad_pos_y,rad_pos_z))
    rot_pos_xy = np.matmul(rad_pos,rot_mat_z) # rotate around z
    rot_pos_xz = np.matmul(rad_pos,rot_mat_y) # rotate around y
    rot_pos_yz = np.matmul(rad_pos,rot_mat_x) # rotate around x

    bin_size = (extent/num_bins)

    # print(f'extent = {set_plot_len(extent)}, num bins = {num_bins}, bin size = {set_plot_len(bin_size)}')
    dust_mass = grid_physical_properties_data['particle_dustmass']
    dust_mass = dust_mass[pos_mask]
    # print(f'sum dust mass = {"{:e}".format( np.sum(dust_mass))}')
    print((f'max dust mass = {"{:e}".format( np.max(dust_mass[dust_mass>0]))}'))
    print((f'min dust mass = {"{:e}".format( np.min(dust_mass[dust_mass>0]))}\n'))
    # data_xy = plt.hist2d(set_plot_len(rot_pos_xy[:,0]),set_plot_len(rot_pos_xy[:,1]),bins=num_bins,data=dust_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_xy = np.histogram2d(set_plot_len(rot_pos_xy[:,0]),set_plot_len(rot_pos_xy[:,1]),bins=num_bins,weights=dust_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig(f"./test_plots/rotated_{dm.split('.')[0]}_{SN_fac}_dust_xy_hist_snap_{snap_num}_{i}_{method}",dpi=400)
    plt.close()
    # data_xz = plt.hist2d(set_plot_len(rot_pos_xz[:,0]),set_plot_len(rot_pos_xz[:,2]),bins=num_bins,data=dust_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_xz = np.histogram2d(set_plot_len(rot_pos_xz[:,0]),set_plot_len(rot_pos_xz[:,2]),bins=num_bins,weights=dust_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    # plt.xlabel("x")
    # plt.ylabel("z")
    # plt.savefig(f"./test_plots/rotated_{dm.split('.')[0]}_{SN_fac}__dust_xz_hist_snap_{snap_num}_{i}_{method}",dpi=400)
    plt.close()
    # data_yz = plt.hist2d(set_plot_len(rot_pos_yz[:,1]),set_plot_len(rot_pos_yz[:,2]),bins=num_bins,data=dust_mass,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    data_yz = np.histogram2d(set_plot_len(rot_pos_yz[:,1]),set_plot_len(rot_pos_yz[:,2]),bins=num_bins,weights=dust_mass)#,norm=mpl.colors.LogNorm(vmin=1,vmax=1e4))
    # if snap_num==26:
    # plt.xlabel("y")
    # plt.ylabel("z")
    # plt.savefig(f"./test_plots/rotated_{dm.split('.')[0]}_{SN_fac}__dust_yz_hist_snap_{snap_num}_{i}_{method}",dpi=400)
    # plt.close()
    # plt.colorbar()
    # plt.savefig(f"./test_plots/rotated_dust_hist_snap_{snap_num}",dpi=400)
    plt.close()
    # print(data_xy[0].shape,data_xz[0].shape,data_yz[0].shape)
    # print(f'max bin mass = {"{:e}".format(np.max(data_xy[0]))}')#,(data_xz[0]),(data_yz[0]))
    return data_xy, data_xz, data_yz

def gini_dust(folder,snap_list,a_list,num_bins,out_folder):
    print("Calculating dust gini coefficient")
    z_list = []
    num_rotations = 3 # has to be odd number
    rotation_radian_list = np.linspace(0,2.*np.pi,num_rotations,endpoint=False)
    rotation_axis = np.zeros(num_rotations)
    num_bins = np.geomspace(1e-9,1e7,50)
    for i_snap in trange(len(snap_list)):

        snap_num = snap_list[i_snap]
        a_fac = a_list[i_snap]
        z = 1/a_fac-1
        z_list.append(z)
        print(" ")
        print(f"Snapshot = {snap_num}, Redshift = {z}\n")

        for i in range(len(rotation_radian_list)):
            # print('')
            # rot_mat = rotation_matrix(rotation_radian_list[i],rotation_radian_list[i],rotation_radian_list[i])
            rot_mat_x = rotation_matrix(rotation_radian_list[i],rotation_axis[i],rotation_axis[i]) # Apply to yz
            rot_mat_y = rotation_matrix(rotation_axis[i],rotation_radian_list[i],rotation_axis[i]) # Apply to xz
            rot_mat_z = rotation_matrix(rotation_axis[i],rotation_axis[i],rotation_radian_list[i]) # Apply to xy
            # print(np.round(rot_mat,2))
            dm='cdm'
            data_cdm_sn_05_xy,data_cdm_sn_05_xz,data_cdm_sn_05_yz = dust_mass_hist(dm,'sn_005',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_x,rot_mat_y,rot_mat_z,i)
            # data_cdm_sn_05_xy,data_cdm_sn_05_xz = dust_mass_hist(dm,'sn_005',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_y,rot_mat_z,i)
            data_cdm_sn_10_xy,data_cdm_sn_10_xz,data_cdm_sn_10_yz = dust_mass_hist(dm,'sn_010',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_x,rot_mat_y,rot_mat_z,i)
            # data_cdm_sn_10_xy,data_cdm_sn_10_xz = dust_mass_hist(dm,'sn_010',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_y,rot_mat_z,i)
            
            dm='wdm_3.5'
            data_wdm_sn_05_xy,data_wdm_sn_05_xz,data_wdm_sn_05_yz = dust_mass_hist(dm,'sn_005',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_x,rot_mat_y,rot_mat_z,i)
            # data_wdm_sn_05_xy,data_wdm_sn_05_xz = dust_mass_hist(dm,'sn_005',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_y,rot_mat_z,i)
            data_wdm_sn_10_xy,data_wdm_sn_10_xz,data_wdm_sn_10_yz = dust_mass_hist(dm,'sn_010',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_x,rot_mat_y,rot_mat_z,i)
            # data_wdm_sn_10_xy,data_wdm_sn_10_xz = dust_mass_hist(dm,'sn_010',folder,out_folder,num_bins,snap_num,a_fac,rot_mat_y,rot_mat_z,i)

            flattened_cdm_sn_05_xy = np.sort(np.ravel(data_cdm_sn_05_xy[0],order='C'))
            flattened_cdm_sn_05_xz = np.sort(np.ravel(data_cdm_sn_05_xz[0],order='C'))
            flattened_cdm_sn_05_yz = np.sort(np.ravel(data_cdm_sn_05_yz[0],order='C'))

            print(f'cdm sn=5% xy max = {"{:e}".format(np.max(flattened_cdm_sn_05_xy))}, min = {"{:e}".format(np.min(flattened_cdm_sn_05_xy[flattened_cdm_sn_05_xy>0]))}')
            print(f'cdm sn=5% xz max = {"{:e}".format(np.max(flattened_cdm_sn_05_xz))}, min = {"{:e}".format(np.min(flattened_cdm_sn_05_xz[flattened_cdm_sn_05_xz>0]))}')
            print(f'cdm sn=5% yz max = {"{:e}".format(np.max(flattened_cdm_sn_05_yz))}, min = {"{:e}".format(np.min(flattened_cdm_sn_05_yz[flattened_cdm_sn_05_yz>0]))}\n')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout = True)
            ax1.hist(flattened_cdm_sn_05_xy[flattened_cdm_sn_05_xy>0],bins=num_bins,log=True)
            ax1.set_title('XY')
            ax1.set_xscale('log')
            ax2.hist(flattened_cdm_sn_05_xz[flattened_cdm_sn_05_xz>0],bins=num_bins,log=True)
            ax2.set_title('XZ')
            ax2.set_xscale('log')
            ax3.hist(flattened_cdm_sn_05_yz[flattened_cdm_sn_05_yz>0],bins=num_bins,log=True)
            ax3.set_title('YZ')
            ax3.set_xscale('log')
            fig.savefig(f'./test_plots/dust_mass_hist_cdm_sn_05_snap_{snap_num}_rotation_{i}.png',dpi=400)
            plt.close()

            flattened_cdm_sn_10_xy = np.sort(np.ravel(data_cdm_sn_10_xy[0],order='C'))
            flattened_cdm_sn_10_xz = np.sort(np.ravel(data_cdm_sn_10_xz[0],order='C'))
            flattened_cdm_sn_10_yz = np.sort(np.ravel(data_cdm_sn_10_yz[0],order='C'))
            print(f'cdm sn=10% xy max = {"{:e}".format(np.max(flattened_cdm_sn_10_xy))}, min = {"{:e}".format(np.min(flattened_cdm_sn_10_xy[flattened_cdm_sn_10_xy>0]))}')
            print(f'cdm sn=10% xz max = {"{:e}".format(np.max(flattened_cdm_sn_10_xz))}, min = {"{:e}".format(np.min(flattened_cdm_sn_10_xz[flattened_cdm_sn_10_xz>0]))}')
            print(f'cdm sn=10% yz max = {"{:e}".format(np.max(flattened_cdm_sn_10_yz))}, min = {"{:e}".format(np.min(flattened_cdm_sn_10_yz[flattened_cdm_sn_10_yz>0]))}\n')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout = True)
            ax1.hist(flattened_cdm_sn_10_xy[flattened_cdm_sn_10_xy>0],bins=num_bins,log=True)
            ax1.set_title('XY')
            ax1.set_xscale('log')
            ax2.hist(flattened_cdm_sn_10_xz[flattened_cdm_sn_10_xz>0],bins=num_bins,log=True)
            ax2.set_title('XZ')
            ax2.set_xscale('log')
            ax3.hist(flattened_cdm_sn_10_yz[flattened_cdm_sn_10_yz>0],bins=num_bins,log=True)
            ax3.set_title('YZ')
            ax3.set_xscale('log')
            fig.savefig(f'./test_plots/dust_mass_hist_cdm_sn_10_snap_{snap_num}_rotation_{i}.png',dpi=400)
            plt.close()
            
            flattened_wdm_sn_05_xy = np.sort(np.ravel(data_wdm_sn_05_xy[0],order='C'))
            flattened_wdm_sn_05_xz = np.sort(np.ravel(data_wdm_sn_05_xz[0],order='C'))
            flattened_wdm_sn_05_yz = np.sort(np.ravel(data_wdm_sn_05_yz[0],order='C'))
            print(f'wdm sn=5% xy max = {"{:e}".format(np.max(flattened_wdm_sn_05_xy))}, min = {"{:e}".format(np.min(flattened_wdm_sn_05_xy[flattened_wdm_sn_05_xy>0]))}')
            print(f'wdm sn=5% xz max = {"{:e}".format(np.max(flattened_wdm_sn_05_xz))}, min = {"{:e}".format(np.min(flattened_wdm_sn_05_xz[flattened_wdm_sn_05_xz>0]))}')
            print(f'wdm sn=5% yz max = {"{:e}".format(np.max(flattened_wdm_sn_05_yz))}, min = {"{:e}".format(np.min(flattened_wdm_sn_05_yz[flattened_wdm_sn_05_yz>0]))}\n')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout = True)
            ax1.hist(flattened_wdm_sn_05_xy[flattened_wdm_sn_05_xy>0],bins=num_bins,log=True)
            ax1.set_title('XY')
            ax1.set_xscale('log')
            ax2.hist(flattened_wdm_sn_05_xz[flattened_wdm_sn_05_xz>0],bins=num_bins,log=True)
            ax2.set_title('XZ')
            ax2.set_xscale('log')
            ax3.hist(flattened_wdm_sn_05_yz[flattened_wdm_sn_05_yz>0],bins=num_bins,log=True)
            ax3.set_title('YZ')
            ax3.set_xscale('log')
            fig.savefig(f'./test_plots/dust_mass_hist_wdm_sn_05_snap_{snap_num}_rotation_{i}.png',dpi=400)
            plt.close()

            flattened_wdm_sn_10_xy = np.sort(np.ravel(data_wdm_sn_10_xy[0],order='C'))
            flattened_wdm_sn_10_xz = np.sort(np.ravel(data_wdm_sn_10_xz[0],order='C'))
            flattened_wdm_sn_10_yz = np.sort(np.ravel(data_wdm_sn_10_yz[0],order='C'))
            print(f'wdm sn=10% xy max = {"{:e}".format(np.max(flattened_wdm_sn_10_xy))}, min = {"{:e}".format(np.min(flattened_wdm_sn_10_xy[flattened_wdm_sn_10_xy>0]))}')
            print(f'wdm sn=10% xz max = {"{:e}".format(np.max(flattened_wdm_sn_10_xz))}, min = {"{:e}".format(np.min(flattened_wdm_sn_10_xz[flattened_wdm_sn_10_xz>0]))}')
            print(f'wdm sn=10% yz max = {"{:e}".format(np.max(flattened_wdm_sn_10_yz))}, min = {"{:e}".format(np.min(flattened_wdm_sn_10_yz[flattened_wdm_sn_10_yz>0]))}\n')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout = True)
            ax1.hist(flattened_wdm_sn_10_xy[flattened_wdm_sn_10_xy>0],bins=num_bins,log=True)
            ax1.set_title('XY')
            ax1.set_xscale('log')
            ax2.hist(flattened_wdm_sn_10_xz[flattened_wdm_sn_10_xz>0],bins=num_bins,log=True)
            ax2.set_title('XZ')
            ax2.set_xscale('log')
            ax3.hist(flattened_wdm_sn_10_yz[flattened_wdm_sn_10_yz>0],bins=num_bins,log=True)
            ax3.set_title('YZ')
            ax3.set_xscale('log')
            fig.savefig(f'./test_plots/dust_mass_hist_wdm_sn_10_snap_{snap_num}_rotation_{i}.png',dpi=400)
            plt.close()

def gini_history(folder,out_folder):
    # snap_list = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    snap_list = [26]
    # a_list = [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1.0]
    a_list = [1.0]
    
    num_bins=50
    
    gini_dust(folder,snap_list,a_list,num_bins,out_folder)

if __name__ =='__main__':
    
    # folder = 'N2048_L65_sd34920'
    # out_folder = './plots/gini/N2048_L65_sd34920'
    
    folder = 'N2048_L65_sd46371'
    # out_folder = './plots/gini/N2048_L65_sd46371'
    out_folder = './plots/N2048_L65_sd46371/dtm/'
    # out_folder = './plots/N2048_L65_sd46371/rr_out/'
    # out_folder = './plots/N2048_L65_sd46371/li_bf/'

    # folder = 'N2048_L65_sd57839'
    # out_folder = './plots/gini/N2048_L65_sd57839'
    method = out_folder.split('/')[-2]
    print(f"method = {method}")
    print(f"Plotting Gini score for {folder}\n")
    
    if not os.path.exists(f'{folder}'):
        raise Exception("Folder not found")
    if not os.path.exists(f'{out_folder}'):
        os.makedirs(f'{out_folder}')
    
    gini_history(folder,out_folder)
    print()