import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import h5py
import os
from tqdm import trange
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import bootstrap
from matplotlib import gridspec
import cmasher as cmr

# mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['font.size'] = 11

LITTLEH = 0.6688
UNITMASS = 1e10
KPCTOCM = 3.085678e21 
PCTOCM = 3.085678e18 
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

def get_z_and_afac(folder,snap_num,dm='cdm',SN_fac='sn_010'):

    # try:
    with h5py.File(f"/fred/oz217/aussing/N2048_L65_sd46371/cdm/zoom/output/sn_010/snapshot_{str(snap_num).zfill(3)}.hdf5") as f:
        z = f['Header'].attrs['Redshift']
        a_fac = f['Header'].attrs['Time']
        f.close

    return z,a_fac

def load_grid_data(folder,method,dm,SN_fac,snap_num):
    
    data = np.load(f'./{folder}/{method}/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    # data = np.load(f'./{folder}/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')
    return data

def set_center(folder,snap_num,dm,sn_fac):
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{sn_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')

    halo_pos   = np.array(halo_data['Group']['GroupPos'], dtype=np.float64) #/ LITTLEH

    halo_M200c = np.array(halo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS #/ LITTLEH
    halo_masstypes = np.array(halo_data['Group']['GroupMassType'], dtype=np.float64)  #/ LITTLEH

    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0] #Select largest non-contaminated halo / resimulation target

    R200c = np.array(halo_data['Group']['Group_R_Crit200'], dtype=np.float64) #/ LITTLEH

    snapshot = get_snapshot(folder,snap_num,dm,sn_fac)
    # unit_length = get_unit_len(snapshot)

    halo_pos = np.array(halo_pos[mass_mask][halo_mainID]) * 1000 # unit_length
    halo_rad = R200c[mass_mask][halo_mainID] * 1000 # unit_length
    # print(f'halo_mass = {halo_pos}')
    # print(f"R_200 = {halo_rad}")
    
    # extent = halo_rad*2
    extent = halo_rad
    # extent = halo_rad*0.2
    # extent = 25 * 3.085678e21
    return halo_pos,extent

def get_extent(halo_pos,extent):

    xmin,xmax  = halo_pos[0]-extent,halo_pos[0]+extent
    ymin,ymax  = halo_pos[1]-extent,halo_pos[1]+extent
    zmin,zmax  = halo_pos[2]-extent,halo_pos[2]+extent
    
    return xmin,xmax,ymin,ymax,zmin,zmax

def gini(input_data):

    flattened = np.sort(np.ravel(input_data))
    flattened = flattened[np.isfinite(flattened)]
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

def rotate_and_mask(halo_pos,extent,exclusion,dust_coords,roation_maxtrix,axis):
    
    # rotate halo pos
    halo_pos = halo_pos.T
    halo_pos_rot = np.matmul(roation_maxtrix,halo_pos) 
    halo_pos_rot = halo_pos_rot.T
    
    # rotate coordinates 
    dust_coords = dust_coords.T
    rot_pos = np.matmul(roation_maxtrix,dust_coords) 
    rot_pos = rot_pos.T
    
    # calculate distances
    dists_rot = np.sqrt((rot_pos[:,0] - halo_pos_rot[0])**2 + 
                        (rot_pos[:,1] - halo_pos_rot[1])**2 +
                        (rot_pos[:,2] - halo_pos_rot[2])**2) 
    
    if axis == 'z':
        dist_axis = np.sqrt((rot_pos[:,0] - halo_pos_rot[0])**2 + 
                            (rot_pos[:,1] - halo_pos_rot[1])**2) 
        
        dist_mask = (dists_rot<=extent) & (dist_axis>=exclusion)  

    if axis == 'y':
        dist_axis = np.sqrt((rot_pos[:,0] - halo_pos_rot[0])**2 +
                            (rot_pos[:,2] - halo_pos_rot[2])**2)
        
        dist_mask = (dists_rot<=extent) & (dist_axis>=exclusion) 

    if axis == 'x':
        dist_axis = np.sqrt((rot_pos[:,1] - halo_pos_rot[1])**2 +
                            (rot_pos[:,2] - halo_pos_rot[2])**2)
        
        dist_mask = (dists_rot<=extent) & (dist_axis>=exclusion) 
    
    
    rad_pos_x = rot_pos[:,0]#-halo_pos[0]
    rad_pos_y = rot_pos[:,1]#-halo_pos[1]
    rad_pos_z = rot_pos[:,2]#-halo_pos[2]

    masked_coords = np.column_stack((rad_pos_x,rad_pos_y,rad_pos_z))

    return masked_coords,dist_mask,halo_pos_rot

def remask_hists(data,halo_pos,extent,exclusion,num_bins,axis):
    x_dist,y_dist = data[1],data[2]
    x_width,y_width = np.ones_like(data[1]),np.ones_like(data[2])
    for x in range(num_bins):
        x_dist[x] = (x_dist[x]+x_dist[x+1])/2
        x_width[x]= np.abs(x_dist[x]-x_dist[x+1])

        y_dist[x] = (y_dist[x]+y_dist[x+1])/2
        y_width[x]= np.abs(y_dist[x]-y_dist[x+1])
    # print(x_width)
    x_centre, y_centre = x_dist[:-1],y_dist[:-1]
    bin_cent = np.zeros(shape=(num_bins,num_bins),dtype=np.float64)
    bin_size = np.zeros(shape=(num_bins,num_bins),dtype=np.float64)
    a,b=0,0
    for a in range(num_bins):
        for b in range(num_bins):
            if axis == 'xy':
                dist = np.sqrt((x_centre[a]-halo_pos[0])**2+(y_centre[b]-halo_pos[1])**2)
            elif axis == 'xz':
                dist = np.sqrt((x_centre[a]-halo_pos[0])**2+(y_centre[b]-halo_pos[2])**2)
            elif axis =='yz':
                dist = np.sqrt((x_centre[a]-halo_pos[1])**2+(y_centre[b]-halo_pos[2])**2)
            bin_cent[a,b] = dist
            bin_size[a,b] = x_width[a]*y_width[b]
    
    data[0][bin_cent>extent]=np.nan
    data[0][bin_cent<exclusion]=np.nan

    return data,bin_size

def dust_mass_hist(dm,SN_fac,folder,out_folder,num_bins,snap_num,rot_mat_x,rot_mat_y,rot_mat_z,i,exclusion,z,a_fac):    
    z = np.round(z,3)
    redshift = str(np.round(z,1)).split('.')
    method = out_folder.split('/')[-2]
    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)
    
    halo_pos,extent = set_center(folder,snap_num,dm,SN_fac)
    
    halo_pos = np.array(halo_pos) 
    extent = extent 
    dust_pos_x = grid_physical_properties_data['gas_pos_x'] / 1000 / a_fac * LITTLEH #* PCTOCM
    dust_pos_y = grid_physical_properties_data['gas_pos_y'] / 1000 / a_fac * LITTLEH #* PCTOCM
    dust_pos_z = grid_physical_properties_data['gas_pos_z'] / 1000 / a_fac * LITTLEH #* PCTOCM

    dust_mass = grid_physical_properties_data['particle_dustmass']
    
    dust_coords = np.column_stack((dust_pos_x,dust_pos_y,dust_pos_z))


    rot_pos_xy,mask_xy,halo_pos_xy = rotate_and_mask(halo_pos,extent,exclusion,dust_coords,rot_mat_z,'z')
    xy_coords = rot_pos_xy[mask_xy]
    dust_mass_xy = dust_mass[mask_xy]

    rot_pos_xz,mask_xz,halo_pos_xz = rotate_and_mask(halo_pos,extent,exclusion,dust_coords,rot_mat_y,'y')
    xz_coords = rot_pos_xz[mask_xz]
    dust_mass_xz = dust_mass[mask_xz]

    rot_pos_yz,mask_yz,halo_pos_yz = rotate_and_mask(halo_pos,extent,exclusion,dust_coords,rot_mat_x,'x')
    yz_coords = rot_pos_yz[mask_yz]
    dust_mass_yz = dust_mass[mask_yz]
    
    bin_size = 2.5 # most used binning
    # bin_size = 1.5 # binning based on softening
    num_bins = int(extent/bin_size) # NOTE extent changes with redshift, number of bins changes, same relative to other simulaitons 

    data_xy = np.histogram2d(xy_coords[:,0],xy_coords[:,1],bins=num_bins,weights=dust_mass_xy)
    data_xy,bin_size_xy = remask_hists(data_xy,halo_pos_xy,extent,exclusion,num_bins,'xy')
    # data_xy[0] = np.array(data_xy[0])/np.array(bin_size_xy)
    
    data_xz = np.histogram2d(xz_coords[:,0],xz_coords[:,2],bins=num_bins,weights=dust_mass_xz)
    data_xz,bin_size_xz = remask_hists(data_xz,halo_pos_xz,extent,exclusion,num_bins,'xz')
    # data_xz[0] = data_xz[0]/bin_size_xz

    data_yz = np.histogram2d(yz_coords[:,1],yz_coords[:,2],bins=num_bins,weights=dust_mass_yz)
    data_yz,bin_size_yz = remask_hists(data_yz,halo_pos_yz,extent,exclusion,num_bins,'yz')
    # data_yz[0] = data_yz[0]/bin_size_yz
    
    return data_xy, data_xz, data_yz

def gini_dust(dm,sn,folder,snap_list,num_bins,out_folder,rotations,softening):
    # print("Calculating dust gini coefficient")
    # out_folder = np.array(out_folder)
    method = out_folder.split('/')[-2]
    z_list = []
    gini_score_test_xy, gini_score_cdm_sn_10_xy = [],[]

    gini_score_test_xz, gini_score_cdm_sn_10_xz = [],[]

    gini_score_test_yz, gini_score_cdm_sn_10_yz = [],[]
    
    
    num_rotations = rotations # has to be odd number
    # np.random()
    rotation_radian_list = np.linspace(0,2.*np.pi,num_rotations,endpoint=False)
    rotation_axis = np.zeros(num_rotations)
    softening = softening #kpc

    # for i_snap in trange(len(snap_list)):
    for i_snap in range(len(snap_list)):
        
        score_test_xy, score_cdm_sn_10_xy = [],[]
        score_test_xz, score_cdm_sn_10_xz = [],[]
        score_test_yz, score_cdm_sn_10_yz = [],[]

        snap_num = snap_list[i_snap]
        z,a_fac = get_z_and_afac(folder,snap_num,dm='cdm',SN_fac='sn_010')
        z_list.append(z)
        # print(" ")
        # print(f"Snapshot = {snap_num}, Redshift = {np.round(z,2)}\n")

        for i in range(len(rotation_radian_list)):
        # for i in trange(len(rotation_radian_list)):
            
            rot_mat_x = rotation_matrix(rotation_radian_list[i],rotation_axis[i],rotation_axis[i]) # Apply to yz
            rot_mat_y = rotation_matrix(rotation_axis[i],rotation_radian_list[i],rotation_axis[i]) # Apply to xz
            rot_mat_z = rotation_matrix(rotation_axis[i],rotation_axis[i],rotation_radian_list[i]) # Apply to xy
            
            data_test_xy, data_test_xz, data_test_yz = dust_mass_hist(dm,sn,folder,out_folder,num_bins,snap_num,rot_mat_x,rot_mat_y,rot_mat_z,i,softening,z,a_fac)
            data_cdm_sn_10_xy, data_cdm_sn_10_xz, data_cdm_sn_10_yz = dust_mass_hist('cdm','sn_010',folder,out_folder,num_bins,snap_num,rot_mat_x,rot_mat_y,rot_mat_z,i,softening,z,a_fac)
            

            score_test_xy.append(gini(data_test_xy[0]))
            score_cdm_sn_10_xy.append(gini(data_cdm_sn_10_xy[0]))
            
            score_test_xz.append(gini(data_test_xz[0]))
            score_cdm_sn_10_xz.append(gini(data_cdm_sn_10_xz[0]))
            
            score_test_yz.append(gini(data_test_yz[0]))
            score_cdm_sn_10_yz.append(gini(data_cdm_sn_10_yz[0]))
        
        gini_score_test_xy.append((score_test_xy))
        gini_score_cdm_sn_10_xy.append((score_cdm_sn_10_xy))
        
        gini_score_test_xz.append((score_test_xz))
        gini_score_cdm_sn_10_xz.append((score_cdm_sn_10_xz))

        gini_score_test_yz.append((score_test_yz))
        gini_score_cdm_sn_10_yz.append((score_cdm_sn_10_yz))

    # print(gini_score_test_xy)
    # print(gini_score_test_xz)
    # print(gini_score_test_yz)
    gini_test = np.column_stack((gini_score_test_xy,gini_score_test_xz,gini_score_test_yz))
    gini_cdm_10 = np.column_stack((gini_score_cdm_sn_10_xy,gini_score_cdm_sn_10_xz,gini_score_cdm_sn_10_yz))

    gini_test_median    = np.median(gini_test,axis=1)
    gini_cdm_10_median  = np.median(gini_cdm_10,axis=1)


    z_list = np.round(z_list,2)
    a_list = 1/(1+np.array(z_list))
    return a_list, z_list, gini_test_median, (gini_test_median/gini_cdm_10_median-1)*100

def gini_history(folder,out_folder,dm,sn):
    # snap_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    snap_list = [26]
    num_bins = 80
    rotations = 1 # normally 25 for full combined analysis
    # softening_list = [25.0,30.0]
    softening_list = [25.0] # normally 25 for combined analysis
    for softening in softening_list:
        a_list, z_list, gini_test_median, gini_test_median_diff = gini_dust(dm,sn,folder,snap_list,num_bins,out_folder,rotations,softening)
    return a_list, z_list, gini_test_median, gini_test_median_diff

if __name__ =='__main__':
    
    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if (simulation_name.startswith("N2048_L65_sd"))]:# and not simulation_name.startswith("N2048_L65_sd1"))]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(len(folder_list),'\n',folder_list)

    sn_list = ['sn_005']#,'sn_010'
    dm_list = ['wdm_3.5']#'cdm',
    for dm in dm_list :
        print(dm)
        for sn in sn_list:
            print(sn)
            # if dm == 'cdm' and sn == 'sn_010':
            #     print('Skipping')
            #     continue
            
            # out_list = [f'./test_plots/gini/{folder}/dtm/',f'./plots/gini/{folder}/rr/',f'./plots/gini/{folder}/li_bf/']    
            method_list = ['dtm']#,'rr','li_bf']#
            for method in method_list:
                print(method)
                fig, ax = plt.subplots()
                # method = out_folder.split('/')[-2]
                all_sims_gini_median = []
                all_sims_gini_median_difference = []
                low_list = []
                high_list = []
                for folder in folder_list:
                    # print()
                    # print(f'Simulation -> {folder}')
                    out_folder = f'./test_plots/gini/{folder}/{method}/'
                    a_list, z_list, gini_test_median, gini_test_median_diff = gini_history(folder,out_folder,dm,sn)
                    all_sims_gini_median.append(gini_test_median)
                    if gini_test_median_diff > 1.1:
                        # print(f'Simulation {folder} is high - {np.round(gini_test_median_diff,3)}')
                        high_list.append(folder) 
                    if gini_test_median_diff < 0.9:
                        # print(f'Simulation {folder} is low - {np.round(gini_test_median_diff,3)}') 
                        low_list.append(folder)       
                    all_sims_gini_median_difference.append(gini_test_median_diff)
                    # print(np.array(all_sims_gini_median_difference)) 
                all_sims_gini_median_difference = np.array(all_sims_gini_median_difference)    
                print(np.max(all_sims_gini_median_difference), folder_list[np.array(all_sims_gini_median_difference).argmax()])  
                print(np.min(all_sims_gini_median_difference), folder_list[np.array(all_sims_gini_median_difference).argmin()])  


                
                
                print(f'\nSims with higher WDM concentration -> \n {np.array(high_list)}\n')
                print(f'Sims with lower WDM concentration -> \n {np.array(low_list)}]\n')
                print((np.argsort(all_sims_gini_median_difference[:,0])))
                for i in range(0,24):
                    print(folder_list[np.argsort(all_sims_gini_median_difference[:,0])[i]],np.sort(all_sims_gini_median_difference[:,0])[i]) #  
                # all_sims_gini_median = np.median(np.array(all_sims_gini_median),axis=0) 
                # all_sims_gini_median_difference = np.median(np.array(all_sims_gini_median_difference),axis=0)
                
      
                
                # gini_test_btstrp_low = []
                # gini_test_btstrp_high = []
                # for i in range(all_sims_gini_median_difference.shape[1]):
                #     diff_data = (all_sims_gini_median_difference[:,i],)
                #     # btstrp_data = bootstrap(data,np.median,confidence_level=0.68,random_state=1,method='percentile')
                #     btstrp_diff_data = bootstrap(diff_data,np.mean,confidence_level=0.99,random_state=1,method='percentile')
                #     gini_test_btstrp_low.append(btstrp_diff_data.confidence_interval[0])
                #     gini_test_btstrp_high.append(btstrp_diff_data.confidence_interval[1])      
                          
                