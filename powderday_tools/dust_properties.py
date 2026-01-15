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
    return np.load(f'./{folder}/{method}/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')

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

def dust_mass_hist(dm,SN_fac,label,folder,out_folder,num_bins,snap_num,make_plots=False):
    print(dm,'-',SN_fac)
    
    halo_pos,extent = set_center(folder,dm,SN_fac,snap_num)
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')
    a = halo_data['Header'].attrs['Time']
    z = halo_data['Header'].attrs['Redshift']
    # print(halo_pos)
    method = out_folder.split('/')[-2]
    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)
    
    pos_x = grid_physical_properties_data['gas_pos_x'] * 3.085678e18 / a# convert parsecs to cm, convert back later
    pos_y = grid_physical_properties_data['gas_pos_y'] * 3.085678e18 / a
    pos_z = grid_physical_properties_data['gas_pos_z'] * 3.085678e18 / a
    # print(np.min(grid_physical_properties_data['gas_pos_x'])/1e6*LITTLEH, np.max(grid_physical_properties_data['gas_pos_x'])/1e6*LITTLEH)
    # print((np.max(grid_physical_properties_data['gas_pos_x']) - np.min(grid_physical_properties_data['gas_pos_x']))/2e6*LITTLEH)
    extent = np.float32(extent)
    extent_r200 = extent
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
    
    
    # print(f'Total gas mass r200c = {np.round(np.sum(gas_mass)/UNITMASS,2)}e10 Msun')
    # print(f'Total gas mass 15kpc = {np.round(np.sum(gas_mass_15kpc)/UNITMASS,2)}e10 Msun\n')
    # total_dust_mass = np.sum(dust_mass)
    # print(f'Total dust mass = {np.round(total_dust_mass,5):.3e} Msun')
    # dust_sum = np.sum(dust_mass_rad200)

    # print(f'dust mass r200c = {np.round(dust_sum,5):.3e} Msun')
    # dust_sum = np.sum(dust_mass_rad200)/UNITMASS
    # print(f'{np.sum(dust_mass_25kpc):.3e}')
    print(f'Total dust mass       = {np.sum(dust_mass):.3e} Msun')
    print(f'Total dust mass r200c = {np.sum(dust_mass_rad200):.3e} Msun')
    print(f'Total dust mass 25kpc = {np.sum(dust_mass_25kpc):.3e} Msun')
    print(f'Total dust mass doughnut = {(np.sum(dust_mass_rad200)-np.sum(dust_mass_25kpc)):.3e} Msun\n')

    # print(f'Total log gas mass r200c ={np.round(np.log10(np.sum(gas_mass)),2)} Msun\n')
    
    # print(f'Total dust mass 25kpc = {np.round(np.sum(dust_mass_25kpc)/UNITMASS*1e3,2)}e7 Msun\n')
    
    bin_size = (set_plot_len(extent)/num_bins)
    data, edges_x, edges_y = np.histogram2d(set_plot_len(pos_x[pos_mask_rad200]),set_plot_len(pos_y[pos_mask_rad200]),bins=num_bins,weights=dust_mass[pos_mask_rad200])
    dust_dens = data/bin_size**2
    
    if make_plots:
        fig,ax = plt.subplots()
        
        cmap = cmr.torch
        
        # print(f'extent = {set_plot_len(extent)}, num bins = {num_bins}, bin size = {(bin_size)}')
        # bin_size = (set_plot_len(extent_15kpc)/num_bins)
        
        ### Only out to R200c
        
        
        ### All dust
        # data, edges_x, edges_y = np.histogram2d(set_plot_len(pos_x)/1e3,set_plot_len(pos_y)/1e3,bins=num_bins*2,weights=dust_mass)
        # data, edges_x, edges_y = np.histogram2d(set_plot_len(pos_x[pos_mask_15kpc])/1e3,set_plot_len(pos_y[pos_mask_15kpc])/1e3,bins=num_bins,weights=dust_mass_15kpc)
        
        
        limits=(np.min(edges_x)/1e3,np.max(edges_x)/1e3,np.min(edges_y)/1e3,np.max(edges_y)/1e3)
        # limits=(-extent/extent_r200,extent/extent_r200,-extent/extent_r200,extent/extent_r200)
        # limits=(set_plot_len(-extent_15kpc),set_plot_len(extent_15kpc),set_plot_len(-extent_15kpc),set_plot_len(extent_15kpc))
        # limits=(-extent_15kpc/extent_r200,extent_15kpc/extent_r200,-extent_15kpc/extent_r200,extent_15kpc/extent_r200)
        
        vmin, vmax = np.min(dust_dens[dust_dens>0]),np.max(dust_dens)
        # vmin, vmax = 1e-1,1e7
        im = plt.imshow(np.transpose(dust_dens),extent=limits,origin='lower',norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),cmap=cmap)

        # snap_data, haloinfo_data, z = get_sim_data(folder,dm,SN_fac,snap_num)
        # group_len_star = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.int64)[0,4]
        # star_pos = snap_data['PartType4']['Coordinates'][0:group_len_star]*1000/LITTLEH
        # star_pos_x = ( star_pos[:,0] - set_plot_len(halo_pos[0]) ) / set_plot_len(extent_r200)
        # star_pos_y = ( star_pos[:,1] - set_plot_len(halo_pos[1]) ) / set_plot_len(extent_r200)
        # star_pos_z = ( star_pos[:,2] - set_plot_len(halo_pos[2]) ) / set_plot_len(extent_r200)
        # plt.scatter(star_pos_x, star_pos_y, c='lime', s=1, alpha=1, marker='.')
        # print(set_plot_len(extent_r200)/1e3*LITTLEH)
        
        print((set_plot_len(halo_pos[0])/1e3,set_plot_len(halo_pos[1])/1e3),set_plot_len(extent_r200)/1e3)
        ax.add_patch(plt.Circle((set_plot_len(halo_pos[0])/1e3,set_plot_len(halo_pos[1])/1e3),set_plot_len(extent_r200)/1e3, fill=False,color='red',alpha=1,lw=5))

        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        # plt.title(f'{label}')
        # t = plt.text(0.05,0.05,f"{label}",ha='left', va='bottom',transform=ax.transAxes,fontsize=50.0,weight='bold',color='white')
        # r = fig.canvas.get_renderer()
        # bb = t.get_window_extent(renderer=r)
        # ax.add_patch(Rectangle((-0.91,-0.9), bb.width/480,bb.height/400, linewidth=10, edgecolor='none', facecolor='gray',alpha=0.7))
        
        # plt.xlabel(r'R/R$_{200c}$')
        # plt.ylabel(r'R/R$_{200c}$')
        # plt.title(f'Snapshot = {snap_num}, Redshift = {np.round(z,4)}')
        plt.xlabel(r'x [Mpc]')
        plt.ylabel(r'y [Mpc]')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)


        plt.colorbar(im,label=r'Dust surface density [M$_\odot$ kpc$^{-2}$]',cax=cax)
        plt.tight_layout()

        fig.savefig(f'{out_folder}/dust_hist_{dm}_{SN_fac}_{method}_{snap_num}.png',dpi=400)


        plt.close()
        print(f'saved fig = {out_folder}/dust_hist_{dm}_{SN_fac}_{method}_{snap_num}.png')
    return dust_dens, dust_mass, dust_mass_rad200, dust_mass_25kpc

def plot_hist_dif(data_comp,model_name,data_base,base_name,folder,out_folder,dm,SN_fac,num_bins,snap_num):
    halo_pos,extent = set_center(folder,dm,SN_fac,snap_num)
    
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)
    extent_25kpc = 25 * 3.085678e21
    dif = (data_comp/data_base-1)*100
    dif[dif==inf]=100
    dif[dif==-inf]=-100
    
    extent = np.float32(extent)
    extent_r200 = extent
    extent = extent * 1.0
    limits=(-extent/extent_r200,extent/extent_r200,-extent/extent_r200,extent/extent_r200)
    # cmap = cmr.prinsenvlag
    cmap = cmr.fusion_r
    vmax_dif = np.max(dif)
    vmax_dif = 150
    
    # fig,ax = plt.subplots(layout="constrained")
    fig,ax = plt.subplots()

    # ax = fig.add_subplot()
    dif = np.transpose(dif)
    print(f"DIF shape = {dif.shape}")
    # xmin = (xmin - halo_pos[0])/(extent/2)
    # xmax = (xmax - halo_pos[0])/(extent/2)
    # ymin = (ymin - halo_pos[1])/(extent/2)
    # ymax = (ymax - halo_pos[1])/(extent/2)
    # extent = np.round([xmin,xmax,ymin,ymax],2) 
    # extent = [-2,2,-2,2]
    # extent = [-25,25,-25,25]

    # extent = [set_plot_len(xmin),set_plot_len(xmax),set_plot_len(ymin),set_plot_len(ymax)] # transform to plot units
    
    
    h = ax.imshow(dif, extent=limits,origin='lower',cmap=cmap,vmin=-vmax_dif,vmax=vmax_dif,aspect='auto')
    # ## WDM contour
    # cs = ax.contour(np.transpose(data_comp),levels=[1e1,1e3,1e5],extent=limits,colors=['r','b','k'],alpha=0.6,linestyles='dashed')
    # ## CDM contour
    # lb = ax.contour(np.transpose(data_base),levels=[1e1,1e3,1e5],extent=limits,colors=['r','b','k'],alpha=0.6)

    # ax.clabel(cs,"WDM",inline=True)
    # ax.clabel(lb,"CDM",inline=True)
    
    plt.colorbar(h, label=f'\% difference')#,fraction=0.046, pad=0.05)

    # plt.hist2d(data[0][:,0]/data_comp[0][:,0],bins=[data_comp[1],data_comp[2]])#,data=data[0]/data_comp[0])
    ax.set_xlabel(r'R/R$_{200c}$')
    ax.set_ylabel(r'R/R$_{200c}$')
    plt.title(f'{model_name} / {base_name} ')
    # plt.text(0.05,0.05,f"{model_name} / {base_name}",ha='left', va='bottom',transform=ax.transAxes,fontsize=40.0,fontweight='heavy',color='black')
    # plt.legend(["dashed - WDM","Solid - CDM"])

    fig.tight_layout()
    fig.savefig(f'{out_folder}/dust_hist_diff_{dm}_{SN_fac}.png',dpi=400)
    print(f' saved figure - {out_folder}dust_hist_diff_{dm}_{SN_fac}.png')
    plt.close()

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
        import matplotlib.patches as mpatches
        arr = mpatches.FancyArrowPatch((0.5*2.8, 10), (0.5*2.8, 100), arrowstyle='->,head_width=.5', mutation_scale=20,color='red',lw=4)
        ax.add_patch(arr)
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
           
def plot_scatter(gas_pos_x,gas_pos_y,gas_pos_z,pos_mask,dm,SN_fac,out_folder):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(gas_pos_x[pos_mask],gas_pos_y[pos_mask],gas_pos_z[pos_mask],s=0.01,alpha=0.6,zorder=2)
    ax.set_aspect('auto')
    # fig.tight_layout()
    fig.savefig(f'./plots/{out_folder}/part_scatter_{dm}_{SN_fac}.png',dpi=400)
    plt.close()


def write_dust_masses(file_name,outer_dust_mass_file_name, folder, method, total_dust_mass, r200_dust_mass, kpc25_dust_mass):
    # file_name = f'./simulation_dust_masses_snap_10.txt'
    dust_mass_r200_cdm_sn_005 = np.sum(r200_dust_mass[0])/UNITMASS * 1000.0
    dust_mass_r200_cdm_sn_010 = np.sum(r200_dust_mass[1])/UNITMASS * 1000.0
    dust_mass_r200_wdm_sn_005 = np.sum(r200_dust_mass[2])/UNITMASS * 1000.0
    dust_mass_r200_wdm_sn_010 = np.sum(r200_dust_mass[3])/UNITMASS * 1000.0

    dust_mass_total_cdm_sn_005 = np.sum(total_dust_mass[0])/UNITMASS * 1000.0
    dust_mass_total_cdm_sn_010 = np.sum(total_dust_mass[1])/UNITMASS * 1000.0
    dust_mass_total_wdm_sn_005 = np.sum(total_dust_mass[2])/UNITMASS * 1000.0
    dust_mass_total_wdm_sn_010 = np.sum(total_dust_mass[3])/UNITMASS * 1000.0

    dust_mass_doughnut_cdm_sn_005 = np.round((np.sum(r200_dust_mass[0])-np.sum(kpc25_dust_mass[0]))/UNITMASS * 1000.0, 2)
    dust_mass_doughnut_cdm_sn_010 = np.round((np.sum(r200_dust_mass[1])-np.sum(kpc25_dust_mass[1]))/UNITMASS * 1000.0, 2)
    dust_mass_doughnut_wdm_sn_005 = np.round((np.sum(r200_dust_mass[2])-np.sum(kpc25_dust_mass[2]))/UNITMASS * 1000.0, 2)
    dust_mass_doughnut_wdm_sn_010 = np.round((np.sum(r200_dust_mass[3])-np.sum(kpc25_dust_mass[3]))/UNITMASS * 1000.0, 2)

    inner_concentration_cdm_sn_005 = np.round((np.sum(kpc25_dust_mass[0])/np.sum(r200_dust_mass[0])*100), 2)
    inner_concentration_cdm_sn_010 = np.round((np.sum(kpc25_dust_mass[1])/np.sum(r200_dust_mass[1])*100), 2)
    inner_concentration_wdm_sn_005 = np.round((np.sum(kpc25_dust_mass[2])/np.sum(r200_dust_mass[2])*100), 2)
    inner_concentration_wdm_sn_010 = np.round((np.sum(kpc25_dust_mass[3])/np.sum(r200_dust_mass[3])*100), 2)

    # inner_concentration_cdm_sn_010 = (np.sum(kpc25_dust_mass[1])) / UNITMASS * 1000.0
    # inner_concentration_wdm_sn_005 = (np.sum(kpc25_dust_mass[2])) / UNITMASS * 1000.0
    # inner_concentration_cdm_sn_010 = np.round(inner_concentration_cdm_sn_010, 2)
    # inner_concentration_wdm_sn_005 = np.round(inner_concentration_wdm_sn_005, 2)

    outer_concentration_cdm_sn_005 = np.round(100-(np.sum(kpc25_dust_mass[0])/np.sum(r200_dust_mass[0])*100), 2)
    outer_concentration_cdm_sn_010 = np.round(100-(np.sum(kpc25_dust_mass[1])/np.sum(r200_dust_mass[1])*100), 2)
    outer_concentration_wdm_sn_005 = np.round(100-(np.sum(kpc25_dust_mass[2])/np.sum(r200_dust_mass[2])*100), 2)
    outer_concentration_wdm_sn_010 = np.round(100-(np.sum(kpc25_dust_mass[3])/np.sum(r200_dust_mass[3])*100), 2)

    with open(file_name, "a") as dust_mass_file:
        print(f"Creating file {dust_mass_file}...")
        
        #### COMMENT FOR NORMAL FILE // UNCOMMENT FOR LATEX FILE
        simulationn_header = f'Simulation -> {folder}'        
        ####
        
        ##### COMMENT FOR NOMRAL FILE // UNCOMMENT FOR LATEX FILE
        sim_id = folder.split('_')[-1]
        dust_mass_file.writelines(f'{str(sim_id)} & {str(np.round(dust_mass_r200_cdm_sn_010,2))} & {str(np.round(dust_mass_r200_wdm_sn_005,2))}  & {np.round(np.sum(dust_mass_r200_cdm_sn_010) / np.sum(dust_mass_r200_wdm_sn_005),3)}  \ \ \n')        
        
        #####

        #### UNCOMMENT FOR NORMAL FILE
        # if method == 'dtm':
        #     dust_mass_file.writelines(f'Simulation -> {folder} \n')
        #     dust_mass_file.writelines(f'\n')
        # dust_mass_file.writelines(("Method -> ", method))
        # dust_mass_file.writelines('\n')
        # dust_mass_file.writelines('Total Mass\n')
        # dust_mass_file.writelines(("CDM SN 5%  ->  ", str(np.round(dust_mass_total_cdm_sn_005,5)),'e7 Msun\n'))
        # dust_mass_file.writelines(("CDM SN 10% ->  ", str(np.round(dust_mass_total_cdm_sn_010,5)),'e7 Msun\n'))
        # dust_mass_file.writelines(("WDM SN 5%  ->  ", str(np.round(dust_mass_total_wdm_sn_005,5)),'e7 Msun\n'))
        # dust_mass_file.writelines(("WDM SN 10% ->  ", str(np.round(dust_mass_total_wdm_sn_010,5)),'e7 Msun\n'))
        # if np.sum(dust_mass_total_cdm_sn_010) / np.sum(dust_mass_total_wdm_sn_005) > 1.1:
        #     dust_mass_file.writelines('Works As Expected :) \n')
        #     dust_mass_file.writelines(f'CDM 10% has {np.round(np.sum(dust_mass_total_cdm_sn_010) / np.sum(dust_mass_total_wdm_sn_005),5)}x more dust \n')
        # elif np.sum(dust_mass_total_cdm_sn_010) / np.sum(dust_mass_total_wdm_sn_005) > 1.0:
        #     dust_mass_file.writelines('Inconclusive -_- \n')
        #     dust_mass_file.writelines(f'CDM 10% has {np.round(np.sum(dust_mass_total_cdm_sn_010) / np.sum(dust_mass_total_wdm_sn_005),5)}x more dust \n')
        # else:                                
        #     dust_mass_file.writelines("Doesn't Work As Expected :( \n")
        #     dust_mass_file.writelines(f'WDM 5% has {np.round(np.sum(dust_mass_total_wdm_sn_005)/np.sum(dust_mass_total_cdm_sn_010),5)}x more dust \n')
        


        #### UNCOMMENT FOR NORMAL FILE
        # dust_mass_file.writelines('\nMass in R_200c\n')
        # dust_mass_file.writelines(("CDM SN 5%  ->  ", str(np.round(dust_mass_r200_cdm_sn_005,5)),'e7 Msun\n'))
        # dust_mass_file.writelines(("CDM SN 10% ->  ", str(np.round(dust_mass_r200_cdm_sn_010,5)),'e7 Msun\n'))
        # dust_mass_file.writelines(("WDM SN 5%  ->  ", str(np.round(dust_mass_r200_wdm_sn_005,5)),'e7 Msun\n'))
        # dust_mass_file.writelines(("WDM SN 10% ->  ", str(np.round(dust_mass_r200_wdm_sn_010,5)),'e7 Msun\n'))
        
        # if np.sum(dust_mass_r200_cdm_sn_010) / np.sum(dust_mass_r200_wdm_sn_005) > 1.1:
        #     dust_mass_file.writelines('Works As Expected :) \n')
        #     dust_mass_file.writelines(f'CDM 10% has {np.round(np.sum(dust_mass_r200_cdm_sn_010) / np.sum(dust_mass_r200_wdm_sn_005),5)}x more dust \n')
        
        # elif np.sum(dust_mass_r200_cdm_sn_010) / np.sum(dust_mass_r200_wdm_sn_005) > 1.0:
        #     dust_mass_file.writelines('Inconclusive :| \n')
        #     dust_mass_file.writelines(f'CDM 10% has {np.round(np.sum(dust_mass_r200_cdm_sn_010) / np.sum(dust_mass_r200_wdm_sn_005),5)}x more dust \n')
        
        # else:                                
        #     dust_mass_file.writelines("Doesn't Work As Expected :( \n")
        #     dust_mass_file.writelines(f'WDM 5% has {np.round(np.sum(dust_mass_r200_wdm_sn_005)/np.sum(dust_mass_r200_cdm_sn_010),5)}x more dust \n')
        
        # dust_mass_file.writelines('\n')
        #####
        dust_mass_file.close()

    # with open('./simulation_dust_concentration.txt', "a") as dust_concentration_file:
    #     sim_id = folder.split('_')[-1]
    #     dust_concentration_file.writelines(f'{str(sim_id)} & {str((dust_concentration_cdm_sn_005))} \% & {str(dust_concentration_cdm_sn_010)} \% & {str(dust_concentration_wdm_sn_005)} \% & {str(dust_concentration_wdm_sn_010)} \%  \ \ \n')
    #     dust_concentration_file.close()

    with open(outer_dust_mass_file_name, "a") as dust_concentration_file:
        sim_id = folder.split('_')[-1]
        dust_concentration_file.writelines(f'{str(sim_id)} & {str((dust_mass_doughnut_cdm_sn_005))}    & {str(dust_mass_doughnut_cdm_sn_010)}    & {str(dust_mass_doughnut_wdm_sn_005)}    & {str(dust_mass_doughnut_wdm_sn_010)}    \ \ \n')
        # dust_concentration_file.writelines(f'{str(sim_id)} & {str(inner_concentration_cdm_sn_010)}\% & {str(outer_concentration_cdm_sn_010)}\% & {str(inner_concentration_wdm_sn_005)}\% & {str(outer_concentration_wdm_sn_005)}\%   \ \ \n')
        # dust_concentration_file.writelines('\n')
        dust_concentration_file.close()


if __name__ =='__main__':

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if (simulation_name.startswith(("N2048_L65_sd")) )]:#and not simulation_name.startswith(("N2048_L65_sd00","N2048_L65_sd01","N2048_L65_sd02","N2048_L65_sd03","N2048_L65_sd04","N2048_L65_sd05","N2048_L65_sd17")))]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    print(folder_list)

    method_list = ['dtm','rr','li_bf']#
    for method in method_list:
        print()
        file_name = f'./dust_mass_files/simulation_dust_masses_{method}.txt'
        with open(file_name, "w") as dust_mass_file:
            # dust_mass_file.writelines(f'        & CDM 5\% & CDM 10\% & WDM 5\% & WDM 10\%  \ \ \n')
            dust_mass_file.writelines(f'        & CDM 10\% & \% within R<25kpc & WDM 5\% & \% within R<25kpc \ \ \n')
            # dust_mass_file.close()

        # concentration_file_name = f'./simulation_dust_concentration.txt'
        outer_dust_mass_file_name = f'./dust_mass_files/simulation_dust_mass_concentration_{method}.txt'
        with open(outer_dust_mass_file_name, "w") as dust_concentration_file:
            dust_concentration_file.writelines(f'        & CDM 5\% & CDM 10\% & WDM 5\% & WDM 10\%  \ \ \n')
            # out_list = [f'./plots/{folder}/dtm/',f'./plots/{folder}/rr/',f'./plots/{folder}/li_bf/']    

        cdm_sn_05_dust_mass_array = []
        cdm_sn_10_dust_mass_array = []
        wdm_sn_05_dust_mass_array = []
        wdm_sn_10_dust_mass_array = []
        cdm_sn_010_concentration, wdm_sn_005_concentration = [], []

        for folder in folder_list:
            out_folder = f'./plots/{folder}/{method}/'
            # method = out_folder.split('/')[-2]
            print(f'Simulation -> {folder}')
            print(f"method     -> {method}")

            if not os.path.exists(f'./{out_folder}'):
                os.makedirs(f'./{out_folder}')        
                
            snap_list = np.linspace(10,26,17,dtype=int)

            # for snap_num in snap_list:
            snap_num = 26
            num_bins = 120

            #### No longer plotting histograms - I commented it

            dm='cdm'
            data_cdm_sn_05, total_dust_mass_cdm_sn_005, dust_mass_r200_cdm_sn_005, dust_mass_25kpc_cdm_sn_005 = dust_mass_hist(dm,'sn_005','CDM SN=5\%',folder,out_folder,num_bins,snap_num,make_plots=False)
            data_cdm_sn_10, total_dust_mass_cdm_sn_010, dust_mass_r200_cdm_sn_010, dust_mass_25kpc_cdm_sn_010 = dust_mass_hist(dm,'sn_010','CDM SN=10\%',folder,out_folder,num_bins,snap_num,make_plots=False)
            
            dm='wdm_3.5'
            data_wdm_sn_05, total_dust_mass_wdm_sn_005, dust_mass_r200_wdm_sn_005, dust_mass_25kpc_wdm_sn_005 = dust_mass_hist(dm,'sn_005','WDM SN=5\%',folder,out_folder,num_bins,snap_num,make_plots=False)
            data_wdm_sn_10, total_dust_mass_wdm_sn_010, dust_mass_r200_wdm_sn_010, dust_mass_25kpc_wdm_sn_010 = dust_mass_hist(dm,'sn_010','WDM SN=10\%',folder,out_folder,num_bins,snap_num,make_plots=False)
                
            total_dust_mass = [np.sum(total_dust_mass_cdm_sn_005), np.sum(total_dust_mass_cdm_sn_010), np.sum(total_dust_mass_wdm_sn_005), np.sum(total_dust_mass_wdm_sn_010)]
            # print(np.array(total_dust_mass)/UNITMASS * 1000.0)
            r200_dust_mass  = [dust_mass_r200_cdm_sn_005,  dust_mass_r200_cdm_sn_010,  dust_mass_r200_wdm_sn_005,  dust_mass_r200_wdm_sn_010]
            kpc25_dust_mass = [dust_mass_25kpc_cdm_sn_005, dust_mass_25kpc_cdm_sn_010, dust_mass_25kpc_wdm_sn_005, dust_mass_25kpc_wdm_sn_010]
            write_dust_masses(file_name, outer_dust_mass_file_name, folder, method, total_dust_mass, r200_dust_mass, kpc25_dust_mass)
            # plot_hist_dif(data_wdm_sn_05,'WDM SN 5\% ',data_cdm_sn_10,'CDM SN 10\%',folder,out_folder,'wdm_3.5','sn_010',num_bins,snap_num)

            cdm_sn_05_dust_mass_array.append(np.sum(dust_mass_r200_cdm_sn_005)/UNITMASS * 1000.0)
            cdm_sn_10_dust_mass_array.append(np.sum(dust_mass_r200_cdm_sn_010)/UNITMASS * 1000.0)
            wdm_sn_05_dust_mass_array.append(np.sum(dust_mass_r200_wdm_sn_005)/UNITMASS * 1000.0)
            wdm_sn_10_dust_mass_array.append(np.sum(dust_mass_r200_wdm_sn_010)/UNITMASS * 1000.0)
            cdm_sn_010_concentration.append((np.sum(dust_mass_25kpc_cdm_sn_010)/np.sum(dust_mass_r200_cdm_sn_010)*100))
            wdm_sn_005_concentration.append((np.sum(dust_mass_25kpc_wdm_sn_005)/np.sum(dust_mass_r200_wdm_sn_005)*100))
        
        cdm_sn_05_dust_mass_array = np.array(cdm_sn_05_dust_mass_array)
        cdm_sn_10_dust_mass_array = np.array(cdm_sn_10_dust_mass_array)
        wdm_sn_05_dust_mass_array = np.array(wdm_sn_05_dust_mass_array)
        wdm_sn_10_dust_mass_array = np.array(wdm_sn_10_dust_mass_array)

        cdm_sn_010_concentration = np.array(cdm_sn_010_concentration)
        wdm_sn_005_concentration = np.array(wdm_sn_005_concentration)
                
        
        mean_ratio_cdm_sn_10_to_wdm_sn_05 = np.mean(cdm_sn_10_dust_mass_array/wdm_sn_05_dust_mass_array)
        median_ratio_cdm_sn_10_to_wdm_sn_05 = np.median(cdm_sn_10_dust_mass_array/wdm_sn_05_dust_mass_array)

        mean_cdm_sn_010_concentration = np.mean(cdm_sn_010_concentration)
        mean_wdm_sn_005_concentration = np.mean(wdm_sn_005_concentration)

        median_cdm_sn_010_concentration = np.median(cdm_sn_010_concentration)
        median_wdm_sn_005_concentration = np.median(wdm_sn_005_concentration)

        print(f'Median CDM sn_005 dust = {np.median(cdm_sn_05_dust_mass_array)}e7 Msun')
        print(f'Median CDM sn_010 dust = {np.median(cdm_sn_10_dust_mass_array)}e7 Msun')
        
        print(f'Median WDM sn_005 dust = {np.median(wdm_sn_05_dust_mass_array)}e7 Msun')
        print(f'Median WDM sn_010 dust = {np.median(wdm_sn_10_dust_mass_array)}e7 Msun')
        
        with open(file_name, "a") as dust_mass_file:
            dust_mass_file.writelines(f'Median & {np.median(cdm_sn_10_dust_mass_array)} & {np.median(wdm_sn_05_dust_mass_array)} & {median_ratio_cdm_sn_10_to_wdm_sn_05} \ \ \n')
            dust_mass_file.writelines(f'Mean & {np.mean(cdm_sn_10_dust_mass_array)} & {np.mean(wdm_sn_05_dust_mass_array)} & {mean_ratio_cdm_sn_10_to_wdm_sn_05} \ \ ')
            dust_mass_file.close()
        with open(outer_dust_mass_file_name, "a") as dust_concentration_file:
            dust_concentration_file.writelines(f'Median & {median_cdm_sn_010_concentration} & {100-median_cdm_sn_010_concentration} & {median_wdm_sn_005_concentration} & {100-median_wdm_sn_005_concentration} \ \ \n')
            dust_concentration_file.writelines(f'Mean & {mean_cdm_sn_010_concentration} & {mean_wdm_sn_005_concentration} \ \ \n')
            dust_concentration_file.close()

    # check_converge(folder,'cdm','sn_010',snap_num)
    # plot_hist_dif(data_cdm_sn_05,'CDM SN 5%',data_cdm_sn_10,'CDM SN 10%',folder,out_folder,'cdm','sn_005',num_bins,snap_num)
    # plot_hist_dif(data_cdm_sn_15,'CDM SN 15%',data_cdm_sn_10,'CDM SN 10%',folder,out_folder,'cdm','sn_015',num_bins,halo_pos,extent)

    # plot_hist_dif(data_wdm_sn_05,'WDM SN 5%',data_wdm_sn_10,'WDM SN 10%',folder,out_folder,'wdm_3.5','sn_005',num_bins,snap_num)
    # plot_hist_dif(data_wdm_sn_15,'WDM SN 15%',data_wdm_sn_10,'WDM SN 10%',folder,out_folder,'wdm','sn_015',num_bins,halo_pos,extent)
    
    # plot_hist_dif(data_comp,model_name,data_base,base_name,folder,out_folder,dm,SN_fac,num_bins,snap_num)
          

    plt.close()

    # rad_dust_surf_dens('cdm','sn_005','CDM SN=5\%',folder,out_folder,snap_num,'blue',"-.")
    # rad_dust_surf_dens('cdm','sn_010','CDM SN=10\%',folder,out_folder,snap_num,'blue')
    
    # rad_dust_surf_dens('wdm_3.5','sn_005','WDM SN=5\%',folder,out_folder,snap_num,'orange',"-.")
    # rad_dust_surf_dens('wdm_3.5','sn_010','WDM SN=10\%',folder,out_folder,snap_num,'orange')
    
    # radial_dist_dust_dif(folder, out_folder)
