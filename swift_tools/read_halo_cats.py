import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import sys
import cmasher as cmr
sys.path.insert(0, '/fred/oz381/cpower/analysistools')
import merger_tree_tools as mtt
import snapshot_tools as st
import halo_tools as ht
import galaxy_tools as gt
import analysis_tools as at

plt.style.use('/home/aussing/sty.mplstyle')
cmap = cmr.torch

UNITMASS = 1e10
LITTLEH = 0.6688
DUST_TO_METALS_RATIO = 0.4

def read_halo_catalogue(simulation_location, snap_number):
    
    halo_properties = h5py.File(f'{simulation_location}/halos/snap_{str(snap_number).zfill(4)}.VELOCIraptor.properties.0', 'r')
    halo_group_info = h5py.File(f'{simulation_location}/halos/snap_{str(snap_number).zfill(4)}.VELOCIraptor.catalog_groups.0', 'r')
    group_particles = h5py.File(f'{simulation_location}/halos/snap_{str(snap_number).zfill(4)}.VELOCIraptor.catalog_particles.0', 'r')
    # VR_subhalo_data_wdm_010 = h5py.File(f'{simulation_location}/snap_{str(snap_number).zfill(4)}.sublevels.properties', 'r')
    # VR_subhalo_info_wdm_010 = h5py.File(f'{simulation_location}/snap_{str(snap_number).zfill(4)}.sublevels.catalog_groups', 'r')
    return halo_properties, halo_group_info, group_particles

def find_MW_systems(M_200c, structure_type, min_mass=0.55e12,max_mass=2.62e12): ## Default set to Milky Way Masses
    # MW_mass_min = min_mass
    # MW_mass_max = max_mass
    
    MWA_index = np.where((M_200c > min_mass) & (M_200c < max_mass))[0]
    non_subhaloes = np.where(structure_type==10)[0]
    MWA_index  = np.intersect1d(MWA_index, non_subhaloes)
    return MWA_index

def read_snap(simulation_location, snap_number):
    return h5py.File(f'{simulation_location}/snap_{str(snap_number).zfill(4)}.hdf5')

def plot_hist_2d(particle_coords, particle_mass, label):
    data, edges_x, edges_y = np.histogram2d(particle_coords[:,0],particle_coords[:,1],bins=100,weights=particle_mass)

    limits=(np.min(edges_x),np.max(edges_x),np.min(edges_y),np.max(edges_y))
    # vmin, vmax = np.min(data[data>0]),np.max(data)
    vmin, vmax = 1,np.max(data)
    im = plt.imshow(np.transpose(data),extent=limits,origin='lower',norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax ))
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'{label}.png',dpi=200)
    plt.close()
    return

def get_dust_mass(snapshot,mask):
    gas_metalmass_fraction = np.array(snapshot['PartType0']['MetalMassFractions'], dtype=np.float64) 
    gas_mass               = np.array(snapshot['PartType0']['Masses'],             dtype=np.float64) * UNITMASS #/ LITTLEH

    # print(f'Average gas metal mass fraction = {np.mean(gas_metalmass_fraction[mask])}')

    dust_mass = gas_mass[mask]*gas_metalmass_fraction[mask]*DUST_TO_METALS_RATIO
    return dust_mass, np.mean(gas_metalmass_fraction[mask])

def get_particles(snapshot, halo_pos, halo_rad):

    gas_coords  = np.array(snapshot['PartType0']['Coordinates'], dtype=np.float64)
    gas_mass    = np.array(snapshot['PartType0']['Masses'], dtype=np.float64) * UNITMASS #/ LITTLEH

    # dm_coords   = np.array(snapshot['PartType1']['Coordinates'], dtype=np.float64)
    # dm_mass     = np.array(snapshot['PartType1']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH

    # star_coords = np.array(snapshot['PartType4']['Coordinates'], dtype=np.float64)
    # star_mass   = np.array(snapshot['PartType4']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH

    # sink_coords = np.array(snapshot['PartType5']['Coordinates'], dtype=np.float64)
    # sink_mass   = np.array(snapshot['PartType5']['SubgridMasses'], dtype=np.float64) * UNITMASS / LITTLEH

    gas_pos_minus_halo  = np.sqrt( (gas_coords[:,0]  - halo_pos[0])**2 + (gas_coords[:,1]  - halo_pos[1])**2 + (gas_coords[:,2]  - halo_pos[2])**2 )
    # dm_pos_minus_halo   = np.sqrt( (dm_coords[:,0]   - halo_pos[0])**2 + (dm_coords[:,1]   - halo_pos[1])**2 + (dm_coords[:,2]   - halo_pos[2])**2 )
    # star_pos_minus_halo = np.sqrt( (star_coords[:,0] - halo_pos[0])**2 + (star_coords[:,1] - halo_pos[1])**2 + (star_coords[:,2] - halo_pos[2])**2 )
    # sink_pos_minus_halo = np.sqrt( (sink_coords[:,0] - halo_pos[0])**2 + (sink_coords[:,1] - halo_pos[1])**2 + (sink_coords[:,2] - halo_pos[2])**2 )

    gas_mask  = (gas_pos_minus_halo  <= halo_rad)
    gas_mask_25kpc = (gas_pos_minus_halo  <= 0.025)
    # dm_mask   = (dm_pos_minus_halo   <= halo_rad)
    # star_mask = (star_pos_minus_halo <= halo_rad)
    # sink_mask = (sink_pos_minus_halo <= halo_rad)
    
    dust_mass, average_metallicity = get_dust_mass(snapshot,gas_mask)
    dust_mass_25kpc = []#get_dust_mass(snapshot,gas_mask_25kpc)
    # print(f'{np.sum(dust_mass):.2e}')

    new_gas_coords  = gas_coords[ gas_mask]
    new_gas_mass    = gas_mass[   gas_mask]
    
    new_dm_coords   = [] #dm_coords[  dm_mask]
    new_dm_mass     = [] #dm_mass[    dm_mask]
    
    new_star_coords = [] #star_coords[star_mask]
    new_star_mass   = [] #star_mass[  star_mask]
    
    new_sink_coords = [] #sink_coords[sink_mask]
    new_sink_mass   = [] #sink_mass[  sink_mask]
    
    # print(f'Gas Mass   = {np.sum(new_gas_mass):.2e} Msun')
    # print(f'DM Mass    = {np.sum(new_dm_mass):.2e} Msun')
    # print(f'Star Mass  = {np.sum(new_star_mass):.2e} Msun')
    # print(f'Sink Mass  = {np.sum(new_sink_mass):.2e} Msun\n')

    # print(f'Total Mass = {(np.sum(new_gas_mass)+np.sum(new_dm_mass)+np.sum(new_star_mass)+np.sum(new_sink_mass)):.2e} Msun\n')
    
    return new_gas_coords, new_gas_mass, dust_mass, dust_mass_25kpc, average_metallicity #, new_dm_coords, new_dm_mass,new_star_coords,new_star_mass, new_sink_coords, new_sink_mass


if __name__ == "__main__":
    
    ### Define path to files
    
    path_to_cdm_simulation = f'/fred/oz390/L25/N376/'
    path_to_wdm_simulation = f'/fred/oz381/cpower/L25N376_WDM0pt5/'
    snap_number = 31

    ### Get initial CDM and WDM properties

    cdm_halo_properties, cdm_halo_group_info, cdm_particles = read_halo_catalogue(path_to_cdm_simulation, snap_number)
    cdm_snapshot = read_snap(path_to_cdm_simulation, snap_number)
    cdm_gas_mass  = np.array(cdm_snapshot['PartType0']['Masses'],dtype=np.float64)[0] * UNITMASS #/ LITTLEH
    cdm_dm_mass   = np.array(cdm_snapshot['PartType1']['Masses'],dtype=np.float64)[0] * UNITMASS #/ LITTLEH
    # cdm_star_mass = np.array(cdm_snapshot['PartType4']['Masses'],dtype=np.float64)[0] * UNITMASS #/ LITTLEH

    cdm_mvir = np.array(cdm_halo_properties['Mvir'],dtype=np.float64) * UNITMASS #/ LITTLEH
    cdm_m200c = np.array(cdm_halo_properties['Mass_200crit'],dtype=np.float64) * UNITMASS #/ LITTLEH
    cdm_r200c = np.array(cdm_halo_properties['R_200crit'],dtype=np.float64) 
    cdm_structure_type = np.array(cdm_halo_properties['Structuretype'],dtype=np.float32) 
    cdm_x, cdm_y, cdm_z = np.array(cdm_halo_properties['Xc'],dtype=np.float64), np.array(cdm_halo_properties['Yc'],dtype=np.float64), np.array(cdm_halo_properties['Zc'],dtype=np.float64)

    wdm_halo_properties, wdm_halo_group_info, wdm_particles = read_halo_catalogue(path_to_wdm_simulation, snap_number)
    wdm_snapshot = read_snap(path_to_wdm_simulation, snap_number)

    wdm_mvir = np.array(wdm_halo_properties['Mvir'],dtype=np.float64) * UNITMASS #/ LITTLEH
    wdm_m200c = np.array(wdm_halo_properties['Mass_200crit'],dtype=np.float64) * UNITMASS #/ LITTLEH
    wdm_r200c = np.array(wdm_halo_properties['R_200crit'],dtype=np.float64) 
    wdm_structure_type = np.array(wdm_halo_properties['Structuretype'],dtype=np.float32) 
    wdm_x, wdm_y, wdm_z = np.array(wdm_halo_properties['Xc'],dtype=np.float64), np.array(wdm_halo_properties['Yc'],dtype=np.float64), np.array(wdm_halo_properties['Zc'],dtype=np.float64)
    

    ### Define the mass range of haloes to select, get IDs

    min_mass, max_mass = 0.55e12, 2.62e12 # Milky Way range
    # min_mass, max_mass = 3e12, 1e14 
    cdm_MWA_index = find_MW_systems(cdm_m200c, cdm_structure_type, min_mass, max_mass)
    wdm_MWA_index = find_MW_systems(wdm_m200c, wdm_structure_type, min_mass, max_mass)
    
    print(cdm_MWA_index)
    print(wdm_MWA_index)

    cdm_halo_pos = np.array((cdm_x,cdm_y,cdm_z)).T
    wdm_halo_pos = np.array((wdm_x,wdm_y,wdm_z)).T

    new_cdm_index_list = []
    new_wdm_index_list = []

    ### Check that haloes in one list have a nearby neighbour, and then check that the neighbour is also in the original index list

    if len(cdm_MWA_index)>len(wdm_MWA_index):
        for i in range(len(cdm_MWA_index)):
            min_index = np.sqrt((wdm_halo_pos[:,0]         - cdm_halo_pos[cdm_MWA_index[i],0])**2 + (wdm_halo_pos[:,1]         - cdm_halo_pos[cdm_MWA_index[i],1])**2 + (wdm_halo_pos[:,2]         - cdm_halo_pos[cdm_MWA_index[i],2])**2).argmin()            
            distance  = np.sqrt((wdm_halo_pos[min_index,0] - cdm_halo_pos[cdm_MWA_index[i],0])**2 + (wdm_halo_pos[min_index,1] - cdm_halo_pos[cdm_MWA_index[i],1])**2 + (wdm_halo_pos[min_index,2] - cdm_halo_pos[cdm_MWA_index[i],2])**2)            
            if distance < 0.1 :
                # print(i, min_index, distance)
                # print(cdm_halo_pos[i])
                new_cdm_index_list.append(cdm_MWA_index[i])
                new_wdm_index_list.append(min_index)

        for i in range(len(new_wdm_index_list)):
            if (new_wdm_index_list[i] in wdm_MWA_index)==True:
                continue
            else:
                new_wdm_index_list[i]=np.nan
    else:
        for i in range(len(wdm_MWA_index)):
            min_index = np.sqrt((cdm_halo_pos[:,0]         - wdm_halo_pos[wdm_MWA_index[i],0])**2 + (cdm_halo_pos[:,1]         - wdm_halo_pos[wdm_MWA_index[i],1])**2 + (cdm_halo_pos[:,2]         - wdm_halo_pos[wdm_MWA_index[i],2])**2).argmin()
            distance  = np.sqrt((cdm_halo_pos[min_index,0] - wdm_halo_pos[wdm_MWA_index[i],0])**2 + (cdm_halo_pos[min_index,1] - wdm_halo_pos[wdm_MWA_index[i],1])**2 + (cdm_halo_pos[min_index,2] - wdm_halo_pos[wdm_MWA_index[i],2])**2)
            if distance < 0.1 :
                # print(wdm_MWA_index[i], min_index)#, distance)
                # print(wdm_halo_pos[i])
                new_wdm_index_list.append(wdm_MWA_index[i])
                new_cdm_index_list.append(min_index)
        for i in range(len(new_cdm_index_list)):
            if (new_cdm_index_list[i] in cdm_MWA_index)==True:
                continue
            else:
                new_cdm_index_list[i]=np.nan

    print(new_cdm_index_list)
    print(new_wdm_index_list)
    print(f'Num MWAs in both = {len(np.array(new_cdm_index_list)[np.array(new_cdm_index_list)>0])}')
    
    ### Define variables to calculate means later on

    count_cdm_more = 0
    count_wdm_more = 0
    count_inconclusive = 0
    bad_haloes = 0

    count_cdm_halo_more = 0
    count_wdm_halo_more = 0

    cdm_halo_mass_list, cdm_dust_mass_list, cdm_gas_mass_list = [], [], []
    wdm_halo_mass_list, wdm_dust_mass_list, wdm_gas_mass_list = [], [], []
    cdm_average_metallicity, wdm_average_metallicity = [], []

    ### Loop to calculate halo & dust properties

    for list_index in range(len(new_cdm_index_list)): #range(3):

        print(f'\nTesting Milky Way analogue number {list_index}')
        try:
            plt.close()
            cdm_milky_way_analogue_index = new_cdm_index_list[list_index]
            wdm_milky_way_analogue_index = new_wdm_index_list[list_index]

            cdm_halo_pos = np.array([cdm_x[cdm_milky_way_analogue_index],cdm_y[cdm_milky_way_analogue_index],cdm_z[cdm_milky_way_analogue_index]])
            cdm_halo_mass = cdm_m200c[cdm_milky_way_analogue_index]
            cdm_halo_rad_200c = cdm_r200c[cdm_milky_way_analogue_index]
            
            wdm_halo_pos = np.array([wdm_x[wdm_milky_way_analogue_index],wdm_y[wdm_milky_way_analogue_index],wdm_z[wdm_milky_way_analogue_index]])
            wdm_halo_mass = wdm_m200c[wdm_milky_way_analogue_index]
            wdm_halo_rad_200c = wdm_r200c[wdm_milky_way_analogue_index]
            
            print(f'CDM Halo M200c = {cdm_halo_mass:.2e} Msun')
            print(f'CDM Halo R200c = {cdm_halo_rad_200c*1000} kpc')
            print(cdm_halo_pos)
            
            # print(f'WDM Halo M200c = {wdm_halo_mass:.2e} Msun\n')

            cdm_halo_mass_list.append(cdm_halo_mass)
            wdm_halo_mass_list.append(wdm_halo_mass)


            cdm_gas_coords, cdm_gas_mass, cdm_dust_mass, cdm_dust_mass_25kpc, cdm_metallicity = get_particles(cdm_snapshot, cdm_halo_pos, cdm_halo_rad_200c)       
            
            gas_coords     = np.array(cdm_snapshot['PartType0']['Coordinates'], dtype=np.float64)
            cdm_part_index = st.select_particles(gas_coords,cdm_halo_pos,cdm_halo_rad_200c,geometry='spherical')#,periodic=True,scale_length=65)     
            
            cdm_gas_coods_select       = gas_coords[cdm_part_index]
            cdm_gas_mass               = np.array(cdm_snapshot['PartType0']['Masses'][cdm_part_index], dtype=np.float64) * UNITMASS # / LITTLEH
            cdm_gas_metalmass_fraction = np.array(cdm_snapshot['PartType0']['MetalMassFractions'][cdm_part_index], dtype=np.float64) 
            cdm_dust_mass_array        = cdm_gas_mass * cdm_gas_metalmass_fraction * DUST_TO_METALS_RATIO
            cdm_dust_mass_total        = np.sum(cdm_dust_mass_array)
            
            bin_size = 2.5 # 2.5kpc, radius needs to be converted
            num_bins = int(cdm_halo_rad_200c*1000/bin_size) # NOTE extent changes with redshift, number of bins changes, same relative to other simulaitons 
            
            hist, hist_bins_x, hist_bins_y = np.histogram2d(cdm_gas_coods_select[:,0], cdm_gas_coods_select[:,2], bins=num_bins,weights=cdm_dust_mass_array)
            bin_size_norm = (hist_bins_x[1]-hist_bins_x[0])*1000 * ((hist_bins_y[1]-hist_bins_y[0]))*1000
            
            plt.imshow(hist/bin_size_norm,origin='lower', extent=(np.min(hist_bins_x), np.max(hist_bins_x), np.min(hist_bins_y), np.max(hist_bins_y)),norm=mpl.colors.LogNorm(vmin=1e-2,vmax=1e6), cmap=cmap)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'./hists/xz_cdm_halo_{list_index}.png',dpi=100)
            plt.close()

            wdm_gas_coords, wdm_gas_mass, wdm_dust_mass, wdm_dust_mass_25kpc, wdm_metallicity = get_particles(wdm_snapshot, wdm_halo_pos, wdm_halo_rad_200c)
            cdm_average_metallicity.append(np.mean(cdm_gas_metalmass_fraction))
            wdm_average_metallicity.append(wdm_metallicity)

            wdm_part_index = st.select_particles(gas_coords,wdm_halo_pos,wdm_halo_rad_200c,geometry='spherical')#,periodic=True,scale_length=65)    spherical 
            
            wdm_gas_coods_select       = gas_coords[wdm_part_index]
            wdm_gas_mass               = np.array(wdm_snapshot['PartType0']['Masses'][wdm_part_index], dtype=np.float64) * UNITMASS #/ LITTLEH
            wdm_gas_metalmass_fraction = np.array(wdm_snapshot['PartType0']['MetalMassFractions'][wdm_part_index], dtype=np.float64) 
            wdm_dust_mass_array        = wdm_gas_mass * wdm_gas_metalmass_fraction * DUST_TO_METALS_RATIO
            wdm_dust_mass_total        = np.sum(wdm_dust_mass_array)
            
            
            num_bins = int(wdm_halo_rad_200c*1000/bin_size) # NOTE extent changes with redshift, number of bins changes, same relative to other simulaitons 
            
            hist, hist_bins_x, hist_bins_y = np.histogram2d(wdm_gas_coods_select[:,0], wdm_gas_coods_select[:,2], bins=num_bins,weights=wdm_dust_mass_array)
            bin_size_norm = (hist_bins_x[1]-hist_bins_x[0])*1000 * ((hist_bins_y[1]-hist_bins_y[0]))*1000
            plt.imshow(hist/bin_size_norm,origin='lower', extent=(np.min(hist_bins_x), np.max(hist_bins_x), np.min(hist_bins_y), np.max(hist_bins_y)),norm=mpl.colors.LogNorm(vmin=1e-2,vmax=1e6), cmap=cmap)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'./hists/xz_wdm_halo_{list_index}.png',dpi=100)

            cdm_gas_mass_list.append(np.sum(cdm_gas_mass))
            wdm_gas_mass_list.append(np.sum(wdm_gas_mass))

            cdm_dust_mass_list.append(np.sum(cdm_dust_mass))
            wdm_dust_mass_list.append(np.sum(wdm_dust_mass))
    
            if np.sum(cdm_dust_mass)/np.sum(wdm_dust_mass) >= 1:
                count_cdm_more = count_cdm_more+1
            elif np.sum(wdm_dust_mass)/np.sum(cdm_dust_mass) >=  1: #else:                                
                count_wdm_more = count_wdm_more+1
            else:  ## elif np.sum(cdm_dust_mass)/np.sum(wdm_dust_mass) >=  1: 
                count_inconclusive = count_inconclusive+1

            if np.sum(cdm_halo_mass) > np.sum(wdm_halo_mass):
                # print(np.sum(cdm_dust_mass)/np.sum(wdm_dust_mass))
                count_cdm_halo_more = count_cdm_halo_more+1         
            else:                                                
                count_wdm_halo_more = count_wdm_halo_more+1

        except:
            
            print(f"WDM halo didn't qualify as an MW analogue")
            bad_haloes = bad_haloes+1
            # print(f'CDM dust / WDM dust = {np.sum(cdm_dust_mass)/np.sum(wdm_dust_mass):.2e}')
    print()
    
    print(f"Bad haloes                           = {bad_haloes}")
    print(f"Galaxies with more CDM dust than WDM = {count_cdm_more}")
    print(f"Galaxies with more WDM dust than CDM = {count_wdm_more}")
    print(f"Galaxies inconclusive dust           = {count_inconclusive}")
    print()
    print(f"Galaxies with heavier CDM haloes     = {count_cdm_halo_more}")
    print(f"Galaxies with heavier WDM haloes     = {count_wdm_halo_more}\n")

    cdm_halo_mass_list = np.array(cdm_halo_mass_list)
    wdm_halo_mass_list = np.array(wdm_halo_mass_list)
    print(f'CDM mean halo mass    = {np.mean(cdm_halo_mass_list):.4}')
    print(f'WDM mean halo mass    = {np.mean(wdm_halo_mass_list):.4}\n')

    print(f'CDM median halo mass  = {np.median(cdm_halo_mass_list):.4}')
    print(f'WDM median halo mass  = {np.median(wdm_halo_mass_list):.4}\n')

    cdm_gas_mass_list = np.array(cdm_gas_mass_list)
    wdm_gas_mass_list = np.array(wdm_gas_mass_list)
    print(f'CDM mean gas mass     = {np.mean(cdm_gas_mass_list):.4}')
    print(f'WDM mean gas mass     = {np.mean(wdm_gas_mass_list):.4}\n')
 
    print(f'CDM median gas mass   = {np.median(cdm_gas_mass_list):.4}')
    print(f'WDM median gas mass   = {np.median(wdm_gas_mass_list):.4}\n')

    cdm_dust_mass_list = np.array(cdm_dust_mass_list)
    wdm_dust_mass_list = np.array(wdm_dust_mass_list)
    print(f'CDM mean dust mass    = {np.mean(cdm_dust_mass_list):.4}')
    print(f'WDM mean dust mass    = {np.mean(wdm_dust_mass_list):.4}\n')

    print(f'CDM median dust mass  = {np.median(cdm_dust_mass_list):.4}')
    print(f'WDM median dust mass  = {np.median(wdm_dust_mass_list):.4}\n')

    cdm_average_metallicity = np.array(cdm_average_metallicity)
    wdm_average_metallicity = np.array(wdm_average_metallicity)

    print(np.mean(cdm_average_metallicity))
    print(np.mean(wdm_average_metallicity))
