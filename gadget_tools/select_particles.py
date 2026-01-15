import numpy as np
import h5py


LITTLEH    = 0.6688
UNITMASS   = 1.0e10
GASTYPE    = 0 
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5

def read_data(sim_directory,i_file):
    snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    print(f"Reading snapshot {snap_directory}")

    haloinfo_fname     = f'/fof_subhalo_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname

    snap_data     = h5py.File(snap_directory, 'r')
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    # Halo properties
    # Divide by LITTLEH to change to physical units
    subh_mass  = np.array(haloinfo_data['Subhalo']['SubhaloMass'], dtype=np.float64) * UNITMASS / LITTLEH
    subh_pos   = np.array(haloinfo_data['Subhalo']['SubhaloPos'], dtype=np.float64) 
    subh_rad   = np.array(haloinfo_data['Subhalo']['SubhaloHalfmassRad'], dtype=np.float64)
    halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH
    halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
    halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64)
    group_len = np.array(haloinfo_data['Group']['GroupLen'], dtype=np.float64)
    group_len_type = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.float64)
    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0] #Select largest non-contaminated halo / resimulation target

    # print(f"main halo ID = {mass_mask[halo_mainID]}")
    # print(f"main halo mass = {np.round(halo_M200c[mass_mask[halo_mainID]]/1e12,4)} e12 h^-1 M_sun")
    # print(f"Main halo pos = {halo_pos[mass_mask][halo_mainID,0]*1000,halo_pos[mass_mask][halo_mainID,1]*1000,halo_pos[mass_mask][halo_mainID,2]*1000}")
    print(f"M_200 = {np.round(halo_M200c[mass_mask[halo_mainID]]/1e12 ,5)} e12 M_sun")
    print(f"R_200 = {np.round(halo_R200c[mass_mask[halo_mainID]]*1000 ,5)} Kpc")

    # print(f"x_cent = {np.round(halo_pos[mass_mask][halo_mainID,0]*1000,6)}")
    # print(f"y_cent = {np.round(halo_pos[mass_mask][halo_mainID,1]*1000,6)}")
    # print(f"z_cent = {np.round(halo_pos[mass_mask][halo_mainID,2]*1000,6)}")

    #Get the snapshot data that might be useful
    main_halo_pos = halo_pos[mass_mask][halo_mainID]
    print(main_halo_pos)
    star_pos = np.array(snap_data[f'PartType{STARTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH
    star_mass = np.array(snap_data[f'PartType{STARTYPE}']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH

    gas_pos = np.array(snap_data[f'PartType{GASTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH
    gas_mass = np.array(snap_data[f'PartType{GASTYPE}']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH

    hdm_pos = np.array(snap_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH
    hdm_mass = np.array(snap_data[f'PartType{HIGHDMTYPE}']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH


    star_pos_dists = np.sqrt((star_pos[:,0]-main_halo_pos[0])**2+(star_pos[:,1]-main_halo_pos[1])**2+(star_pos[:,2]-main_halo_pos[2])**2)
    star_pos = star_pos[star_pos_dists<=halo_R200c[mass_mask[halo_mainID]]]
    star_mass = star_mass[star_pos_dists<=halo_R200c[mass_mask[halo_mainID]]]

    gas_pos_dists = np.sqrt((gas_pos[:,0]-main_halo_pos[0])**2+(gas_pos[:,1]-main_halo_pos[1])**2+(gas_pos[:,2]-main_halo_pos[2])**2)
    gas_pos = gas_pos[gas_pos_dists<=halo_R200c[mass_mask[halo_mainID]]]
    gas_mass = gas_mass[gas_pos_dists<=halo_R200c[mass_mask[halo_mainID]]]

    hdm_pos_dists = np.sqrt((hdm_pos[:,0]-main_halo_pos[0])**2+(hdm_pos[:,1]-main_halo_pos[1])**2+(hdm_pos[:,2]-main_halo_pos[2])**2)
    hdm_pos  = hdm_pos_dists[hdm_pos_dists<=halo_R200c[mass_mask[halo_mainID]]]
    hdm_mass = hdm_mass[hdm_pos_dists<=halo_R200c[mass_mask[halo_mainID]]]

    # print(len(star_pos))
    print(f"gas mass in r200     = {np.round(np.sum(gas_mass)/1e10,5)} e10 M_sun")
    print(f"stellar mass in r200 = {np.round(np.sum(star_mass)/1e10,5)} e10 M_sun")
    print(f"dm mass in r200      = {np.round(np.sum(hdm_mass)/1e12,5)} e12 M_sun")

    print(f"& {np.round(halo_M200c[mass_mask[halo_mainID]]/1e12 ,5)} & {np.round(np.sum(hdm_mass)/1e12,6)} & {np.round(np.sum(gas_mass)/1e10,5)} & {np.round(np.sum(star_mass)/1e10,5)} & {np.round(halo_R200c[mass_mask[halo_mainID]]*1000 ,5)} ")
    # gas_ids = np.array(snap_data[f'PartType{GASTYPE}']['ParticleIDs'], dtype=np.float64)
    
    # gas_vel = np.array(snap_data[f'PartType{GASTYPE}']['Velocities'], dtype=np.float64)

    # high_dm_ids = np.array(snap_data[f'PartType{HIGHDMTYPE}']['ParticleIDs'], dtype=np.float64)
    
    # high_dm_vel = np.array(snap_data[f'PartType{HIGHDMTYPE}']['Velocities'], dtype=np.float64)

    # # star_ids = np.array(snap_data[f'PartType{STARTYPE}']['ParticleIDs'], dtype=np.float64)
    
    # # star_vel = np.array(snap_data[f'PartType{STARTYPE}']['Velocities'], dtype=np.float64)

    # #Get the distance of each particle from the main halo
    # dist_from_mainhalo = (high_dm_pos[:,0]-halo_pos[mass_mask[halo_mainID],0])**2+(high_dm_pos[:,1]-halo_pos[mass_mask[halo_mainID],1])**2+(high_dm_pos[:,2]-halo_pos[mass_mask[halo_mainID],2])**2
    # dist_from_mainhalo = np.sqrt(dist_from_mainhalo)

    # #Select all particles within 5*R200 of the halo - not sure how many you'd really need
    # dm_select = np.where(dist_from_mainhalo<=1.2*halo_R200c[mass_mask[halo_mainID]])
    # dm_particle_vels = high_dm_vel[dm_select]
    # # print(f'Main halo group_len = {group_len[mass_mask[halo_mainID]]}')
    # # print(f'fraction of particles compared to main halo = {len(dm_particle_vels)/group_len_type[mass_mask[halo_mainID],1]}')
    # # print(f'Selected {len(dm_particle_vels)} dark matter particles in the main halo out of {len(high_dm_pos)} total particles')
    # # print(f'Dark matter average velocity = {np.mean(dm_particle_vels)} km/s')

if __name__=="__main__":
    folder = 'N2048_L65_sd46371'
    
    cdm_05 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_005/'
    cdm_10 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'
    
    wdm_05 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
    wdm_10 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'

    # i_file = 26
    # files = np.linspace(10,20,11,dtype=int)
    # print(files)
    # for iter,i_file in enumerate(files):
    #     print(i_file)
    #     read_data(sim_directory,i_file)
    print("--- cdm_sn_005 ---")
    read_data(cdm_05,26)
    print("--- cdm_sn_010 ---")
    read_data(cdm_10,26)
    print("--- wdm_sn_005 ---")
    read_data(wdm_05,26)
    print("--- wdm_sn_010 ---")
    read_data(wdm_10,26)
    