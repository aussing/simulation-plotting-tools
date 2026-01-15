from importlib.machinery import FileFinder
from re import T
import statistics
import numpy as np
import h5py 
import yt
from yt.utilities.cosmology import Cosmology as ytCosmology
import matplotlib.pyplot as plt
import scipy.stats as stat

plt.rcParams['text.usetex'] = True

# sim_directory = '/fred/oz004/aussing/gadget4_output/music/lcdm/second_run/DM_L100_N512'
sim_directory = '/fred/oz004/aussing/gadget4_output/music/wdm/small_box/DM_L100_N1024'
numfiles = 20

redshifts  = np.array([127, 0])
filenumber = np.arange(numfiles,numfiles)

LITTLEH   = 0.703
UNITMASS  = 1.0e10
GASTYPE = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5


z=np.zeros(numfiles)
gas_avg = np.zeros(numfiles)
star_avg = np.zeros(numfiles)
i_file = 0
snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
snap_directory = sim_directory + snap_fname

# haloinfo_fname     = f'/fof_subhalo_tab_{str(i_file).zfill(3)}.hdf5'
# haloinfo_directory = sim_directory + haloinfo_fname
# print(f"snap directory+name = {snap_directory}")

snap_data     = h5py.File(snap_directory, 'r')
# haloinfo_data = h5py.File(haloinfo_directory, 'r')
# halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH 
# halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
# halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
# halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.int32)
# halo_partlen = np.array(haloinfo_data['Group']['GroupLenType'],dtype=np.int32)
gas_metallicity = np.array(snap_data['PartType0']['Metallicity'],dtype=np.float64)
# star_metallicity = np.array(snap_data['PartType4']['Metallicity'],dtype=np.float64)
gas_density = np.array(snap_data['PartType0']['Density'],dtype=np.float64)   
        
bins = np.geomspace(np.min((gas_density)),np.max((gas_density)),num=15)
gas_dens_bins = np.digitize(gas_density,bins, right=True)

# mass_mask = np.argsort(halo_M200c)[::-1]
# halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]

# highdm_pos = np.array(snap_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH 
# lowdm_pos  = np.array(snap_data[f'PartType{LOWDMTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH 
# gas_pos    = np.array(snap_data[f'PartType{GASTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH 
# star_pos   = np.array(snap_data[f'PartType{STARTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH

# tot_pos  = np.concatenate((highdm_pos, lowdm_pos, gas_pos, star_pos), axis=0)
# target_halocenter = halo_pos[mass_mask][halo_mainID]
# target_R200c      = halo_R200c[mass_mask][halo_mainID]
# print(target_R200c)
sim_directory = '/fred/oz217/aussing/therm_test/'
i_file = 0
snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
snap_directory = sim_directory + snap_fname
snap_data     = h5py.File(snap_directory, 'r')
# print(snap_data['Header'].attrs['Redshift'])
high_dm_vel = np.array(snap_data['PartType1']['Velocities'],dtype=np.float64)
low_dm_vel = np.array(snap_data['PartType5']['Velocities'],dtype=np.float64)

all_vel_high = np.concatenate((high_dm_vel[:,0],high_dm_vel[:,1],high_dm_vel[:,2]))
all_vel_low = np.concatenate((low_dm_vel[:,0],low_dm_vel[:,1],low_dm_vel[:,2]))
modulus = np.sqrt(high_dm_vel[:,0]**2+high_dm_vel[:,1]**2+high_dm_vel[:,2]**2)
mod_hist_1,bins_1 = np.histogram(modulus,75)

# fig=plt.hist(modulus,100,histtype='step')
# fig=plt.hist(abs(all_vel_low),100,histtype='step')

sim_directory = '/fred/oz217/aussing/vel_disp_sims/cdm/zoom/'
snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
snap_directory = sim_directory + snap_fname
snap_data     = h5py.File(snap_directory, 'r')
z = (snap_data['Header'].attrs['Redshift'])
high_dm_vel = np.array(snap_data['PartType1']['Velocities'],dtype=np.float64)
low_dm_vel = np.array(snap_data['PartType5']['Velocities'],dtype=np.float64)

all_vel_high = np.concatenate((high_dm_vel[:,0],high_dm_vel[:,1],high_dm_vel[:,2]))
all_vel_low = np.concatenate((low_dm_vel[:,0],low_dm_vel[:,1],low_dm_vel[:,2]))
modulus = np.sqrt(high_dm_vel[:,0]**2+high_dm_vel[:,1]**2+high_dm_vel[:,2]**2)
mod_hist_2,bins_2 = np.histogram(all_vel_high,150)

# fig=plt.hist(modulus,100,histtype='step')
# fig=plt.hist(abs(all_vel_low),100,histtype='step')

fig = plt.bar(bins_2[:-1],mod_hist_2,width=np.diff(bins_2), align='edge')
# fig = plt.yscale('log')
fig = plt.xlabel('Absolute velocity (km/s)')
fig = plt.ylabel('Frequency')
# fig = plt.legend(('Kicked', 'Not kicked'))
fig = plt.title(f'Redshift = {z}')
fig = plt.savefig("Vel.png")

# print(stat.ks_2samp(mod_hist_2,mod_hist_1,alternative='greater'))

# for species in ['gas']:#, 'star',]:
#     if species == 'gas':
#         species_pos   = gas_pos
#         species_metal = gas_metallicity
#         species_color = 'purple'
#     elif species == 'star':
#         species_pos   = star_pos
#         species_metal = star_metallicity
#         species_color = 'r'
#     else:
#         raise NameError(f'Invalid species "{species}"!')

#     # print(f'Calculating profile for {species} species...')

#     # Select particles up to R200c
#     species_dist  = (species_pos[:,0] - target_halocenter[0])**2
#     species_dist += (species_pos[:,1] - target_halocenter[1])**2
#     species_dist += (species_pos[:,2] - target_halocenter[2])**2
#     species_dist  = np.sqrt(species_dist)
    
#     species_distmask, = np.where(species_dist < target_R200c*0.2)
#     halo_metals =   species_metal[species_distmask]#/(4/3 * np.pi * (target_R200c*0.2)**3)
#     z[i_file] = snap_data['Header'].attrs['Redshift']


#     test  = scipy.stats.binned_statistic(gas_density[species_distmask],halo_metals,statistic='mean',bins=bins)
#     bin_centre = []
#     for i in (test[i].shape):
#         bin_centre[i] = (test[1][i+1]-test[1][i]/2)
#     print(bin_centre)
    
# fig = plt.plot(bin_centre,test[0])
# fig = plt.yscale('log')
# fig = plt.savefig("metals.png")
