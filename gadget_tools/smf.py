import numpy as np
import matplotlib.pyplot as plt
import h5py
import py
import pynbody
import decimal
import os
from astropy.io import fits

plt.style.use('/home/aussing/sty.mplstyle')

LITTLEH    = 0.6688
UNITMASS   = 1.0e10
GASTYPE    = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5
G          = 4.3e-9# Mpc Msun^-1 (km/s)^2
I_FILE     = 23

def plot_smf_data():#ax):

    Baldry = np.array([
        [7.05, 1.3531e-01, 6.0741e-02],
        [7.15, 1.3474e-01, 6.0109e-02],
        [7.25, 2.0971e-01, 7.7965e-02],
        [7.35, 1.7161e-01, 3.1841e-02],
        [7.45, 2.1648e-01, 5.7832e-02],
        [7.55, 2.1645e-01, 3.9988e-02],
        [7.65, 2.0837e-01, 4.8713e-02],
        [7.75, 2.0402e-01, 7.0061e-02],
        [7.85, 1.5536e-01, 3.9182e-02],
        [7.95, 1.5232e-01, 2.6824e-02],
        [8.05, 1.5067e-01, 4.8824e-02],
        [8.15, 1.3032e-01, 2.1892e-02],
        [8.25, 1.2545e-01, 3.5526e-02],
        [8.35, 9.8472e-02, 2.7181e-02],
        [8.45, 8.7194e-02, 2.8345e-02],
        [8.55, 7.0758e-02, 2.0808e-02],
        [8.65, 5.8190e-02, 1.3359e-02],
        [8.75, 5.6057e-02, 1.3512e-02],
        [8.85, 5.1380e-02, 1.2815e-02],
        [8.95, 4.4206e-02, 9.6866e-03],
        [9.05, 4.1149e-02, 1.0169e-02],
        [9.15, 3.4959e-02, 6.7898e-03],
        [9.25, 3.3111e-02, 8.3704e-03],
        [9.35, 3.0138e-02, 4.7741e-03],
        [9.45, 2.6692e-02, 5.5029e-03],
        [9.55, 2.4656e-02, 4.4359e-03],
        [9.65, 2.2885e-02, 3.7915e-03],
        [9.75, 2.1849e-02, 3.9812e-03],
        [9.85, 2.0383e-02, 3.2930e-03],
        [9.95, 1.9929e-02, 2.9370e-03],
        [10.05, 1.8865e-02, 2.4624e-03],
        [10.15, 1.8136e-02, 2.5208e-03],
        [10.25, 1.7657e-02, 2.4217e-03],
        [10.35, 1.6616e-02, 2.2784e-03],
        [10.45, 1.6114e-02, 2.1783e-03],
        [10.55, 1.4366e-02, 1.8819e-03],
        [10.65, 1.2588e-02, 1.8249e-03],
        [10.75, 1.1372e-02, 1.4436e-03],
        [10.85, 9.1213e-03, 1.5816e-03],
        [10.95, 6.1125e-03, 9.6735e-04],
        [11.05, 4.3923e-03, 9.6254e-04],
        [11.15, 2.5463e-03, 5.0038e-04],
        [11.25, 1.4298e-03, 4.2816e-04],
        [11.35, 6.4867e-04, 1.6439e-04],
        [11.45, 2.8294e-04, 9.9799e-05],
        [11.55, 1.0617e-04, 4.9085e-05],
        [11.65, 3.2702e-05, 2.4546e-05],
        [11.75, 1.2571e-05, 1.2571e-05],
        [11.85, 8.4589e-06, 8.4589e-06],
        [11.95, 7.4764e-06, 7.4764e-06],
        ], dtype=np.float32)

    Baldry_xval = np.log10(10 ** Baldry[:, 0] / LITTLEH / LITTLEH)

    Baldry_yvalU = (Baldry[:, 1]+Baldry[:, 2]) * pow(LITTLEH, 3)
    Baldry_yvalL = (Baldry[:, 1]-Baldry[:, 2]) * pow(LITTLEH, 3)

    plt.fill_between(Baldry_xval, Baldry_yvalU, Baldry_yvalL,
                    facecolor='purple', alpha=0.25,
                    label='Baldry et al. 2008 (z=0.1)')

    # return ax

def get_sim_data(sim_directory):
    snap_fname     = f'/snapshot_{str(I_FILE).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(I_FILE).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = np.round((snap_data['Header'].attrs['Redshift']),2)
    return snap_data, haloinfo_data, z

def smf(sim_directory,label,folder):
    # print(f'\nPlotting HMF for {label}')
    snap_data, haloinfo_data, z = get_sim_data(sim_directory)

    halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) 
    halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
    halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS  / LITTLEH
    
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH
    stellar_mass = halo_masstypes[:,4]
    bin_width = 0.2
    max_stel_mass = np.log10(np.sort(np.unique(stellar_mass))[-1])+1e-1
    min_stel_mass = np.log10(np.sort(np.unique(stellar_mass))[1])-1e-1
    
    num_of_bins = int((max_stel_mass -min_stel_mass)/bin_width)
    print(num_of_bins)

    bin_range = np.linspace(min_stel_mass,max_stel_mass,num_of_bins)
    box_volume = snap_data['Header'].attrs['BoxSize']**3

    # mass_mask = np.argsort(halo_mass)[::-1]
    # halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]

    # clean_halos = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0]
    # grouplen_types = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.float64)
    # real_halos = np.where(grouplen_types[:,HIGHDMTYPE] > 2)[0]
    # true_halos = np.isin(clean_halos,real_halos)
    # print(f"main halo ID = {mass_mask[halo_mainID]}")
    # print(f"Main halo pos = {halo_pos[mass_mask[halo_mainID],0],halo_pos[mass_mask[halo_mainID],1],halo_pos[mass_mask[halo_mainID],2]}")
    # print(f'halo_mass = {np.round(halo_mass[mass_mask[halo_mainID]]/1e10,3)}e10 h^-1 Msun')
    # print(f"R_200 = {np.round(halo_R200c[mass_mask[halo_mainID]],2)} Mpc")
    # # Need to normalise by volume, to hard to determine in final snap, just use well defined cubic volume defined in t=0/z=127 snapshot
    # pos_data = h5py.File(sim_directory+'/snapshot_000.hdf5', 'r') 
    # highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64)
    # x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
    # y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
    # z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
    # box_volume=x_extent*y_extent*z_extent

    
    mass_func,bins_func = np.histogram(np.log10(stellar_mass),bin_range)
    # print(mass_func)
    for i in range(num_of_bins-1): #Change from bin edges to bin centres, remove last value in bins_func when plotting
        bins_func[i]=bins_func[i]+(bins_func[i+1]-bins_func[i])/2
    bins_func=bins_func[:-1]
    
    # Normalise the hmf by log bin width & box volume
    smf = mass_func/(bin_width*box_volume)
    # fig,ax = plt.subplots()
    if label=='cdm sn=10\%':
        plot_smf_data()
    plt.plot(bins_func,smf,label=f"{label} ")

    plt.yscale('log')
    plt.xlabel(r'log$_{10}M_{stars}$ [M$_{\odot}h^{-1}$]')
    plt.ylabel(r' $dn/d$log$_{10}M$ [Mpc$^{-3}$ $h^3$ dex$^{-1}$]')
    plt.title(f'Stellar Mass at Redshift = {np.round(z,2)}')
    plt.ylim((1e-5,5e-1))
    plt.legend()
       
    plt.savefig(f'{folder}/smf.png',dpi=400)

           
if __name__ == "__main__":

    folder = 'N2048_L65_sd46371'
    # folder = 'N2048_L65_sd57839'

    folder = 'N512_L65_sd12345'

    if not os.path.exists(f'./{folder}'):
        os.makedirs(f'./{folder}')

    # cdm_sn_005 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_005/'
    # cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'

    # wdm_sn_005 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
    # wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'


    cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/output/'
    wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm/output/sn_010'

    # smf(cdm_sn_005,'cdm sn=5%',folder)
    smf(cdm_sn_010,'cdm sn=10\%',folder)
    # smf(cdm_sn_015,'cdm sn=15%',folder)
    # smf(wdm_sn_005,'wdm sn=5%',folder)
    smf(wdm_sn_010,'wdm sn=10\%',folder)



    plt.close()
    