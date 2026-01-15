# from turtle import position
import numpy as np
import matplotlib.pyplot as plt
import h5py
import py
import pynbody
import decimal
import os
# from sklearn.preprocessing import binarize
# from sympy import threaded
plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['text.usetex'] = False
# plt.rcParams['font.size'] = 14

LITTLEH    = 0.6688#0.703
UNITMASS   = 1.0e10
GASTYPE    = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5
# i_file     = 26

UNIT_LENGTH_FOR_PLOTS = 'Mpc'


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

def get_sim_data(sim_directory,i_file):
    snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data      = h5py.File(snap_directory, 'r')
    unit_length    = get_unit_len(snap_data) 
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = np.round((snap_data['Header'].attrs['Redshift']),2)
    return snap_data, haloinfo_data, z, unit_length
    
def critical_density(h,OmegaR0,OmegaM0,OmegaK0,OmegaL0,z,comoving=False,hCorrect=False):
    '''Calculates critical density'''

    G  = 6.67430e-11     # in m^3 kg^-1 s^-2 = m kg^-1 (m/s)^2
    factor1 = 1e-6       # (m/s)^2 to (km/s)^2
    factor2 = 1.98855e30 # kg^-1 to Msun^-1
    factor3 = 1/3.086e22 # m to Mpc
    G  = G*factor1*factor2*factor3 #in Mpc Msun^-1 (km/s)^2

    H0 = 100 * h  # H0 = 100h (km/s) Mpc^-1 !!!!
    H2 = H0**2 *(OmegaR0*(1+z)**4 + OmegaM0*(1+z)**3 + OmegaK0*(1+z)**2 + OmegaL0)

    if comoving: # go from PHYSICAL critical density to COMOVING critical density. rho_phys = rho_comov/a**3 -->
                 # rho_comov = a**3 * rho_phys = (1/(1+z))**3 * rho_phys
        result = ((1/(1+z))**3)*(3*H2)/(8*np.pi*G)
    else:
        result = (3*H2)/(8*np.pi*G)

    if hCorrect: # in simulations: [rho] = h**2 * (Msun /Mpc**3). Turn this parameter ON so that rho_crit has the
                 # same units as the rho from simulations: rho_crit/h**2 --> [rho_crit] = h**2 (Msun /Mpc**3)
        result = (1/h**2)*result
    else:
        pass
    return result

def free_streaming_mass(mass, snap_data):
        dm_mass = mass
        h = snap_data['Parameters'].attrs['HubbleParam']
        z = (snap_data['Header'].attrs['Redshift'])
        print(z)
        Omega_matter = snap_data['Parameters'].attrs['Omega0']
        Omega_baryons = snap_data['Parameters'].attrs['OmegaBaryon']
        Omega_lambda = snap_data['Parameters'].attrs['OmegaLambda']
        Omega_DM = Omega_matter - Omega_baryons
        Omega_r = 0  # LCDM
        Omega_k = 0  # LCDM
        
        crit_density = critical_density(h,Omega_r,Omega_matter,Omega_k,Omega_lambda,z,comoving=True,hCorrect=True)
        # print(f"critical density = {crit_density}")
        rho_mean = crit_density * Omega_matter

        nu = 1.12 ## Viel 2005 / 1.0 for lovell 2014
        
        lambda_fs = 0.049 * (dm_mass)**(-1.11) * (Omega_DM/0.25)**(0.11) * (h/0.7)**1.22
        
        mass_fs = (4*np.pi)/3 * rho_mean * (lambda_fs/2)**3
        mass_hm = 2.7e3*mass_fs


        return lambda_fs, mass_fs, mass_hm

def analytic_halo_mass(sim_directory,folder):
    snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    haloinfo_fname     = f'/fof_subhalo_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    
    halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS
    z = (snap_data['Header'].attrs['Redshift'])

    s = pynbody.load(snap_directory)
    my_cosmology = pynbody.analysis.hmf.PowerSpectrumCAMB(s, filename='/home/aussing/.conda/envs/py38/lib/python3.8/site-packages/pynbody/analysis/CAMB_WMAP7')
    m, sig, dn_dlogm = pynbody.analysis.hmf.halo_mass_function(s, log_M_min=np.min(np.log10(halo_mass)-1e-6), log_M_max=12, delta_log_M=0.1, kern="ST", pspec=my_cosmology)
    plt.plot(np.log10(m),dn_dlogm,ls='--',color='k',label='ST WMAP7 fit')
    plt.yscale('log')
    plt.xlabel(r'log$_{10}M_{H}$ [$M_{\odot}$ ]')
    plt.ylabel(r'$dn/d$log$_{10}M_H$ [Mpc$^{-3}$ $h^3$ dex$^{-1}$]')
    plt.title(f'Halo Mass Function at Redshift = {np.round(z,2)}')
    plt.legend()
    plt.savefig(f'{folder}/Halo_mass_function_z_{z}.png',dpi=400)
    
    return

def halo_mass_funtion(sim_directory,label,folder,i_file,color,linestyle='-'):
    print(f'\nPlotting HMF for {label}')
    snap_data, haloinfo_data, z, unit_length = get_sim_data(sim_directory,i_file)
    print(sim_directory)
    halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) #/ LITTLEH
    halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
    halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    group_len   = np.array(haloinfo_data['Group']['GroupLen'], dtype=np.int32)
    
    bin_width = 0.2
    num_of_bins = np.int32((np.max(np.log10(halo_mass)+1e-4) - np.min(np.log10(halo_mass)-1e-4))/bin_width)
    bin_range = np.linspace(np.min(np.log10(halo_mass)-1e-4),np.max(np.log10(halo_mass)+1e-4),num_of_bins)
    
    box_volume = (snap_data['Header'].attrs['BoxSize'] * unit_length / LITTLEH )**3 
    
    try:
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH

        mass_mask = np.argsort(halo_mass)[::-1]
        halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
        # print(mass_mask)
        clean_halos = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0]
        grouplen_types = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.float64)
        real_halos = np.where(grouplen_types[:,HIGHDMTYPE] > 5)[0]
        true_halos = np.isin(clean_halos,real_halos)
        print(f"main halo ID = {mass_mask[halo_mainID]}")
        print(f"Main halo coods = {np.round(halo_pos[mass_mask[halo_mainID]],3)}")
        print(f'halo_mass = {np.round(halo_mass[mass_mask[halo_mainID]]/1e12,3)}e12 Msun')
        print(f"R_200 = {np.round(halo_R200c[mass_mask[halo_mainID]],2)*1000} kpc")
        # print(f"group len for main halo = {group_len[mass_mask[halo_mainID]]}")

        # Need to normalise by volume, to hard to determine in final snap, just use well defined cubic volume defined in t=0/z=127 snapshot
        pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
        highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64) * unit_length
        x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
        y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
        z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
        box_volume=x_extent*y_extent*z_extent
        # print(box_volume**(1/3))
        # clean_halos = np.isin(clean_halos,true_halos)
        
        mass_func,bins_func = np.histogram(np.log10(halo_mass[clean_halos[true_halos]]),bin_range)
        # mass_func,bins_func = np.histogram(np.log10(halo_mass[clean_halos]),bin_range)
        # print(len(grouplen_types[np.where(grouplen_types[:,HIGHDMTYPE] > 1)]))
        # print(len(grouplen_types[np.where(grouplen_types[:,HIGHDMTYPE] == 1)]))
        # test = np.mean(halo_mass[grouplen_types[np.where(grouplen_types[:,HIGHDMTYPE] == 1)]])
        # print(f'aaa =')
    except:
        mass_func,bins_func = np.histogram(np.log10(halo_mass),bin_range)
        # print(f"biggest halo mass = {np.round(np.max(halo_M200c)/UNITMASS,2)} e10")
        # print(f"Halo radius= {np.round(halo_R200c[np.where(halo_M200c==np.max(halo_M200c))[0]],2)} Mpc")
        # print(f"Halo particle number = {group_len[np.where(halo_M200c==np.max(halo_M200c))[0]]}")
    
    for i in range(num_of_bins-1): #Change from bin edges to bin centres, remove last value in bins_func when plotting
        bins_func[i]=bins_func[i]+(bins_func[i+1]-bins_func[i])/2
    bins_func=bins_func[:-1]
    
    # Normalise the hmf by log bin width & box volume
    hmf = mass_func/(bin_width*set_plot_len(set_plot_len(set_plot_len(box_volume))))
    plt.plot(bins_func,hmf,label=f"{str(label)} at z={z}",color=color, ls=linestyle)
    z = np.round(z,2)
    plt.yscale('log')
    plt.xlabel(r'log$_{10}M_{H}$ [$M_{\odot}$]')
    plt.ylabel(r'$dn/d$log$_{10}$M$_H$ '+f'[{UNIT_LENGTH_FOR_PLOTS}'+r'$^{-3}$  dex$^{-1}$]') #$h^3$
    # plt.title(f'Total Halo Mass Function at Redshift = {z}')
    plt.title(f'Total Halo Mass Function')
    plt.legend()
    plt.savefig(f'{folder}/Halo_mass_function_z_{z}.png',dpi=400,bbox_inches='tight')
    # plt.savefig(f'{folder}/Halo_mass_function_comb.png',dpi=400,bbox_inches='tight')

def cumulative_halo_mass(sim_directory,mass, label,folder,i_file,color,linestyle='-'):
    snap_data, haloinfo_data, z, unit_lenght = get_sim_data(sim_directory,i_file)

    halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
    # halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS 

    bin_width = 0.2
    num_of_bins = np.int32((np.max(np.log10(halo_mass)+1e-6) - np.min(np.log10(halo_mass)-1e-6))/bin_width)
    bin_range = np.linspace(np.min(np.log10(halo_mass)-1e-6),np.max(np.log10(halo_mass)+1e-6),num_of_bins)
    
    box_volume = (snap_data['Header'].attrs['BoxSize'])**3 
    try:
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.int32) / LITTLEH

        mass_mask = np.argsort(halo_mass)[::-1]
        clean_halos = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0]
        
        grouplen_types = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.float64)
        real_halos = np.where(grouplen_types[:,HIGHDMTYPE] > 5)[0]
        true_halos = np.isin(clean_halos,real_halos)
        # Need to normalise by volume, to hard to determine in final snap, just use well defined cubic volume defined in t=0/z=127 snapshot
        pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
        highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64)
        x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
        y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
        z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
        box_volume=x_extent*y_extent*z_extent
        
        halo_mass = halo_mass[clean_halos[true_halos]]
        ordered_halos = np.argsort(halo_mass)[::-1]
        cummass_func = np.cumsum(np.ones(halo_mass[ordered_halos].shape[0]))
        
    except:
        ordered_halos = np.argsort(halo_mass)[::-1]
        cummass_func = np.cumsum(np.ones(halo_mass[ordered_halos].shape[0]))
        
    
    # for i in range(num_of_bins-1): #Change from bin edges to bin centres, remove last value in bins_func when plotting
    #     bins_func[i]=bins_func[i]+(bins_func[i+1]-bins_func[i])/2
    # bins_func=bins_func[:-1]

    # if mass != 999:
    #     lambda_fs, mass_fs, mass_hm = free_streaming_mass(mass, snap_data)
    #     print(f"The free streaming length for {mass}Kev WDM is = {lambda_fs:.2E} Mpc/h")
    #     print(f"The free streaming halo mass for {mass}Kev WDM is = {mass_fs:.2E} M_sun/h")
    #     print(f"The half mode halo mass for {mass}Kev WDM is = {mass_hm:.2E} M_sun/h\n")

        # plt.axvline(mass_fs, color='k',ls='--')
        # plt.annotate("M_fs", (mass_fs+0.15*mass_fs,np.min(cummass_func/box_volume)),rotation=90)
        # plt.axvline(mass_hm, color='red',ls=':')
        # plt.annotate("M_hm", (mass_hm+0.15*mass_hm,np.min(cummass_func/box_volume)),rotation=90)
    
    plt.plot(halo_mass[ordered_halos],cummass_func,label=f"{label}",color=color, ls=linestyle)
    
    z = np.round(z,2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'M$_{H}$ [$M_{\odot}$]')
    # plt.ylabel(r'n $>$ M$_{200}$ [Mpc$^{-3}$ $h^3$]')
    plt.ylabel(r'n $>$ M$_{200}$')
    # plt.ylabel(r'n$>$M')
    # plt.title(f'Cumulative HMF at Redshift = {z}')
    plt.title(f'Cumulative HMF')
    plt.legend()
    plt.savefig(f'{folder}/Cumulative_hmf_z_{z}.png',dpi=400,bbox_inches='tight')
    # plt.savefig(f'{folder}/Cumulative_hmf_comb.png',dpi=400,bbox_inches='tight')
    print(f'Saving figure {folder}/Cumulative_hmf_z_{z}.png')

def R_fit(wdm_sim_directory,cdm_sim_directory,mass,label,folder,i_file):
    sim_num =0
    num_of_bins = 25
    bin_range = np.logspace(7.5,12.5,num_of_bins)
    bin_width = bin_range[1]-bin_range[0]
    hmf = np.ones((num_of_bins-1,2),dtype=np.float128)
    
    for sim_directory in cdm_sim_directory, wdm_sim_directory:
        print(f'Reading from {sim_directory}\n')
        snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
        # box_volume = (snap_data['Header'].attrs['BoxSize'])**3 

        # halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS
        halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS
        try:
            halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.int32)

            mass_mask = np.argsort(halo_mass)[::-1]
            clean_halos = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0]
            grouplen_types = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.float64)
            real_halos = np.where(grouplen_types[:,HIGHDMTYPE] > 2)[0]
            true_halos = np.isin(clean_halos,real_halos)
            # Need to normalise by volume, to hard to determine in final snap, just use well defined cubic volume defined in t=0/z=127 snapshot
            pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
            highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64)
            x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
            y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
            z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
            box_volume=x_extent*y_extent*z_extent
            mass_func,bins_func = np.histogram((halo_mass[clean_halos[true_halos]]),bin_range)
        except:
            mass_func,bins_func = np.histogram(np.log10(halo_mass),bin_range)

        # print(mass_func)
        for i in range(num_of_bins-1): #Change from bin edges to bin centres, remove last value in bins_func when plotting
            bins_func[i]=bins_func[i]+(bins_func[i+1]-bins_func[i])/2
        bins_func=bins_func[:-1]
        # Normalise the hmf by log bin width & box volume
        hmf[:,sim_num] = mass_func/(bin_width*box_volume)
        # print(hmf)
        sim_num+=1

    R_fit = hmf[:,1]/hmf[:,0] # WDM / CDM
    
    R_fit = np.ma.masked_invalid(R_fit)
    # print(R_fit)
    # plt.scatter(bins_func,R_fit,label = f'{label} / CDM')
    plt.plot(bins_func,R_fit,label = f'{label}/CDM')

    lambda_fs, mass_fs, mass_hm = free_streaming_mass(mass, snap_data)
    print(f"The half mode mass = {mass_hm:.2E} M_sun/h")
    # plt.axvline(np.log10(mass_hm),ls='--',color='k')
    alpha = 2.3
    beta = 0.8
    gamma = -1.0
    R_fit_analytic = (1+(alpha*mass_hm/bin_range)**beta)**gamma
    plt.plot(bin_range,R_fit_analytic,ls='-.', alpha=0.8)

    plt.xscale('log')
    plt.axhline(1,ls='--',color='red')
    plt.ylim((-0.1,1.2))
    plt.xlabel(r'M$_{H}$ [$M_{\odot}$]')
    plt.ylabel(r'R$_{n200,WDM}$/R$_{n200,CDM}$ ') # $dn/d$log$_{10}M_H$ [Mpc$^{-3}$ $h^3$ dex$^{-1}$]
    plt.title(f'Cumulative HMF at Redshift = {np.round(z,2)}')
    plt.legend()
    # plt.savefig('test_plots/Rfit_hmf.png')
    plt.savefig(f'{folder}/Rfit_hmf.png')

def subhalo_mass_function(sim_directory,label,folder,i_file):

    snap_data, haloinfo_data, z, unit_length = get_sim_data(sim_directory,i_file)

    subh_mass  = np.array(haloinfo_data['Subhalo']['SubhaloMass'], dtype=np.float64) * UNITMASS / LITTLEH
    
    subhalo_halo_num = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'], dtype=np.float64)
    subhalo_rank = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.float64)
    subhalo_len_type = np.array(haloinfo_data['Subhalo']['SubhaloLenType'], dtype=np.float64)
    subhalo_masstypes = np.array(haloinfo_data['Subhalo']['SubhaloMassType'], dtype=np.int32) 


    bin_width = 0.15
    num_of_bins = np.int64((np.max(np.log10(subh_mass)+1e-4) - np.min(np.log10(subh_mass)-1e-4))/bin_width)
    bin_range = np.linspace(np.min(np.log10(subh_mass)-1e-4),np.max(np.log10(subh_mass)+1e-4),num_of_bins)
    
    # box_volume = (snap_data['Header'].attrs['BoxSize'])**3 
    try:
        clean_halos = np.where(subhalo_masstypes[:,LOWDMTYPE] == 0)[0]
        real_halos  = np.where(subhalo_len_type[:,HIGHDMTYPE] > 5)[0]
        clean_halos = clean_halos[np.isin(clean_halos,real_halos)]
        
        # Need to normalise by volume, to hard to determine in final snap, just use well defined cubic volume defined in t=0/z=127 snapshot
        pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
        highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64) * unit_length
        x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
        y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
        z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
        box_volume=x_extent*y_extent*z_extent

        mass_func,bins_func = np.histogram(np.log10(subh_mass[clean_halos]),bin_range)
    except:
        pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
        highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64) * unit_length
        x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
        y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
        z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
        box_volume=x_extent*y_extent*z_extent

        mass_func,bins_func = np.histogram(np.log10(subh_mass),bin_range)
    
    for i in range(num_of_bins-1): #Change from bin edges to bin centres, remove last value in bins_func when plotting
        bins_func[i]=bins_func[i]+(bins_func[i+1]-bins_func[i])/2
    bins_func=bins_func[:-1]
    
    # Normalise the hmf by log bin width & box volume
    hmf = mass_func/(bin_width*set_plot_len(set_plot_len(set_plot_len(box_volume))))

    plt.plot(bins_func,hmf,label=f"{str(label)} at z={z}") 
    # # plt.xscale('log')
    plt.yscale('log')
    z = np.round(z,2)
    plt.xlabel(r'Log$_{10}M_{H}$ [$M_{\odot}$]')
    plt.ylabel(r'$dn/d$log$_{10}M_H$ [Mpc$^{-3}$ $h^3$ dex$^{-1}$]')
    plt.title("Subhalo Mass Function")
    plt.legend()
    # plt.savefig(f'{folder}/SHMF_z_{z}.png',dpi=400,bbox_inches='tight')
    plt.savefig(f'{folder}/SHMF_comb.png',dpi=400,bbox_inches='tight')

def outer_subhalo_mass(sim_directory,label,folder,i_file):
    snap_data, haloinfo_data, z, unit_length = get_sim_data(sim_directory,i_file)

    subh_mass  = np.array(haloinfo_data['Subhalo']['SubhaloMass'], dtype=np.float64) * UNITMASS
    subhalo_mass_type = np.array(haloinfo_data['Subhalo']['SubhaloMassType'],dtype=np.float64) * UNITMASS
    stellar_mass = subhalo_mass_type[:,4]

    subhalo_halo_num = np.array(haloinfo_data['Subhalo']['SubhaloGroupNr'], dtype=np.float64)
    subhalo_rank = np.array(haloinfo_data['Subhalo']['SubhaloRankInGr'], dtype=np.float64)
    subhalo_len_type = np.array(haloinfo_data['Subhalo']['SubhaloLenType'], dtype=np.float64)
    
    outer_index = np.where(subhalo_rank!=0)

    bin_width = 0.2
    num_of_bins = np.int64((np.max(np.log10(subh_mass)+1e-4) - np.min(np.log10(subh_mass)-1e-4))/bin_width)
    bin_range = np.linspace(np.min(np.log10(subh_mass)-1e-4),np.max(np.log10(subh_mass)+1e-4),num_of_bins)
    # print(f"Max = {np.max(np.log10(subh_mass)+1e-4)}, Min = {np.min(np.log10(subh_mass)-1e-4)},\n Range = {bin_range}")
    
    box_volume = (snap_data['Header'].attrs['BoxSize'])**3 

    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.int32)
    subhalo_masstypes = np.array(haloinfo_data['Subhalo']['SubhaloMassType'], dtype=np.int32)

    mass_mask = np.argsort(subh_mass)[::-1]
    clean_halos = np.where(subhalo_masstypes[:,LOWDMTYPE] == 0)[0]
    real_halos  = np.where(subhalo_len_type[:,HIGHDMTYPE] > 5)[0]
    clean_halos = clean_halos[np.isin(clean_halos,real_halos)]
    
    # Need to normalise by volume, to hard to determine in final snap, just use well defined cubic volume defined in t=0/z=127 snapshot
    pos_data = h5py.File(sim_directory+'snapshot_000.hdf5', 'r') 
    highdm_pos = np.array(pos_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64) * unit_length
    x_extent = np.max(highdm_pos[:,0])-np.min(highdm_pos[:,0])
    y_extent = np.max(highdm_pos[:,1])-np.min(highdm_pos[:,1])
    z_extent = np.max(highdm_pos[:,2])-np.min(highdm_pos[:,2])
    box_volume=x_extent*y_extent*z_extent 
    # print(box_volume)
    mass_func,bins_func = np.histogram(np.log10(subh_mass[clean_halos][outer_index]),bin_range)
    
    for i in range(num_of_bins-1): #Change from bin edges to bin centres, remove last value in bins_func when plotting
        bins_func[i]=bins_func[i]+(bins_func[i+1]-bins_func[i])/2
    bins_func=bins_func[:-1]
    
    # Normalise the hmf by log bin width & box volume
    hmf = mass_func/(bin_width*set_plot_len(set_plot_len(set_plot_len(box_volume))))

    # plt.hist(subh_mass[clean_list],bins=np.logspace(7,15,70),histtype='step',label=f"{mass} Subhalo Mass Function")
    plt.plot(bins_func,hmf,label=label) 
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'log$_{10}M_{*}$ [$M_{\odot}$]')
    plt.ylabel(r'dn/dlog$_{10}M_{*}$ [$Mpc^{-3}$ $h^3$ $dex^{-1}$]')
    
    plt.title('Outer SHMF')
    plt.legend()
    plt.savefig(f'{folder}/outer_SHMF_z_{z}.png')

def cumulative_subhalo_mass(sim_directory, label,folder,i_file):
    snap_data, haloinfo_data, z, unit_length = get_sim_data(sim_directory,i_file)

    halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
    # print(f"Max = {np.max(np.log10(subh_mass)+1e-4)}, Min = {np.min(np.log10(subh_mass)-1e-4)},\n Range = {bin_range}")
    
    subh_mass  = np.array(haloinfo_data['Subhalo']['SubhaloMass'], dtype=np.float64) * UNITMASS / LITTLEH  

    try:
        subhalo_masstypes = np.array(haloinfo_data['Subhalo']['SubhaloMassType'], dtype=np.int32)
        subhalo_len_type = np.array(haloinfo_data['Subhalo']['SubhaloLenType'], dtype=np.float64)
        clean_halos = np.where(subhalo_masstypes[:,LOWDMTYPE] == 0)[0]
        real_halos  = np.where(subhalo_len_type[:,HIGHDMTYPE] > 5)[0]
        clean_halos = clean_halos[np.isin(clean_halos,real_halos)]
        
        subh_mass = subh_mass[clean_halos]
    except:
        
        subh_mass = subh_mass

    ordered_halos = np.argsort(subh_mass)[::-1]
    cummass_func = np.cumsum(np.ones(subh_mass[ordered_halos].shape[0]))

    # plt.plot(subh_mass[ordered_halos],cummass_func/box_volume,label=str(label))
    plt.plot(subh_mass[ordered_halos],cummass_func,label=f"{str(label)} at z={z}")
    z = np.round(z,2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'M$_{H}$ [$M_{\odot}$]')
    # plt.ylabel(r'n $>$ M [Mpc$^{-3}$ $h^3$]')
    plt.ylabel(r'n $>$ M ')
    # plt.title(f'Cumulative SHMF at z={z}')
    plt.title(f'Cumulative SHMF')
    plt.legend()
    # plt.savefig(f'{folder}/cumulative_shmf_z_{z}.png',dpi=400,bbox_inches='tight')
    plt.savefig(f'{folder}/cumulative_shmf_comb.png',dpi=400,bbox_inches='tight')

if __name__ == "__main__":

    # folder  = 'N2048_L65_sd17492'
    # folder  = 'N2048_L65_sd28504'
    # folder  = 'N2048_L65_sd34920'
    # folder  = 'N2048_L65_sd46371'
    # folder  = 'N2048_L65_sd57839'
    # folder  = 'N2048_L65_sd61284'
    # folder  = 'N2048_L65_sd70562'
    folder  = 'N2048_L65_sd80325'
    # folder  = 'N2048_L65_sd93745'

    cdm_sn_005 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_005/'
    cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'

    
    wdm_sn_005 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
    wdm_sn_010 = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/'

    # folder = 'a_knebe'
    # cdm = '/fred/oz217/aussing/a_knebe/cdm/output/'
    # wdm_2 = '/fred/oz217/aussing/a_knebe/wdm_2/output/'
    # wdm_05 = '/fred/oz217/aussing/a_knebe/wdm_0.5/output/'

    if not os.path.exists(f'./{folder}'):
        os.makedirs(f'./{folder}')

    halo_mass_funtion(cdm_sn_005,'CDM SN=5%',folder,26,'blue','--') 
    halo_mass_funtion(cdm_sn_010,'CDM SN=10%',folder,26,'blue')

    halo_mass_funtion(wdm_sn_005,'WDM SN=5%',folder,26,'orange','--')
    halo_mass_funtion(wdm_sn_010,'WDM SN=10%',folder,26,'orange')

    plt.close()
    print("Cumulative HMF")
    # cumulative_halo_mass(sim_directory,mass, label,folder)
    cumulative_halo_mass(cdm_sn_005,999,'CDM SN=5\%',folder,26,'blue','--') 
    cumulative_halo_mass(cdm_sn_010,999,'CDM SN=10\%',folder,26,'blue')
    
    cumulative_halo_mass(wdm_sn_005,3.5,'WDM SN=5\%',folder,26,'orange','--')
    cumulative_halo_mass(wdm_sn_010,3.5,'WDM SN=10\%',folder,26,'orange')

    # plt.close()
    # print("SHMF")
    # # subhalo_mass_function(sim_directory,label,folder,file_num)
    # subhalo_mass_function(dmo_Kpc,'DMO_Kpc',folder,file_num)
    # subhalo_mass_function(dmo_Mpc,'DMO_Mpc',folder,file_num)
    # subhalo_mass_function(cdm,'CDM',folder,i_file_cdm)
    # subhalo_mass_function(wdm_2,'WDM 2KeV',folder,i_file_wdm_2)
    # subhalo_mass_function(wdm_05,'WDM 0.5KeV',folder,i_file_wdm_05)
    # subhalo_mass_function(cdm_sn_005,'CDM SN=5%',folder)
    # subhalo_mass_function(cdm_sn_010,'CDM SN=10%',folder)
    # # subhalo_mass_function(cdm_sn_015,'CDM SN=15%',folder)

    # subhalo_mass_function(wdm_sn_005,'WDM SN=5%',folder)
    # subhalo_mass_function(wdm_sn_010,'WDM SN=10%',folder)
    # subhalo_mass_function(wdm_sn_015,'WDM SN=15%',folder)

    # plt.close()
    # outer_subhalo_mass(cdm,'CDM',folder)
    # outer_subhalo_mass(wdm,'WDM',folder)
    # outer_subhalo_mass(cdm_sn_005,'CDM SN=5%',folder)
    # outer_subhalo_mass(cdm_sn_010,'CDM SN=10%',folder)
    # # outer_subhalo_mass(cdm_sn_015,'CDM SN=15%',folder)

    # outer_subhalo_mass(wdm_sn_005,'WDM SN=5%',folder)
    # outer_subhalo_mass(wdm_sn_010,'WDM SN=10%',folder)
    # outer_subhalo_mass(wdm_sn_015,'WDM SN=15%',folder)

    # plt.close()
    # print("Cumulative SHMF")
    # cumulative_subhalo_mass(dmo_Kpc,'DMO_Kpc',folder,file_num)
    # cumulative_subhalo_mass(dmo_Mpc,'DMO_Mpc',folder,file_num)
    # cumulative_subhalo_mass(cdm,'CDM',folder,i_file_cdm)
    # cumulative_subhalo_mass(wdm_2,'WDM 2KeV',folder,i_file_wdm_2)
    # cumulative_subhalo_mass(wdm_05,'WDM 0.5KeV',folder,i_file_wdm_05)  
    # cumulative_subhalo_mass(cdm_sn_005,'CDM SN=5%',folder)
    # cumulative_subhalo_mass(cdm_sn_010,'CDM SN=10%',folder)
    # # cumulative_subhalo_mass(cdm_sn_015,'CDM SN=15%',folder)
    
    
    # cumulative_subhalo_mass(wdm_sn_005,'WDM SN=5%',folder)   
    # cumulative_subhalo_mass(wdm_sn_010,'WDM SN=10%',folder)   
    # cumulative_subhalo_mass(wdm_sn_015,'WDM SN=15%',folder)   
    

