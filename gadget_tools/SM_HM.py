import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 14

LITTLEH    = 0.6688
UNITMASS   = 1.0e10
GASTYPE    = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5


def get_sim_data(sim_directory,i_file):
    snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = (snap_data['Header'].attrs['Redshift'])
    print(z)
    return snap_data, haloinfo_data, z
     
def analytic_SMHM():
    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('behroozi10',redshift=0)
    halo_mass = np.logspace(9, 14, 100)
    # halo_mass = np.logspace(10, 14, 100)
    mean_sm = model.mean_stellar_mass(prim_haloprop = halo_mass)

    plt.plot(halo_mass,mean_sm, label="Behroozi 2010",color='k', ls='-.')

def MWA_stellar_halo_mass(sim_directory,i_file):
    snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
    halo_M200c     = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH 
    
    mass_mask = np.argsort(halo_M200c)[::-1]
    halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
    
    return halo_M200c[mass_mask[halo_mainID]], halo_masstypes[mass_mask[halo_mainID],4]

def stellar_mass_halo_mass(sim_directory,label,folder,color,i_file,alpha=1):
        print('\n',label)
        snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)

        halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH

        # print(halo_masstypes)
        # clean_haloes = np.where(halo_masstypes[:,5]==0)

        plt.scatter(halo_M200c[:],halo_masstypes[:,4],label=label,color=color)
        # plt.scatter(halo_masstypes[clean_haloes,1],halo_masstypes[clean_haloes,4],label=label,s=2,color=color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$M_{total}$')
        plt.ylabel(r'$M_{*}$')
        plt.title(f'Stellar Mass Halo Mass relation at z = {np.round(z,2)}')
        plt.legend()
        plt.tight_layout()
        
        
        plt.savefig(f'{folder}/stel_mas_halo_mass_{np.round(z,2)}.png',dpi=400)
        # plt.savefig(f'./Stel_mas_halo_mass.png',dpi=400)
        
if __name__ == "__main__":

    # if not os.path.exists(f'./{folder}'):
    #     os.makedirs(f'./{folder}')
    i_file = 26
    analytic_SMHM()

    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/clean_pd/powderday/') if ( simulation_name.startswith("N2048_L65_sd1") )]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    
    for folder in folder_list:
        cdm_file_loc = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'
        wdm_file_loc = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'

        CDM_halo_mass, CDM_stellar_mass = MWA_stellar_halo_mass(cdm_file_loc,i_file)
        WDM_halo_mass, WDM_stellar_mass = MWA_stellar_halo_mass(wdm_file_loc,i_file)

        plt.scatter(CDM_halo_mass,CDM_stellar_mass,c='blue')
        plt.scatter(WDM_halo_mass,WDM_stellar_mass,c='orange')
    plt.savefig(f'./test_SM_HM_MWA.png',dpi=250)
        # single_stellar_mass_halo_mass(cdm_file_loc,'',folder,'blue',i_file)
        # single_stellar_mass_halo_mass(wdm_file_loc,'',folder,'orange',i_file)
    
    # stellar_mass_halo_mass(cdm_sn_005,'CDM SN=5\%',folder,'C0',i_file)
    # stellar_mass_halo_mass(cdm_sn_010,'CDM SN=10\%',folder,'C1',i_file)
    # stellar_mass_halo_mass(wdm_sn_005,'WDM SN=5\%',folder,'C2',i_file)
    # stellar_mass_halo_mass(wdm_sn_010,'WDM SN=10\%',folder,'C3',i_file)
    
    # cdm_sn_010 = f'/fred/oz217/aussing/{folder}/cdm/zoom/test_010/'

    # kpc = '/fred/oz004/aussing/len_test/N256_L50_hydro_Kpc/'
    # mpc = '/fred/oz004/aussing/len_test/N256_L50_hydro_Mpc/'
    # kpc_new = '/fred/oz004/aussing/len_test/N256_L50_hydro_Kpc_unit_test'
    # analytic_SMHM(kpc_new,1)
    # stellar_mass_halo_mass(mpc,'Mpc unit',folder,'C0',1)
    # stellar_mass_halo_mass(kpc,'kpc unit',folder,'C1',1)
    # stellar_mass_halo_mass(kpc_new,'kpc new unit',folder,'C2',1)


    print('Finished')

