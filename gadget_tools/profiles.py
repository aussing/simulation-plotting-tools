# from importlib.machinery import FileFinder
# from re import T
import numpy as np
import h5py 
import matplotlib.pyplot as plt
import matplotlib as mpl
# import scipy.stats as stat

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
plt.style.use('/home/aussing/sty.mplstyle')
# SAVGOL_SMOOTH = True

# sim_directory = '/fred/oz217/vel_disp_sims/cdm/zoom/'
# sim_directory = '/fred/oz217/aussing/N2048_L65_sd34920/cdm/zoom/output/sn_010'

# sim_directory = '/fred/oz217/aussing/N2048_L65_sd57839/cdm/zoom/output/sn_010'


redshifts  = np.array([127, 0])
filenumber = np.array([0,  26])

LITTLEH   = 0.6688
UNITMASS  = 1.0e10
GASTYPE = 0
HIGHDMTYPE = 1
LOWDMTYPE  = 5
STARTYPE   = 4

i_file = 26


def critical_density(h,OmegaR0,OmegaM0,OmegaK0,OmegaL0,z,comoving=False,hCorrect=False):
    '''Calculates critical density'''

    G  = 6.67430e-11     # in m^3 kg^-1 s^-2 = m kg^-1 (m/s)^2
    factor1 = 1e-6       # (m/s)^2 to (km/s)^2
    factor2 = 1.98855e30 # kg^-1 to Msun^-1
    factor3 = 1/3.086e22 # m to Mpc
    # factor3 = 1/3.086e19 # m to Kpc
    G  = G*factor1*factor2*factor3 #in Mpc Msun^-1 (km/s)^2

    H0 = 100 * h  # H0 = 100h (km/s) Mpc^-1 !!!!
    # H0 = 0.1 * h # Kpc
    H2 = H0**2 *(OmegaR0*(1+z)**4 + OmegaM0*(1+z)**3 + OmegaK0*(1+z)**2 + OmegaL0)

    if comoving: # go from PHYSICAL critical density to COMOVING critical density. rho_phys = rho_comov/a**3 -->
                 # rho_comov = a**3 * rho_phys = (1/(1+z))**3 * rho_phys
        result = ((1/(1+z))**3)*(3*H2)/(8*np.pi*G)
    else:
        result = (3*H2)/(8*np.pi*G)

    if hCorrect: # in simulations: [rho] = h**2 * (Msun /Mpc**3). Turn this parameter ON so that rho_crit has the
                 # same units as the rho from simulations: rho_crit/h**2 --> [rho_crit] = h**2 (Msun /kpc**3)
        result = (1/LITTLEH**2)*result
    else:
        pass
    return result

def scaled_nfw_analytical(s,c):
    ''' Returns scaled NFW DM density rho/rho_crit from Lokas & Mamon 2001'''
    # s = r/r_200c
    # c = c_200c =  r_200c/rs
    gc = (np.log(1+c) - c/(1+c))

    # returns rho/rho_crit

    return 200/(3 * gc * s * (1/c + s)**2) 

def ludlow_cm_relation(m200c,z,OmegaL0,OmegaM0):
    '''Returns concentration parameter c200 from the mass-concentration relation in Ludlow et al. '16'''

    # Mass must be given in Msun/h !
    xi      = (m200c/1e10)**(-1) 
    a       = (1+z)**(-1)
    OmegaLZ = OmegaL0/(OmegaL0 + OmegaM0*(1+z)**3)
    OmegaMZ = 1 - OmegaLZ
    PsiZ    = OmegaMZ**(4/7) - OmegaLZ + (1 + OmegaMZ/2)*(1+OmegaLZ/70)
    Psi0    = OmegaM0**(4/7) - OmegaL0 + (1 + OmegaM0/2)*(1+OmegaL0/70)

    # Lahav et al. '91 approximation for the linear growth factor:
    growthZ = OmegaMZ/OmegaM0 * Psi0/PsiZ * a  
    
    # For halo masses between 1e7 < M/ Msun h^-1 < 1e15 sigma can be approximated within ~2.5% by:
    sigmaMZ = growthZ * (22.26 * xi**0.292)/(1 + 1.53 * xi**0.275 + 3.36 * xi**0.198)

    # Change from masses to dimensionless peak height
    delta_sc = 1.686 # spherical top-hat collapse threshold
    nu  = delta_sc/sigmaMZ

    # The following relations are valid for 1 >= log(1+z) >= 0 and -8 <= log M(Msun h^-1) <= 16.5:
    nu0    = growthZ**(-1)*(4.135 - 0.564*a**(-1) - 0.210*a**(-2) + 0.0557*a**(-3) -0.00348*a**(-4))
    c0     = 3.395 * (1+z)**(-0.215)
    beta   = 0.307 * (1+z)**(0.540)
    gamma1 = 0.628 * (1+z)**(-0.047)
    gamma2 = 0.317 * (1+z)**(-0.893)

    aux1 = c0 * (nu/nu0)**(-gamma1)
    aux2 = 1 + (nu/nu0)**(1/beta)

    return aux1*aux2**(-beta*(gamma2-gamma1))

if __name__ == "__main__":
    # snap_list = np.linspace(10,26,17,dtype=int)
    snap_list = np.linspace(20,26,7,dtype=int)
    
    folder = 'N2048_L65_sd46371'
    sim_directory_list = [f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010',f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010']
    colour_list = ['Blues', 'Oranges']
    name = ['CDM','WDM']
    element_list = ['tot','Gas', 'Star', 'DM']  ##  'tot'
    for element in element_list:
        plt.close()
        fig = plt.figure(constrained_layout=True)
        colour_list = ['Blues', 'Oranges']
        name = ['CDM','WDM']
        for i,sim_directory in enumerate(sim_directory_list):
        # cmap = mpl.cm.get_cmap('copper_r')
            cmap = mpl.cm.get_cmap(colour_list[i])
            # norm = mpl.colors.Normalize(vmin=5, vmax=30)
            norm = mpl.colors.Normalize(vmin=18, vmax=28)
            # print(cmap(26))
            for snap_num in snap_list:
                snap_fname     = f'/snapshot_{str(snap_num).zfill(3)}.hdf5'
                snap_directory = sim_directory + snap_fname

                haloinfo_fname     = f'/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5'
                haloinfo_directory = sim_directory + haloinfo_fname
                print(f"snap directory+name = {snap_directory}")

                snap_data     = h5py.File(snap_directory, 'r')
                haloinfo_data = h5py.File(haloinfo_directory, 'r')
                halo_pos   = np.array(haloinfo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH 
                halo_R200c = np.array(haloinfo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
                halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
                halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH
                halo_partlen = np.array(haloinfo_data['Group']['GroupLenType'],dtype=np.int32)

                mass_mask = np.argsort(halo_M200c)[::-1]
                halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]

                print(f"halo {mass_mask[halo_mainID]} has {halo_partlen[mass_mask][halo_mainID][1]} DM paricles")
                # Plot radial density plots (tot, dm, gas, stars)
                
                # ax  = plt.axes([0.1,0.1,0.9,0.9])

                OMEGAR0 = 0  # LCDM
                OMEGAK0 = 0  # LCDM
                OMEGAM0 = snap_data['Parameters'].attrs['Omega0']
                OMEGAB0 = snap_data['Parameters'].attrs['OmegaBaryon']
                OMEGAL0 = snap_data['Parameters'].attrs['OmegaLambda']
                Z       = snap_data['Header'].attrs['Redshift']

                CRITDENS = critical_density(LITTLEH,OMEGAR0,OMEGAM0,OMEGAK0,OMEGAL0,Z,comoving=True,hCorrect=False)


                # Alternatively, calculate critical density from yt
                # ytcosmo = ytCosmology(
                #     hubble_constant=LITTLEH,
                #     omega_matter=OMEGAM0,
                #     omega_lambda=OMEGAL0,
                #     omega_curvature=OMEGAK0,
                #     omega_radiation=OMEGAR0,
                # )
                # CRITDENS = ytcosmo.critical_density(Z).to('Msun/Mpc**3')

                highdm_pos = np.array(snap_data[f'PartType{HIGHDMTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH 
                lowdm_pos  = np.array(snap_data[f'PartType{LOWDMTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH 
                gas_pos    = np.array(snap_data[f'PartType{GASTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH 
                star_pos   = np.array(snap_data[f'PartType{STARTYPE}']['Coordinates'], dtype=np.float64) / LITTLEH

                highdm_pid = np.array(snap_data[f'PartType{HIGHDMTYPE}']['ParticleIDs'], dtype=np.uint32)
                lowdm_pid  = np.array(snap_data[f'PartType{LOWDMTYPE}']['ParticleIDs'], dtype=np.uint32)
                gas_pid    = np.array(snap_data[f'PartType{GASTYPE}']['ParticleIDs'], dtype=np.uint32)
                star_pid   = np.array(snap_data[f'PartType{STARTYPE}']['ParticleIDs'], dtype=np.uint32)

                # High resolution DM particles do not have a "Masses" attribute, as they all have the same mass. 
                # Low resolution DM particles do have one, as they include all the coarser resolutions.
                # We must create ourselves a mass array for high resolution particles. 
                highdm_mass = np.repeat(snap_data['Header'].attrs['MassTable'][HIGHDMTYPE] * UNITMASS / LITTLEH,
                                        snap_data['Header'].attrs['NumPart_Total'][HIGHDMTYPE])
                lowdm_mass  = np.array(snap_data[f'PartType{LOWDMTYPE}']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH
                gas_mass    = np.array(snap_data[f'PartType{GASTYPE}']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH
                star_mass   = np.array(snap_data[f'PartType{STARTYPE}']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH

                tot_pos  = np.concatenate((highdm_pos, lowdm_pos, gas_pos, star_pos), axis=0)
                tot_mass = np.concatenate((highdm_mass, lowdm_mass, gas_mass, star_mass), axis=0)
                tot_pid  = np.concatenate((highdm_pid, lowdm_pid, gas_pid, star_pid), axis=0)

                # Pick the target halo
                target_halocenter = halo_pos[mass_mask][halo_mainID]
                target_M200c      = halo_M200c[mass_mask][halo_mainID]
                target_R200c      = halo_R200c[mass_mask][halo_mainID]

                num_bins = 40
                target_rbins      = np.logspace(np.log10(0.001*target_R200c), np.log10(target_R200c), num=num_bins, base=10)

                # Profiles
                target_profiles = {}
                target_profiles['rbins']    = target_rbins
                target_profiles['rmean']    = np.insert(( target_rbins[:-1] 
                                                        +target_rbins[1:] )/2, 
                                                        0, target_rbins[0])
                # Use the logspace mean of the radial bins as the radial values at which we evaluate the (differential) profiles
                target_profiles['logrmean'] = 10**np.insert((  np.log10(target_rbins[:-1]) 
                                                            + np.log10(target_rbins[1:]) )/2, 
                                                            0, np.log10(target_rbins[0]))
                target_profiles['logrmean_scaled'] = np.divide(target_profiles['logrmean'],target_R200c)
                target_profiles['vol']      = 4/3 * np.pi * target_rbins**3
                target_profiles['diffvol']  = np.insert( np.diff(target_profiles['vol']), 0, target_profiles['vol'][0])


                for species in [element]:# ['tot', 'dm', 'gas', 'star']:
                    if species == 'tot':
                        species_mass  = tot_mass
                        species_pos   = tot_pos
                        species_pid   = tot_pid
                        # species_color = 'purple'
                        # species_color = f'C{snap_num-10}'
                        species_color = cmap(norm(snap_num))
                    elif species == 'DM':
                        species_mass  = highdm_mass
                        species_pos   = highdm_pos
                        species_pid   = highdm_pid
                        # species_color = 'k'
                        species_color = cmap(norm(snap_num))
                    elif species == 'Gas':
                        species_mass  = gas_mass
                        species_pos   = gas_pos
                        species_pid   = gas_pid
                        # species_color = 'b'
                        species_color = cmap(norm(snap_num))
                    elif species == 'Star':
                        species_mass  = star_mass
                        species_pos   = star_pos
                        species_pid   = star_pid
                        # species_color = 'r'
                        species_color = cmap(norm(snap_num))
                    else:
                        raise NameError(f'Invalid species "{species}"!')

                    print(f'Calculating profile for {species} species...')
                    target_profiles[species] = {}
                    
                    # Select particles up to R200c
                    species_dist  = (species_pos[:,0] - target_halocenter[0])**2
                    species_dist += (species_pos[:,1] - target_halocenter[1])**2
                    species_dist += (species_pos[:,2] - target_halocenter[2])**2
                    species_dist  = np.sqrt(species_dist)
                    
                    species_distmask, = np.where(species_dist < target_R200c)
                    species_mass = species_mass[species_distmask]
                    species_dist = species_dist[species_distmask]
                    species_pid  = species_pid[species_distmask]

                    # Check if there's any low resolution DM in the profiles
                    if species == 'tot':
                        lowdm_in_target_halo_mask = np.isin(lowdm_pid, species_pid)
                        if lowdm_in_target_halo_mask.any():
                            print(f'Target halo contaminated by {lowdm_in_target_halo_mask.sum()} low resolution DM particles!')
                        
                    # Enclosed mass
                    species_mass_in_r = []
                    for rightrbin in target_rbins:
                        species_mass_in_r_mask, = np.where(species_dist <= rightrbin)
                        species_mass_in_r.append(np.sum(species_mass[species_mass_in_r_mask]))
                    species_mass_in_r = np.array(species_mass_in_r)

                    target_profiles[species]['mass_in_r'] = species_mass_in_r
                    species_diffmass                      = np.insert( np.diff(species_mass_in_r), 0, species_mass_in_r[0])
                    target_profiles[species]['ovdens']    = np.divide(species_mass_in_r, target_profiles['vol'] ) / CRITDENS
                    target_profiles[species]['dens']      = np.divide(species_diffmass, target_profiles['diffvol'] ) / CRITDENS
                    target_profiles[species]['scaled_dens'] = target_profiles['logrmean_scaled']**2 * target_profiles[species]['dens'] 
                    # ax = plt.plot(target_rbins,target_profiles[species]['ovdens'], color=species_color)

                    # ax = plt.plot(target_profiles['logrmean_scaled'], target_profiles[species]['scaled_dens'], color=species_color)
                    if snap_num in [10,11,26]:
                        ax = plt.plot(target_profiles['logrmean_scaled'], target_profiles[species]['dens'], color=species_color,label=name[i])#label=f"{np.round(Z,2)}")
                    else:
                        ax = plt.plot(target_profiles['logrmean_scaled'], target_profiles[species]['dens'], color=species_color)

                # Converegence from Power et al 2003 eq 15, normalised by r200c
                r_converge = (4*target_R200c/(np.sqrt(halo_partlen[mass_mask][halo_mainID][1])))/target_R200c 
                if snap_num == 10 or snap_num==26:
                    ax = plt.axvline(x=r_converge,color='k', ls='-.')

                c200_analytical        = ludlow_cm_relation(target_M200c*LITTLEH, Z,OMEGAL0,OMEGAM0)
                print(f"Concentration = {c200_analytical}")
                rscaled_analytical     = np.linspace(target_profiles['logrmean_scaled'].min(), target_profiles['logrmean_scaled'].max(), 100)
                # scaled_dens_analytical = (rscaled_analytical)**2 * scaled_nfw_analytical(rscaled_analytical,c200_analytical)

                if snap_num == 10 or snap_num==26:
                    scaled_dens_analytical = scaled_nfw_analytical(rscaled_analytical,c200_analytical)
                    ax = plt.plot(rscaled_analytical,scaled_dens_analytical, color='k', ls='dashed')

        # max_nfw = np.max(scaled_dens_analytical)
        # max_nfw_radius = rscaled_analytical[np.where(scaled_dens_analytical==max_nfw)[0][0]]
        # print("max radius = ",max_nfw_radius)
        # ax = plt.axvline(x=max_nfw_radius,color='k', ls='dotted')

        ax = plt.xscale('log')
        # ax = plt.xlim((1e-3,1))
        ax = plt.yscale('log')
        # ax = plt.ylim((1e-1,1e3))

        ax = plt.xlabel(r"$r/r_{200c}$")
        # ax = plt.ylabel(r'$(r/r_{200c})^2 (\rho/\rho_{crit})$')
        ax = plt.ylabel(r'$\rho/\rho_{crit}$') #/\rho_{crit})
        ax = plt.legend()
        # ax = plt.title(f"Density profiles for sim {folder}")
        if element=='tot':
            ax = plt.title(f"Total mass density profile")
        else:
            ax = plt.title(f"{element} density profile")
        ax = plt.savefig(f'./Profiles_{element}.png', dpi=400)
        print(f"Saving figure Profiles_{element}.png")
        plt.close
