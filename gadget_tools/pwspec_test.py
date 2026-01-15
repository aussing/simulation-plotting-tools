from nbodykit.lab import FFTPower, cosmology
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import nbodykit
import matplotlib.pyplot as plt
import h5py

plt.style.use('/home/aussing/sty.mplstyle')

LITTLEH = 0.6688
OMEGA_MATTER = 0.321
OMEGA_BARYON = 0.045
n_s = 0.9626
sigma8 = 0.84
OMEGA_DM = OMEGA_MATTER - OMEGA_BARYON
# a_s = 2.1064e-9 # corresponding to sigma8 = 0.8105
k_p = 0.05
# n_ur = 2.046
t_cmb = 2.7255

def read_file(File):
    # print(f"Reading file: {File}")
    startline=5
    with open(File,'r') as f:
        powerspec_bins_b1 = f.readlines()[1]
        powerspec_bins_b1 = np.int32(powerspec_bins_b1)
    
    with open(File,'r') as f:
        powerspec_b1 = f.readlines()[startline:powerspec_bins_b1+startline]
        powerspec_b1 = np.genfromtxt(powerspec_b1, delimiter=' ')

    with open(File,'r') as f:
        powerspec_bins_b2 = f.readlines()[powerspec_bins_b1+startline+1]
        powerspec_bins_b2 = np.int32(powerspec_bins_b2)
        
    with open(File,'r') as f:
        powerspec_b2 = f.readlines()[powerspec_bins_b1+2*startline:powerspec_bins_b2+powerspec_bins_b1+2*startline]
        powerspec_b2 = np.genfromtxt(powerspec_b2, delimiter=' ')

    with open(File,'r') as f:
        powerspec_bins_b3 = f.readlines()[powerspec_bins_b1+powerspec_bins_b2+2*startline+1]
        powerspec_bins_b3 = np.int32(powerspec_bins_b3)
        
    with open(File,'r') as f:
        powerspec_b3 = f.readlines()[powerspec_bins_b1+powerspec_bins_b2+3*startline:powerspec_bins_b1+powerspec_bins_b2+powerspec_bins_b3+3*startline]
        powerspec_b3 = np.genfromtxt(powerspec_b3, delimiter=' ')
    return powerspec_b1, powerspec_b2, powerspec_b3

def M_lim(rho_bar,k_peak):
    
    d_spacing = (2.4e6/rho_bar)**(1/3)
    m_lim = 10.1 * rho_bar * d_spacing / k_peak**2

    new_rho_bar = 2.7754e11 * 0.321 
    new_d_spacing = (2.4e6/new_rho_bar)**(1/3)
    new_m_lim = 10.1 * new_rho_bar * new_d_spacing / k_peak**2

    print(f"Simulation rho_bar = {rho_bar:.3e}")
    print(f"Wikipedia  rho_bar = {new_rho_bar:.3e}\n")

    print(f"Simulation dists   = {d_spacing}")
    print(f"Wikipedia  dists   = {new_d_spacing}\n")

    print(f'Simulation M limit = {m_lim:.3e} Msun/h')
    print(f'Wikipedia  M limit = {new_m_lim:.3e} Msun/h\n')

    m_lim = m_lim / LITTLEH
    new_m_lim = new_m_lim / LITTLEH
    print(f'Simulation M limit = {m_lim:.3e} Msun')
    print(f'Wikipedia  M limit = {new_m_lim:.3e} Msun')

    return m_lim

def calculate_analytical(redshift, wdm_mass,cosmo):
    
    k_nyquist=np.log10(5e2)
    # k = np.logspace(-1, k_nyquist, 400)
    k = np.geomspace(1e-4, 1e3, 500)
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')

    alpha = 0.049 * (wdm_mass/1)**(-1.11) * (OMEGA_DM/0.25)**(0.11) * (LITTLEH/0.7)**1.22
    T2 = ((1+(alpha*k)**(2*1.12))**(-5/1.12))**2

    return k, Plin(k), Plin(k)*T2, T2

wdm_mass = 3.5 # in keV

cosmo = cosmology.Cosmology(h=LITTLEH, Omega0_b=OMEGA_BARYON, Omega0_cdm=OMEGA_DM, n_s=n_s,  P_k_max=5e2)
cosmo = cosmo.match(sigma8=0.840)
rho_bar = cosmo.rho_m(0.0) * 1e10

k_z_127, Plin_cdm_z_127, Plin_wdm_z_127, transfer_func_z_127 = calculate_analytical(127,wdm_mass,cosmo)

Plin_cdm_dim_z_127 = (Plin_cdm_z_127*k_z_127**3)/(2*np.pi**2)
Plin_wdm_dim_z_127 = (Plin_wdm_z_127*k_z_127**3)/(2*np.pi**2)
k_peak_z_127 = k_z_127[Plin_wdm_dim_z_127==np.max(Plin_wdm_dim_z_127)][0]
print(f"Theoretical k max = {k_peak_z_127} @ z = 127")

k_z_0, Plin_cdm_z_0, Plin_wdm_z_0, transfer_func_z_0 = calculate_analytical(0,wdm_mass,cosmo)

Plin_cdm_dim_z_0 = (Plin_cdm_z_0*k_z_0**3)/(2*np.pi**2)
Plin_wdm_dim_z_0 = (Plin_wdm_z_0*k_z_0**3)/(2*np.pi**2)
k_peak_z_0 = k_z_0[Plin_wdm_dim_z_0==np.max(Plin_wdm_dim_z_0)][0]
print(f"Theoretical k max = {k_peak_z_0} @ z = 0 \n")

M_lim(rho_bar,k_peak_z_127)
M_lim(rho_bar,k_peak_z_0)

plt.plot(k_z_127,Plin_wdm_dim_z_127, c='orange')
plt.plot(k_z_127,Plin_cdm_dim_z_127, c='b')
# # plt.plot(k_z_127,Plin_wdm_z_127, c='orange')

# # plt.vlines(k_peak_z_127,1e-6,1e-2)
# nyquist = np.pi/(25/(2*376))
# box = 2*np.pi/(65)
# # plt.vlines(nyquist,1e-6,1e-2, colors='r')
# plt.axvline(box, color='k', ls='--')

plt.ylim(1e-7,1e1)
# plt.xlim(1e-2,9e2)
plt.xscale('log')
plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
plt.yscale('log')
plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^{3}]$")
plt.title('Redshift = 127')
plt.tight_layout()
# plt.ylim(1e-6,1e-2)
# # plt.vlines(nyquist,1e-3,1)
plt.savefig('./test.png',dpi=200)


# snap_directory = f'/fred/oz217/aussing/N2048_L65_sd46371/cdm/zoom/output/sn_010/snapshot_000.hdf5'
# # snap_directory = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/snapshot_{str(file_num).zfill(3)}.hdf5'
# snap_data = h5py.File(snap_directory, 'r')  

# redshift = snap_data['Header'].attrs['Redshift']
# scale_factor = snap_data['Header'].attrs['Time']
# boxlength = snap_data['Header'].attrs['BoxSize']
# num_part =  np.sum(snap_data['Header'].attrs['NumPart_Total'])


# file = '/fred/oz217/aussing/N2048_L65_sd46371/cdm/zoom/output/sn_010/powerspecs/powerspec_000.txt'
# # file = '/fred/oz217/aussing/N2048_L65_sd46371/cdm/coarse/output/powerspecs/powerspec_000.txt'
# powerspec_b1, powerspec_b2, powerspec_b3 = read_file(file)
# k = powerspec_b1[:,0]
# mask = k<(k_peak_z_127+5)
# k = k[mask]
# delta2_k = powerspec_b1[:,1][mask]
# # plt.plot(k,delta2_k)

# file = '/fred/oz217/aussing/N2048_L65_sd46371/wdm_3.5/zoom/output/sn_005/powerspecs/powerspec_000.txt'
# # file = '/fred/oz217/aussing/N2048_L65_sd46371/wdm_3.5/coarse/output/powerspecs/powerspec_000.txt'
# powerspec_b1, powerspec_b2, powerspec_b3 = read_file(file)
# k = powerspec_b1[:,0]
# mask = k<(k_peak_z_127+5)
# k = k[mask]
# delta2_k = powerspec_b1[:,1][mask]
# plt.plot(k,delta2_k)

# print(f"z=127:"
#       f"\n Free streaming mass:\t {mass_free_streaming_z99:.3e} Msun/h"
#       f"\n Half-mode mass:\t {mass_half_mode_z99:.3e} Msun/h")
# print(f"z=99:"
#       f"\n Free streaming mass:\t {mass_free_streaming_z39:.3e} Msun/h"
#       f"\n Half-mode mass:\t {mass_half_mode_z39:.3e} Msun/h")


# print("Schneider et al. 2012, 1keV WDM mass:"
#       "\n Free streaming mass:\t 4.9e+6 Msun/h"
#       "\n Half-mode mass:\t 1.3e+10 Msun/h")
# print("My calculation:"
#       f"\n Free streaming mass:\t {mass_free_streaming_z0:.1e} Msun/h"
#       f"\n Half-mode mass:\t {mass_half_mode_z0:.1e} Msun/h")


# rho_bar = cosmo.rho_m(0.0) * 1e10

# alpha_wdm = 0.049 * (wdm_mass / 1)**(-1.11) * ((OMEGA_MATTER-OMEGA_BARYON) / 0.25)**(0.11) * (LITTLEH / 0.7)**1.22 # in units of Mpc / h
# lovell_alpha = 0.05 * (wdm_mass)**(-1.15) * ((OMEGA_MATTER-OMEGA_BARYON) / 0.4)**(0.15) * (LITTLEH / 0.65)**1.3
# lambda_eff_free_streaming = alpha_wdm
# k_eff_free_streaming = 2 * np.pi / lambda_eff_free_streaming

# # mass_free_streaming_z99 = (4 * np.pi / 3) * mean_rho_z99 * (lambda_eff_free_streaming / 2)**3 # in units of Msun / h
# # mass_free_streaming_z39 = (4 * np.pi / 3) * mean_rho_z39 * (lambda_eff_free_streaming / 2)**3
# # mass_free_streaming_z0  = (4 * np.pi / 3) * mean_rho_z0 * (lambda_eff_free_streaming / 2)**3
# mass_free_streaming_z0  = (4 * np.pi / 3) * rho_bar * (lambda_eff_free_streaming / 2)**3

# # Define the half-mode length (i.e. mass scale below which the transfer function is suppressed by 1/2). This mass scale is where we expect the WDM to first affect the
# # properties of dark matter haloes.
# viel_mu = 1.12
# lambda_half_mode = 2 * np.pi * lambda_eff_free_streaming * (2**(viel_mu / 5) - 1)**(-1/ (2*viel_mu))

# bose_15_k_hm = 1 / lovell_alpha * (2**(viel_mu/5)-1)**(1/(2*viel_mu))
# bose_15_lambda_hm = 2 * np.pi / bose_15_k_hm

# k_half_mode = 2 * np.pi / lambda_half_mode

# # mass_half_mode_z99 = (4 * np.pi / 3) * mean_rho_z99 * (lambda_half_mode / 2)**3
# # mass_half_mode_z39 = (4 * np.pi / 3) * mean_rho_z39 * (lambda_half_mode / 2)**3
# # mass_half_mode_z0  = (4 * np.pi / 3) * mean_rho_z0  * (lambda_half_mode / 2)**3
# # bose_15_mass_hm = (4 * np.pi / 3) * mean_rho_z0  * (bose_15_lambda_hm / 2)**3

# mass_half_mode_z0  = (4 * np.pi / 3) * rho_bar  * (lambda_half_mode / 2)**3
# bose_15_mass_hm = (4 * np.pi / 3) * rho_bar  * (bose_15_lambda_hm / 2)**3

# print(f"approx formula = {np.round(2.73e3 * mass_free_streaming_z0,4):.3e} Msun/h")
# print(f"approx formula = {np.round(2.73e3 * mass_free_streaming_z0,4):.3e} Msun/h")
# print(f"WDM mass = {wdm_mass}")
# print('Schnieder+ 2012')
# print(
#       # f"\n Free streaming l:\t {lambda_eff_free_streaming:.3f} Mpc/h"
#       # f"\n Free streaming k:\t {k_eff_free_streaming:.3f} h/Mpc"
#       f"\nHalf-mode l:\t {lambda_half_mode:.3f} Mpc/h"
#       f"\nHalf-mode k:\t {k_half_mode:.3f} h/Mpc")
# print(#f"z=0:"
#       f"\nFree streaming mass: {mass_free_streaming_z0:.3e} Msun/h"
#       f"\nHalf-mode mass: {mass_half_mode_z0:.3e} Msun/h\n")

# print('Bose+ 2015')
# # print(f"Half-mode l:\t {bose_15_lambda_hm:.3f} Mpc/h"
# #       f"\nHalf-mode k:\t {bose_15_k_hm:.3f} h/Mpc")
# print(#f"z=0:"
#       f"\n Free streaming mass:\t {mass_free_streaming_z0:.3e} Msun/h"
#       f"\nHalf-mode mass: {bose_15_mass_hm:.3e} Msun/h")