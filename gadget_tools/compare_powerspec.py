# from asyncore import read
from cmath import exp, nan
# from sklearn.metrics import max_error
from turtle import left, position
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import binarize
from sqlalchemy import between
from sympy import threaded
from nbodykit.source.catalog import HDFCatalog
from nbodykit.lab import cosmology
from nbodykit.lab import *
from matplotlib import gridspec

plt.style.use('/home/aussing/sty.mplstyle')
# plt.rcParams['text.usetex'] = True

LITTLEH   = 0.6688
UNITMASS  = 1.0e10
M_WDM = 3.5 
OMEGA_DM = 0.321



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

def smooth_powerspec(powerspec):
    print('Smoothing power spectra')
    num_smooth_bins=1000
    Pk_smooth = np.zeros(num_smooth_bins)
    k_smooth = np.zeros(num_smooth_bins)
    dk_smooth = (np.max(powerspec[:,0]) - np.min(powerspec[:,0]))/num_smooth_bins
    # dk_smooth = (np.log10(np.max(powerspec[:,0])) - np.log10(np.min(powerspec[:,0])))/(num_smooth_bins)
    # dk_smooth = np.log10(dk_smooth)
    count = np.min(powerspec[:,0])
    index=0
    while index < num_smooth_bins:
        min_k = (np.searchsorted(powerspec[:,0],count,side='right'))
        max_k = (np.searchsorted(powerspec[:,0],count+dk_smooth,side='right'))
        if min_k == max_k:
            Pk_smooth[index] = np.sum((powerspec[min_k,2]) * powerspec[min_k,3]) / np.sum(powerspec[min_k,3])
            k_smooth[index] = (np.sum(powerspec[min_k,0] * powerspec[min_k,3]) / np.sum(powerspec[min_k,3]))
        else:
            Pk_smooth[index] = np.sum((powerspec[min_k:max_k,2]) * powerspec[min_k:max_k,3]) / np.sum(powerspec[min_k:max_k,3])
            k_smooth[index] = (np.sum(powerspec[min_k:max_k,0] * powerspec[min_k:max_k,3]) / np.sum(powerspec[min_k:max_k,3]))
        count=count+dk_smooth
        index= index+1
    k_smooth = k_smooth[:-5]
    Pk_smooth =  Pk_smooth[:-5]
    return k_smooth, Pk_smooth

def calculate_analytical(redshift, wdm_mass,k_nyquist):
    # print("Plotting analytic power spectra ")
    k = np.logspace(-1, k_nyquist+0.2*k_nyquist, 80)
    cosmo = cosmology.Cosmology(h=0.6688,Omega0_b=0.045,Omega0_cdm=OMEGA_DM,n_s=0.9626)
    cosmo = cosmo.match(sigma8=0.840)
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')
    # plt.loglog(k, Plin(k), c='black',label='Linear CDM') #*k**3/(2*np.pi)

    # cosmo_nl = cosmology.Cosmology(h=0.703,Omega0_b=0.045,Omega0_cdm=OMEGA_DM,P_k_max=2e2,n_s=0.961)
    # cosmo_nl = cosmo_nl.match(sigma8=0.811)
    # Pnonlin = cosmology.HalofitPower(cosmo_nl, redshift=redshift)
    # plt.loglog(k, Pnonlin(k), c='red',label='Non-Linear CDM')

    alpha = 0.049 * (wdm_mass/1)**(-1.11) * (OMEGA_DM/0.25)**(0.11) * (LITTLEH/0.7)**1.22
    T2 = ((1+(alpha*k)**(2*1.12))**(-5/1.12))**2
    # plt.loglog(k, Plin(k)*T2, c='purple',label='Linear WDM')
    return k, Plin(k), Plin(k)*T2#, Pnonlin(k)

def plot(k_bins,P_k,label,color):
    print('Plotting power spectra')
    fig = plt.plot(k_bins,P_k,label=label,c=color)
    fig = plt.xscale('log')
    fig = plt.yscale('log')

def plot_compare(powerspec_reference,powerspec_comparison,k,color,label):
    # comparison = 100 * (powerspec_comparison-powerspec_reference)/powerspec_reference
    comparison = 100 * (powerspec_comparison/powerspec_reference-1)
    plt.plot(k,comparison,color=color,label=label)#+"-1")
    plt.xscale('log')

if __name__ == "__main__":
    I_FILE = 26
    folder = 'N2048_L65_sd46371'

    File_cdm = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/powerspecs/powerspec_{str(I_FILE).zfill(3)}.txt'
    File_wdm = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/powerspecs/powerspec_{str(I_FILE).zfill(3)}.txt'

    snap_directory = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/snapshot_{str(I_FILE).zfill(3)}.hdf5'
    snap_data = h5py.File(snap_directory, 'r')

    redshift = snap_data['Header'].attrs['Redshift']
    # redshift = 127
    boxsize = snap_data['Header'].attrs['BoxSize']
    # boxsize = 25
    num_part =  np.sum(snap_data['Header'].attrs['NumPart_Total'])
    # num_part = 256**3

    print(f"Redshift = {redshift}, Boxsize = {boxsize}")
    cdm_powerspec_b1,        cdm_powerspec_b2,        cdm_powerspec_b3        = read_file(File_cdm)
    wdm_powerspec_b1,        wdm_powerspec_b2,        wdm_powerspec_b3        = read_file(File_wdm)
    # wdm_powerspec_b1_second, wdm_powerspec_b2_second, wdm_powerspec_b3_second = read_file(File_wdm_second)
    
    # smooth_k_cdm, smooth_pow_cdm = smooth_powerspec(cdm_powerspec_b1)
    # smooth_k_cdm_b2, smooth_pow_cdm_b2 = smooth_powerspec(cdm_powerspec_b2)

    # smooth_k_wdm, smooth_pow_wdm = smooth_powerspec(wdm_powerspec_b1)
    # smooth_k_wdm_b2, smooth_pow_wdm_b2 = smooth_powerspec(wdm_powerspec_b2)
    
    # smooth_k_wdm_second, smooth_pow_wdm_second = smooth_powerspec(wdm_powerspec_b1_second)
    nyquist = np.pi/(boxsize/num_part**(1/3))
    # k_nyquist_cdm = (np.abs(cdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    # k_nyquist_wdm = (np.abs(wdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    k_nyquist_cdm = (np.abs(cdm_powerspec_b1[:,0] - (nyquist)).argmin())
    k_nyquist_wdm = (np.abs(wdm_powerspec_b1[:,0] - (nyquist)).argmin())
    k_nyquist = int(np.mean([k_nyquist_cdm,k_nyquist_wdm]))

    print(np.log10(cdm_powerspec_b1[k_nyquist,0]))

    wdm_mass = M_WDM
    k_ana, Pcdm_ana , Pwdm_ana   = calculate_analytical(redshift, wdm_mass,np.log10(cdm_powerspec_b1[k_nyquist,0])) #, P_nonlin
    
    graphs = ['Spectrum','Comparison']
    # graphs = ['Spectrum']
    # graphs = ['Comparison']    

    if graphs == ['Spectrum','Comparison']:
        fig = plt.figure()
        
        gs = gridspec.GridSpec(2,1, height_ratios=[4, 1])
        # fig,(ax1,ax2) = plt.subplots(2,1)
        ax1 = plt.subplot(gs[0])
        # fig.subplots_adjust(hspace=0)
        # ax1.plot(cdm_powerspec_b1[0:k_nyquist,0],cdm_powerspec_b1[0:k_nyquist,2]*(boxsize)**3,label="CDM",color='blue')
        # ax1.plot(wdm_powerspec_b1[0:k_nyquist,0],wdm_powerspec_b1[0:k_nyquist,2]*(boxsize)**3,label="WDM",color='orange')
        ax1.plot(k_ana, Pwdm_ana,label='Analytic WDM',color='red')
        ax1.plot(k_ana, Pcdm_ana,label='Analytic CDM',color='k')
        ax1.axvline(2*np.pi/boxsize,c='k',ls='--')
        ax1.axvline(np.pi/(boxsize/num_part**(1/3)),c='k',ls='--')
        ax1.legend(loc='lower left')
        ax1.set_ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^{3}]$", fontsize=30)
        ax1.set_yscale('log')
        # ax1.set_xtixks([])
        
        ax2 = plt.subplot(gs[1], sharex=ax1)
        comparison_ana = 100 * (Pwdm_ana/Pcdm_ana-1)
        comparison_meas = 100 * (wdm_powerspec_b1[0:k_nyquist_wdm,2]/cdm_powerspec_b1[0:k_nyquist_cdm,2]-1)
        ax2.plot(k_ana,comparison_ana,label='Analytic difference',color='k')
        # ax2.plot(cdm_powerspec_b1[0:k_nyquist_cdm,0],comparison_meas,label='WDM / CDM',color='orange')
        ax2.axvline(2*np.pi/boxsize,c='k',ls='--')
        ax2.axvline(np.pi/(boxsize/num_part**(1/3)),c='k',ls='--')
        ax2.axhline(0,c='k',ls='--',alpha=0.5)
        # ax2.legend()
        ax2.set_ylim([-100,25])
        ax2.set_ylabel("\% Difference", fontsize=30)
        
        
        plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
        plt.tight_layout()
        
        plt.xscale('log')
        plt.subplots_adjust(hspace=.0)
        
        plt.savefig(f'./{folder}/residual_powerspec_{int(redshift)}.png',dpi=400) #,bbox_inches='tight'
        # plt.savefig(f'./residual_powerspec_{int(redshift)}.png',dpi=400) #,bbox_inches='tight'
        

    elif graphs == ['Spectrum']:
        fig = plt.close()
        
        plot(cdm_powerspec_b1[0:k_nyquist_cdm,0],cdm_powerspec_b1[0:k_nyquist_cdm,2]*(boxsize)**3,"CDM",'blue')
        plot(wdm_powerspec_b1[0:k_nyquist_wdm,0],wdm_powerspec_b1[0:k_nyquist_wdm,2]*(boxsize)**3,"WDM",'orange')
        
        # plot(smooth_k_cdm,smooth_pow_cdm*(boxsize)**3,"smoothed CDM powerspec",'blue')
        # plot(smooth_k_wdm,smooth_pow_wdm*(boxsize)**3,"smoothed WDM powerspec",'orange')
    
        plot(k_ana, Pwdm_ana,'Analytic WDM','red')
        plot(k_ana, Pcdm_ana,'Analytic CDM','k')
        
        fig = plt.axvline(2*np.pi/boxsize,c='k',ls='--')
        fig = plt.axvline(np.pi/(boxsize/num_part**(1/3)),c='k',ls='--')
        # fig = plt.axvline(np.pi/(boxsize/num_part),c='k',ls='--')
        # fig = plt.axhline((boxsize/num_part**(1/3))**3,c='k',ls=':')
        
        fig = plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
        fig = plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^{3}]$")
        fig = plt.legend()
        fig = plt.savefig(f'./{folder}/powerspecs_z_{int(redshift)}.png',dpi=400) #,bbox_inches='tight'
        plt.close()
        
    elif graphs == ['Comparison']:
        k_nyquist = int(np.mean([k_nyquist_cdm,k_nyquist_wdm]))
        # plot_compare(smooth_pow_cdm,smooth_pow_wdm,smooth_k_cdm,'blue','Power spectra comparison')
        # plot_compare(smooth_pow_cdm_b2,smooth_pow_wdm_b2,smooth_k_cdm_b2, 'orange', 'Folded spectra comparison')
        plot_compare(Pcdm_ana,Pwdm_ana,k_ana,'k','Analytic difference')
        fig = plt.ylim((-100,100))
        fig = plt.axhline(0,c='k',ls='--',alpha=0.5)
        fig = plt.axvline(2*np.pi/boxsize,c='k',ls='--')
        fig = plt.axvline(np.pi/(boxsize/num_part**(1/3)),c='k',ls='--')
        plot_compare(cdm_powerspec_b1[0:k_nyquist_cdm,2],wdm_powerspec_b1[0:k_nyquist_wdm,2],cdm_powerspec_b1[0:k_nyquist_cdm,0],'orange','WDM / CDM')
        # plt.title(f'Power spectrum difference at z = {np.round(redshift,2)}')
        fig = plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
        fig = plt.ylabel("\% difference")
        fig = plt.tight_layout()
        fig = plt.legend()
        fig = plt.savefig(f'./{folder}/compare_powerspec_z_{int(redshift)}.png',dpi=400)#,bbox_inches='tight'
    
    print('Saving figures')