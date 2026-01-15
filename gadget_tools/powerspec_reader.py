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

LITTLEH   = 0.6688
UNITMASS  = 1.0e10
M_WDM = 2
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
    # k_smooth = np.geomspace()
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

def calculate_analytical(redshift, wdm_mass, k_nyquist, with_nl=True):
    # print("Plotting analytic power spectra ")
    k = np.logspace(-1, k_nyquist+0.2*k_nyquist, 80)

    cosmo = cosmology.Cosmology(h=0.6688,Omega0_b=0.045,Omega0_cdm=OMEGA_DM,n_s=0.9626,P_k_max=2e2)
    cosmo = cosmo.match(sigma8=0.840)
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')
    Plin_vals = Plin(k)

    if with_nl:
        Pnonlin = cosmology.HalofitPower(cosmo, redshift=redshift)
        Pnonlin_vals = Pnonlin(k)
    else:
        Pnonlin_vals = np.nan

    # plt.loglog(k, Plin(k), c='black',label='Linear CDM') #*k**3/(2*np.pi)
    # plt.loglog(k, Pnonlin(k), c='red',label='Non-Linear CDM')

    alpha = 0.049 * (wdm_mass/1)**(-1.11) * (OMEGA_DM/0.25)**(0.11) * (LITTLEH/0.7)**1.22
    T2 = ((1+(alpha*k)**(2*1.12))**(-5/1.12))**2
    # plt.loglog(k, Plin(k)*T2, c='purple',label='Linear WDM')
    
    return k, Plin_vals, Plin_vals*T2, Pnonlin_vals, Pnonlin_vals*T2

# def plot(k_bins,P_k,label,color):
#     print('Plotting power spectra')
#     fig = plt.loglog(k_bins,P_k,label=label,c=color)
#     plt.savefig(f'powerspec_test.png',dpi=400)

if __name__ == "__main__":
    file_num = 26
    # file_num = 64 # z=10
    folder = 'N2048_L65_sd46371'
    # folder = 'N512_L65_sd12345'

    # snap_directory = f'/fred/oz217/aussing/{folder}/cdm/coarse/output/snapshot_{str(file_num).zfill(3)}.hdf5'
    snap_directory = f'/fred/oz217/aussing/a_knebe/cdm/output/snapshot_{str(file_num).zfill(3)}.hdf5'
    # snap_directory = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/snapshot_{str(file_num).zfill(3)}.hdf5'
    snap_data = h5py.File(snap_directory, 'r')  

    
    redshift = snap_data['Header'].attrs['Redshift']
    scale_factor = snap_data['Header'].attrs['Time']
    boxlength = snap_data['Header'].attrs['BoxSize']
    num_part =  np.sum(snap_data['Header'].attrs['NumPart_Total'])
    print(num_part**(1/3))
    # redshift  = 39
    # boxlength = 25.0

    redshift = np.round(redshift,2)
    # print(scale_factor)

    # File_cdm = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # File_wdm = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_010/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    
    # File_cdm = f'/fred/oz217/aussing/{folder}/cdm/coarse/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # File_wdm = f'/fred/oz217/aussing/{folder}/wdm_3.5/coarse/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'

    File_cdm = f'/fred/oz217/aussing/a_knebe/cdm/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    File_wdm = f'/fred/oz217/aussing/a_knebe/wdm_2/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # File_wdm_05 = f'/fred/oz217/aussing/a_knebe/wdm_0.5/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    
    cdm_powerspec_b1, cdm_powerspec_b2, cdm_powerspec_b3 = read_file(File_cdm)
    wdm_powerspec_b1, wdm_powerspec_b2, wdm_powerspec_b3 = read_file(File_wdm)
    # wdm_05_powerspec_b1, wdm_05_powerspec_b2, wdm_05_powerspec_b3 = read_file(File_wdm_05)

    nyquist = np.pi/(boxlength/num_part**(1/3))
    k_nyquist_cdm = (np.abs(cdm_powerspec_b1[:,0] - (nyquist+0.1*nyquist)).argmin())
    k_nyquist_wdm = (np.abs(wdm_powerspec_b1[:,0] - (nyquist+0.1*nyquist)).argmin())
    k_nyquist = int(np.mean([k_nyquist_cdm,k_nyquist_wdm]))


    k_max = np.log10(cdm_powerspec_b1[k_nyquist,0])
    # k_max = np.log10(3e2)
    
    print(f'k max = {np.log10(cdm_powerspec_b1[k_nyquist,0])}')
    k, Pcdm, Pwdm,Pnonlin_cdm,Pnonlin_wdm = calculate_analytical(redshift, 2, k_max, with_nl=False)
    plt.loglog(k,Pcdm,label=r'$\Lambda$CDM',c='blue')
    

    plt.loglog(cdm_powerspec_b1[0:k_nyquist,0],cdm_powerspec_b1[0:k_nyquist,2]*(boxlength)**3,c='black', alpha=1, label=r'$\Lambda$CDM simulation')
    plt.loglog(k,Pwdm,label=r'$\Lambda$WDM',c='orange')
    plt.loglog(wdm_powerspec_b1[0:k_nyquist,0],wdm_powerspec_b1[0:k_nyquist,2]*(boxlength)**3,c='red', alpha=1, label=r'$\Lambda$WDM simulation')
    # plt.loglog(cdm_powerspec_b1[0:k_nyquist,0],cdm_powerspec_b1[0:k_nyquist,4],c='purple',label='noise')

    # plt.loglog(cdm_powerspec_b1[:,0],cdm_powerspec_b1[:,2]*(boxlength)**3,c='black', alpha=1, label=f'CDM simulation')
    # plt.loglog(wdm_powerspec_b1[:,0],wdm_powerspec_b1[:,2]*(boxlength)**3,c='red', alpha=1, label=f'WDM simulation')
    # x_cdm, p_cdm = smooth_powerspec(cdm_powerspec_b1)
    # x_wdm, p_wdm = smooth_powerspec(wdm_powerspec_b1)
    # plt.loglog(x_cdm, p_cdm*(boxlength)**3,c='black', alpha=1, label=f'cdm')
    # plt.loglog(x_wdm, p_wdm*(boxlength)**3,c='red', alpha=1, label=f'wdm')


    # plt.loglog(k,Pnonlin_cdm,label="CDM",c='blue')
    # plt.loglog(k,Pnonlin_wdm,label="CDM",c='blue')
    # k, Pcdm, Pwdm,Pnonlin_cdm,Pnonlin_wdm = calculate_analytical(redshift,   1, k_max, with_nl=False)
    # plt.loglog(k,Pwdm,label="1KeV",c='red')

### plot extra lines
    # file_num = 85 # z=4
    # snap_directory = f'/fred/oz217/aussing/a_knebe/cdm/output/snapshot_{str(file_num).zfill(3)}.hdf5'
    # snap_data = h5py.File(snap_directory, 'r')  
    # redshift = snap_data['Header'].attrs['Redshift']
    # boxlength = snap_data['Header'].attrs['BoxSize']
    # redshift = np.round(redshift,2)
    # File_cdm = f'/fred/oz217/aussing/a_knebe/cdm/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # File_wdm = f'/fred/oz217/aussing/a_knebe/wdm_2/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # cdm_powerspec_b1, cdm_powerspec_b2, cdm_powerspec_b3 = read_file(File_cdm)
    # wdm_powerspec_b1, wdm_powerspec_b2, wdm_powerspec_b3 = read_file(File_wdm)
    # nyquist = np.pi/(boxlength/512)
    # k_nyquist_cdm = (np.abs(cdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    # k_nyquist_wdm = (np.abs(wdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    # k_nyquist = int(np.mean([k_nyquist_cdm,k_nyquist_wdm]))
    # plt.loglog(cdm_powerspec_b1[:,0],cdm_powerspec_b1[:,2]*(boxlength)**3,c='blue', alpha=0.7)#, label=f'z={redshift}')
    # plt.loglog(wdm_powerspec_b1[:,0],wdm_powerspec_b1[:,2]*(boxlength)**3,c='orange', alpha=0.7)#, label=f'z={redshift}')

    # file_num = 127 # z=0
    # snap_directory = f'/fred/oz217/aussing/a_knebe/cdm/output/snapshot_{str(file_num).zfill(3)}.hdf5'
    # snap_data = h5py.File(snap_directory, 'r')  
    # redshift = snap_data['Header'].attrs['Redshift']
    # boxlength = snap_data['Header'].attrs['BoxSize']
    # redshift = np.round(redshift,2)
    # File_cdm = f'/fred/oz217/aussing/a_knebe/cdm/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # File_wdm = f'/fred/oz217/aussing/a_knebe/wdm_2/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # cdm_powerspec_b1, cdm_powerspec_b2, cdm_powerspec_b3 = read_file(File_cdm)
    # wdm_powerspec_b1, wdm_powerspec_b2, wdm_powerspec_b3 = read_file(File_wdm)
    # nyquist = np.pi/(boxlength/512)
    # k_nyquist_cdm = (np.abs(cdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    # k_nyquist_wdm = (np.abs(wdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    # k_nyquist = int(np.mean([k_nyquist_cdm,k_nyquist_wdm]))
    # plt.loglog(cdm_powerspec_b1[:,0],cdm_powerspec_b1[:,2]*(boxlength)**3,c='blue',label=f'CDM')
    # plt.loglog(wdm_powerspec_b1[:,0],wdm_powerspec_b1[:,2]*(boxlength)**3,c='orange',label=f'2KeV WDM')
###

####### Plot analytical stuff
    plt.axvline(2*np.pi/boxlength,c='k',ls='--') # K-min
    plt.axvline(np.pi/(boxlength/num_part**(1/3)),c='k',ls='--') # K-max
    # plt.axvline(np.pi/(boxlength/256),c='k',ls='--') # K-max
    # plt.axvline(np.pi/(boxlength/512),c='k',ls='--') # K-max
    # plt.axvline(np.pi/(boxlength/2048),c='k',ls='--') # K-max

    # plt.axhline((boxlength)**3/num_part,c='k',ls=':')
#######    
####### Adding text labels
    # ax = plt.gca()
    # t = plt.text(0.65,0.75,r'256',ha='left', va='bottom',transform=ax.transAxes,fontsize=30.0,weight='bold',color='black')
    # t = plt.text(0.74,0.7,r'512',ha='left', va='bottom',transform=ax.transAxes,fontsize=30.0,weight='bold',color='black')
    # t = plt.text(0.82,0.65,r'1024',ha='left', va='bottom',transform=ax.transAxes,fontsize=30.0,weight='bold',color='black')
#######
    # plt.ylim(1e-1,3e4)
    # plt.ylim(1e-7,5e0)
    plt.ylim(1e-9,1e1)
    plt.xlim(8e-2,1e2)
    plt.legend(loc='lower left')
    plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)$ $[h^{-3} \mathrm{Mpc}^{3}]$")
    plt.title(f'Redshift = {redshift}')
    plt.tight_layout()
    # fig_name = f'powerspec_z_{redshift}.png'
    fig_name = f'powerspec_z_final.png'
    plt.savefig(f'{fig_name}',dpi=400)
    print(f'saved figure {fig_name}')
    # plt.savefig(f'powerspec_combined.png',dpi=400)
    
    # file_num = 127 # z=0

    # snap_directory = f'/fred/oz217/aussing/a_knebe/cdm/output/snapshot_{str(file_num).zfill(3)}.hdf5'
    # snap_data = h5py.File(snap_directory, 'r')  

    
    # redshift = snap_data['Header'].attrs['Redshift']
    # boxlength = snap_data['Header'].attrs['BoxSize']
    # print(redshift)
    # redshift = np.round(redshift,2)
    # print(redshift)
    # File_cdm = f'/fred/oz217/aussing/a_knebe/cdm/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # File_wdm = f'/fred/oz217/aussing/a_knebe/wdm_2/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    # File_wdm_05 = f'/fred/oz217/aussing/a_knebe/wdm_0.5/output/powerspecs/powerspec_{str(file_num).zfill(3)}.txt'
    
    # cdm_powerspec_b1, cdm_powerspec_b2, cdm_powerspec_b3 = read_file(File_cdm)
    # wdm_powerspec_b1, wdm_powerspec_b2, wdm_powerspec_b3 = read_file(File_wdm)
    # # wdm_05_powerspec_b1, wdm_05_powerspec_b2, wdm_05_powerspec_b3 = read_file(File_wdm_05)

    # nyquist = np.pi/(boxlength/512)
    # k_nyquist_cdm = (np.abs(cdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    # k_nyquist_wdm = (np.abs(wdm_powerspec_b1[:,0] - (nyquist+0.5*nyquist)).argmin())
    # k_nyquist = int(np.mean([k_nyquist_cdm,k_nyquist_wdm]))

    # plt.loglog(cdm_powerspec_b1[:,0],cdm_powerspec_b1[:,2]*(boxlength)**3,c='blue',label='CDM')
    # plt.loglog(wdm_powerspec_b1[:,0],wdm_powerspec_b1[:,2]*(boxlength)**3,c='orange',label='WDM')