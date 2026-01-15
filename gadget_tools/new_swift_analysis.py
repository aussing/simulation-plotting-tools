#!/usr/bin/env python3
"""
Halo Comparison Tool: Analyzes and compares dust distribution in matched CDM/WDM halos.

This script compares properties of dark matter haloes between Cold Dark Matter (CDM) 
and Warm Dark Matter (WDM) cosmological simulations, focusing on the dust mass 
distribution in Milky Way-sized halos.

Usage:
    python read_halo_cats.py

Author: [Junior Developer's Name]
Revised by: Darren Croton
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os
import sys
import cmasher as cmr
from scipy.spatial import distance_matrix
import logging
from collections import namedtuple
from datetime import datetime

# Add additional paths for custom modules
sys.path.insert(0, '/fred/oz381/cpower/analysistools')
import merger_tree_tools as mtt
import snapshot_tools as st
import halo_tools as ht
import galaxy_tools as gt
import analysis_tools as at


# Set up logging
log_filename = f'SM_HM_swift.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure plotting
plt.style.use('/home/aussing/sty.mplstyle')
cmap = cmr.torch

# Define named tuple for organizing halo data
HaloData = namedtuple('HaloData', [
    'index', 'pos', 'mass', 'radius', 'structure_type', 
    'gas_coords', 'gas_mass', 'dust_mass', 'metallicity'
])

# Configuration parameters
CONFIG = {
    'cdm_path': '/fred/oz390/L25/N376/',
    'wdm_path': '/fred/oz381/cpower/L25N376_WDM0pt5/',
    'snap_number': 31,
    'min_mass': 0.55e12,  # Milky Way min mass range in Msun
    'max_mass': 2.62e12,  # Milky Way max mass range in Msun
    'match_distance_threshold': 0.15,  # Mpc
    'hist_bin_size': 2.5,  # kpc
    'figures_dir': './hists/',
    'dust_to_metals_ratio': 0.4,
}

# Physical constants
UNITMASS = 1e10  # Msun
LITTLEH = 0.6688  # Dimensionless Hubble parameter
GASTYPE    = 0
HIGHDMTYPE = 1
STARTYPE   = 4
LOWDMTYPE  = 5
UNIT_LENGTH_FOR_PLOTS = 'kpc'
SOLMASSINGRAMS = 1.989e+33
# LITTLEH = 1  # Dimensionless Hubble parameter

##############
def get_sim_data(sim_directory,i_file):
    snap_fname     = f'/snapshot_{str(i_file).zfill(3)}.hdf5'
    snap_directory = sim_directory + snap_fname
    snap_data     = h5py.File(snap_directory, 'r')
    
    # haloinfo_fname     = f'/fof_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_fname     = f'/fof_subhalo_tab_{str(i_file).zfill(3)}.hdf5'
    haloinfo_directory = sim_directory + haloinfo_fname
    haloinfo_data = h5py.File(haloinfo_directory, 'r')

    z = (snap_data['Header'].attrs['Redshift'])
    # print(z)
    return snap_data, haloinfo_data, z

def analytic_SMHM():
    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('behroozi10',redshift=0)
    halo_mass = np.logspace(11.5, 12.5, 100)
    # halo_mass = np.logspace(10, 14, 100)
    mean_sm = model.mean_stellar_mass(prim_haloprop = halo_mass)
    return halo_mass, mean_sm
    
def MWA_stellar_halo_mass(sim_directory,i_file):
    snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)
    halo_M200c     = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
    halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH 
    
    mass_mask = np.argsort(halo_M200c)[::-1]
    halo_mainID = np.where(halo_masstypes[mass_mask,LOWDMTYPE] == 0)[0][0]
    
    return halo_M200c[mass_mask[halo_mainID]], halo_masstypes[mass_mask[halo_mainID],4]

def stellar_mass_halo_mass(sim_directory,label,color,i_file):
        # print('\n',label)
        snap_data, haloinfo_data, z = get_sim_data(sim_directory,i_file)

        halo_mass  = np.array(haloinfo_data['Group']['GroupMass'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_M200c = np.array(haloinfo_data['Group']['Group_M_Crit200'], dtype=np.float64) * UNITMASS / LITTLEH
        halo_masstypes = np.array(haloinfo_data['Group']['GroupMassType'], dtype=np.float64) * UNITMASS / LITTLEH

        # print(halo_masstypes)
        clean_haloes = np.where(halo_masstypes[:,5]==0)[0]

        plt.scatter(halo_M200c[clean_haloes],halo_masstypes[clean_haloes,4],label=label,color=color)

def get_unit_len(snapshot):
    unit_length = snapshot["Parameters"].attrs['UnitLength_in_cm']
    
    return unit_length

def set_plot_len(data, unit=UNIT_LENGTH_FOR_PLOTS):
    if unit in ['Mpc','mpc']:
        data = data/3.085678e24
    elif unit in ['Kpc','kpc']:
        data = data/3.085678e21
    elif unit == 'pc':
        data = data/3.085678e18
    else:
        print("What units do you want?????!!! AARRHH")
        raise TypeError
    return data

def get_snapshot(folder,snap_num,dm='cdm',SN_fac='sn_010'):

    snapshot = h5py.File(f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/snapshot_{str(snap_num).zfill(3)}.hdf5")
    
    return snapshot

def load_grid_data(folder,method,dm,SN_fac,snap_num):
    data = np.load(f'/fred/oz217/aussing/clean_pd/powderday/{folder}/{method}/{dm}/{SN_fac}/snap_{str(snap_num).zfill(3)}/grid_physical_properties.{str(snap_num).zfill(3)}_galaxy0.npz')

    return data

def set_center(folder,dm,SN_fac,snap_num):
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')

    halo_pos   = np.array(halo_data['Group']['GroupPos'], dtype=np.float64) / LITTLEH
    halo_M200c = np.array(halo_data['Group']['Group_M_Crit200'], dtype=np.float64) / LITTLEH * UNITMASS
    halo_masstypes = np.array(halo_data['Group']['GroupMassType'], dtype=np.float64) / LITTLEH  * UNITMASS
    R200c = np.array(halo_data['Group']['Group_R_Crit200'], dtype=np.float64) / LITTLEH
    Group_SFR = np.array(halo_data['Group']['GroupSFR'], dtype=np.float64) 
    mass_mask = np.argsort(halo_M200c)[::-1] #Sort by most massive halo
    halo_mainID = np.where(halo_masstypes[mass_mask,5] == 0)[0][0] #Select largest non-contaminated halo / resimulation target
    
    # print(f'halo mass = {halo_M200c[mass_mask][halo_mainID]/UNITMASS*LITTLEH}e10')
    snapshot = get_snapshot(folder,snap_num,dm,SN_fac)
    unit_length = get_unit_len(snapshot)

    halo_pos = np.array(halo_pos[mass_mask][halo_mainID]) * unit_length
    halo_rad = R200c[mass_mask][halo_mainID] * unit_length
    stellar_mass = halo_masstypes[mass_mask][halo_mainID][4]
    gas_mass_200c = halo_masstypes[mass_mask][halo_mainID][0]
    sfr = Group_SFR[mass_mask][halo_mainID] 
    print(f"stellar mass = {stellar_mass:.3e} Msun")
    print(f"SFR          = {np.round(sfr,4)} Msun/yr")
    # print(f"Halo Rad = {halo_rad}, halo pos = {halo_pos}\n")
    # extent = halo_rad*2
    extent = halo_rad


    return halo_pos, extent, stellar_mass, sfr, gas_mass_200c

def get_extent(halo_pos,extent):

    xmin,xmax  = halo_pos[0]-extent,halo_pos[0]+extent
    ymin,ymax  = halo_pos[1]-extent,halo_pos[1]+extent
    zmin,zmax  = halo_pos[2]-extent,halo_pos[2]+extent
    
    return xmin,xmax,ymin,ymax,zmin,zmax

def make_mask(pos_x,pos_y,pos_z,halo_pos,extent):
    xmin,xmax,ymin,ymax,zmin,zmax = get_extent(halo_pos,extent)

    x_mask = (pos_x>=xmin) & (pos_x<=xmax)
    y_mask = (pos_y>=ymin) & (pos_y<=ymax)
    z_mask = (pos_z>=zmin) & (pos_z<=zmax)
    
    pos_mask = x_mask & y_mask & z_mask

    return pos_mask

def dust_mass(dm,SN_fac,label,folder,method,num_bins,snap_num):
    print(dm,'-',SN_fac)
    
    halo_pos, extent, stellar_mass, sfr, gas_mass_200c = set_center(folder,dm,SN_fac,snap_num)
    halo_file = f"/fred/oz217/aussing/{folder}/{dm}/zoom/output/{SN_fac}/fof_subhalo_tab_{str(snap_num).zfill(3)}.hdf5"
    halo_data = h5py.File(halo_file, 'r')
    a = halo_data['Header'].attrs['Time']
    z = halo_data['Header'].attrs['Redshift']
    # print(halo_pos)
    # method = out_folder.split('/')[-2]

    grid_physical_properties_data = load_grid_data(folder,method,dm,SN_fac,snap_num)
    
    pos_x = grid_physical_properties_data['gas_pos_x'] * 3.085678e18 / a# convert parsecs to cm, convert back later
    pos_y = grid_physical_properties_data['gas_pos_y'] * 3.085678e18 / a
    pos_z = grid_physical_properties_data['gas_pos_z'] * 3.085678e18 / a
    # print(np.min(grid_physical_properties_data['gas_pos_x'])/1e6*LITTLEH, np.max(grid_physical_properties_data['gas_pos_x'])/1e6*LITTLEH)
    # print((np.max(grid_physical_properties_data['gas_pos_x']) - np.min(grid_physical_properties_data['gas_pos_x']))/2e6*LITTLEH)
    extent = np.float32(extent)
    extent_r200 = extent

    stellar_mass_array = np.geomspace(1e9,5e11,50)

    # M_0 = 3.98e10
    # alpha, beta, gamma = 0.14, 0.39, 0.10 
    # R_bar_late = gamma * (stellar_mass)**alpha * (1+stellar_mass/M_0)**(beta-alpha)
    # R_bar_late_array = gamma * (stellar_mass_array)**alpha * (1+stellar_mass_array/M_0)**(beta-alpha)

    # print(f'R_bar_late        = {R_bar_late} kpc')
    # print(f'extent/R_bar_late = {set_plot_len(extent)/R_bar_late} \n')

    # a, b = 0.56, 2.88e-6
    # R_bar_early = b * (stellar_mass)**a
    # R_bar_early_array = b * (stellar_mass_array)**a
    # print(f'R_bar_early        = {R_bar_early} kpc')
    # print(f'extent/R_bar_early = {set_plot_len(extent)/R_bar_early} \n')

    # fig,ax = plt.subplots()
    # plt.loglog(stellar_mass_array, R_bar_early_array, label='Early')
    # plt.loglog(stellar_mass_array, R_bar_late_array, label='Late')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./R_50_profiles.png',dpi=250)
    # plt.close()
    
    
    # extent = extent * 1.0

    pos_mask = make_mask(pos_x,pos_y,pos_z,halo_pos,extent)
    pos_mask_rad200 = make_mask(pos_x,pos_y,pos_z,halo_pos,extent_r200)

    extent_25kpc = 25 * 3.085678e21
    pos_mask_25kpc = make_mask(pos_x,pos_y,pos_z,halo_pos,extent_25kpc)
    
    extent_15kpc = 15 * 3.085678e21
    pos_mask_15kpc = make_mask(pos_x,pos_y,pos_z,halo_pos,extent_15kpc)
    
    gas_mass_r200c   = grid_physical_properties_data['particle_gas_mass'][pos_mask_rad200] / SOLMASSINGRAMS
    gas_mass_15kpc   = grid_physical_properties_data['particle_gas_mass'][pos_mask_15kpc] / SOLMASSINGRAMS
    # gas_mass         = grid_physical_properties_data['particle_gas_mass'][pos_mask_rad200] / SOLMASSINGRAMS

    dust_mass        = grid_physical_properties_data['particle_dustmass']#[pos_mask]
    dust_mass_rad200 = grid_physical_properties_data['particle_dustmass'][pos_mask_rad200]
    dust_mass_25kpc  = grid_physical_properties_data['particle_dustmass'][pos_mask_25kpc]
    dust_mass_15kpc  = grid_physical_properties_data['particle_dustmass'][pos_mask_15kpc]
    
    
    return dust_mass, np.sum(dust_mass_rad200), stellar_mass, sfr, gas_mass_200c


####################
def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def read_halo_catalogue(simulation_location, snap_number):
    """
    Read halo properties from VELOCIraptor output files.
    
    Parameters:
    -----------
    simulation_location : str
        Path to the simulation directory
    snap_number : int
        Snapshot number to read
        
    Returns:
    --------
    tuple
        Contains halo properties, group info, and particle data files
    """
    snap_str = str(snap_number).zfill(4)
    base_path = f'{simulation_location}/halos/snap_{snap_str}.VELOCIraptor'
    
    try:
        halo_properties = h5py.File(f'{base_path}.properties.0', 'r')
        halo_group_info = h5py.File(f'{base_path}.catalog_groups.0', 'r')
        group_particles = h5py.File(f'{base_path}.catalog_particles.0', 'r')
        
        return halo_properties, halo_group_info, group_particles
    except Exception as e:
        logger.error(f"Failed to read halo catalogue: {str(e)}")
        raise

def read_snapshot(simulation_location, snap_number):
    """
    Read simulation snapshot data.
    
    Parameters:
    -----------
    simulation_location : str
        Path to the simulation directory
    snap_number : int
        Snapshot number to read
        
    Returns:
    --------
    h5py.File
        Snapshot file handle
    """
    snap_str = str(snap_number).zfill(4)
    try:
        return h5py.File(f'{simulation_location}/snap_{snap_str}.hdf5')
    except Exception as e:
        logger.error(f"Failed to read snapshot: {str(e)}")
        raise

def find_matching_halos(dm_type, halo_positions, halo_masses, structure_types, mass_min, mass_max):
    """
    Find halos within a specified mass range that are not subhalos.
    
    Parameters:
    -----------
    dm_type : str
        Label for the dark matter type (for logging)
    halo_positions : np.ndarray
        Array of halo positions (x, y, z)
    halo_masses : np.ndarray
        Array of halo masses (M200c)
    structure_types : np.ndarray
        Array of halo structure types
    mass_min : float
        Minimum halo mass
    mass_max : float
        Maximum halo mass
        
    Returns:
    --------
    np.ndarray
        Indices of halos meeting the criteria
    """
    # Find halos in the correct mass range
    mass_mask = (halo_masses > mass_min) & (halo_masses < mass_max)
    
    # Find halos that are not subhalos (structure_type == 10)
    non_subhalo_mask = (structure_types == 10)
    
    # Combine masks
    combined_mask = mass_mask & non_subhalo_mask
    
    # Get indices
    indices = np.where(combined_mask)[0]
    
    logger.info(f"Found {len(indices)} {dm_type} halos in mass range {mass_min:.2e} - {mass_max:.2e} Msun")
    
    return indices

def match_halos_between_simulations(cdm_positions, wdm_positions, cdm_indices, wdm_indices, max_distance):
    """
    Match halos between CDM and WDM simulations based on spatial proximity.
    
    Parameters:
    -----------
    cdm_positions : np.ndarray
        Array of CDM halo positions
    wdm_positions : np.ndarray
        Array of WDM halo positions
    cdm_indices : np.ndarray
        Indices of CDM halos to match
    wdm_indices : np.ndarray
        Indices of WDM halos to match
    max_distance : float
        Maximum distance for matching (Mpc)
        
    Returns:
    --------
    list
        List of tuples containing (cdm_idx, wdm_idx, distance)
    """
    if len(cdm_indices) == 0 or len(wdm_indices) == 0:
        logger.warning("No halos to match in at least one simulation")
        return []
    
    # Extract positions for the indices we care about
    cdm_halo_positions = cdm_positions[cdm_indices]
    wdm_halo_positions = wdm_positions[wdm_indices]
    
    # Calculate distance matrix between all CDM and WDM halos
    dist = distance_matrix(cdm_halo_positions, wdm_halo_positions)
    
    matches = []
    
    # For each CDM halo, find the closest WDM halo
    for i, cdm_idx in enumerate(cdm_indices):
        min_idx = np.argmin(dist[i])
        min_dist = dist[i, min_idx]
        
        # Only accept matches within the distance threshold
        if min_dist <= max_distance:
            wdm_idx = wdm_indices[min_idx]
            matches.append((cdm_idx, wdm_idx, min_dist))
            
    logger.info(f"Matched {len(matches)} halos between CDM and WDM simulations")
    
    return matches

def select_gas_particles(snapshot, halo_pos, halo_rad):
    """
    Select gas particles within the halo radius.
    
    Parameters:
    -----------
    snapshot : h5py.File
        Snapshot file handle
    halo_pos : np.ndarray
        Halo position [x, y, z]
    halo_rad : float
        Halo radius
        
    Returns:
    --------
    tuple
        (gas_coords, gas_mass, particle_mask)
    """
    gas_coords = np.array(snapshot['PartType0']['Coordinates'], dtype=np.float64)
    gas_mass = np.array(snapshot['PartType0']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH
    
    # Calculate distances from halo center
    gas_distance = np.sqrt(
        (gas_coords[:, 0] - halo_pos[0])**2 + 
        (gas_coords[:, 1] - halo_pos[1])**2 + 
        (gas_coords[:, 2] - halo_pos[2])**2
    )
    
    # Create mask for particles within the halo radius
    gas_mask = (gas_distance <= halo_rad)
    
    # Return selected particles
    return gas_coords[gas_mask], gas_mass[gas_mask], gas_mask

def select_star_particles(snapshot, halo_pos, halo_rad):
    """
    Select gas particles within the halo radius.
    
    Parameters:
    -----------
    snapshot : h5py.File
        Snapshot file handle
    halo_pos : np.ndarray
        Halo position [x, y, z]
    halo_rad : float
        Halo radius
        
    Returns:
    --------
    tuple
        (gas_coords, gas_mass, particle_mask)
    """
    star_coords = np.array(snapshot['PartType4']['Coordinates'], dtype=np.float64)
    star_mass = np.array(snapshot['PartType4']['Masses'], dtype=np.float64) * UNITMASS / LITTLEH
    
    # Calculate distances from halo center
    star_distance = np.sqrt(
        (star_coords[:, 0] - halo_pos[0])**2 + 
        (star_coords[:, 1] - halo_pos[1])**2 + 
        (star_coords[:, 2] - halo_pos[2])**2
    )
    
    # Create mask for particles within the halo radius
    star_mask = (star_distance <= halo_rad)
    
    # Return selected particles
    return star_coords[star_mask], star_mass[star_mask]

def calculate_dust_mass(snapshot, particle_mask):
    """
    Calculate dust mass for selected gas particles.
    
    Parameters:
    -----------
    snapshot : h5py.File
        Snapshot file handle
    particle_mask : np.ndarray
        Boolean mask for particles to include
        
    Returns:
    --------
    tuple
        (dust_mass_array, average_metallicity)
    """
    gas_mass = np.array(snapshot['PartType0']['Masses'], dtype=np.float64)[particle_mask] * UNITMASS / LITTLEH
    gas_metallicity = np.array(snapshot['PartType0']['MetalMassFractions'], dtype=np.float64)[particle_mask]
    
    # Calculate dust mass
    dust_mass = gas_mass * gas_metallicity * CONFIG['dust_to_metals_ratio']
    
    # Calculate average metallicity
    avg_metallicity = np.mean(gas_metallicity) if len(gas_metallicity) > 0 else 0.0
    
    return dust_mass, avg_metallicity

def process_halo_data(dm_type, snapshot, halo_pos, halo_rad):
    """
    Process halo data to extract gas and dust properties.
    
    Parameters:
    -----------
    dm_type : str
        Dark matter type label (for logging)
    snapshot : h5py.File
        Snapshot file handle
    halo_pos : np.ndarray
        Halo position [x, y, z]
    halo_rad : float
        Halo radius
        
    Returns:
    --------
    tuple
        (gas_coords, gas_mass, dust_mass, avg_metallicity)
    """
    try:
        # Select gas particles
        gas_coords, gas_mass, gas_mask = select_gas_particles(snapshot, halo_pos, halo_rad)
        stellar_coords, stellar_mass = select_star_particles(snapshot, halo_pos, halo_rad)
        # print(f"HALO RAD = {halo_rad}")
        # Calculate dust mass
        dust_mass, avg_metallicity = calculate_dust_mass(snapshot, gas_mask)
        
        logger.info(f"{dm_type} halo: {len(gas_coords)} gas particles, "
                   f"total gas mass: {np.sum(gas_mass):.2e} Msun, "
                   f"total dust mass: {np.sum(dust_mass):.2e} Msun")
        
        return gas_coords, gas_mass, dust_mass, avg_metallicity, stellar_mass
    
    except Exception as e:
        logger.error(f"Error processing {dm_type} halo data: {str(e)}")
        raise

def create_dust_map(coords, dust_mass, halo_rad, bin_size, output_path, title=None):
    """
    Create a 2D histogram showing the dust distribution.
    
    Parameters:
    -----------
    coords : np.ndarray
        Particle coordinates
    dust_mass : np.ndarray
        Dust masses for each particle
    halo_rad : float
        Halo radius for setting plot limits
    bin_size : float
        Size of histogram bins (kpc)
    output_path : str
        Path to save the output image
    title : str, optional
        Plot title
        
    Returns:
    --------
    tuple
        (histogram, bin_edges_x, bin_edges_y)
    """
    # Convert halo_rad to kpc and calculate number of bins
    rad_kpc = halo_rad * 1000
    num_bins = int(rad_kpc / bin_size)
    
    # Make sure we have a reasonable number of bins
    num_bins = max(10, min(100, num_bins))
    
    # Create 2D histogram
    hist, bins_x, bins_y = np.histogram2d(
        coords[:, 0], coords[:, 2],
        bins=num_bins,
        weights=dust_mass
    )
    
    # Calculate bin size in physical units for normalization
    bin_size_norm = (bins_x[1] - bins_x[0]) * 1000 * (bins_y[1] - bins_y[0]) * 1000
    
    # Create figure
    plt.figure(figsize=(15, 15))
    
    # Plot histogram
    im = plt.imshow(
        hist.T / bin_size_norm,
        origin='lower',
        extent=(np.min(bins_x), np.max(bins_x), np.min(bins_y), np.max(bins_y)),
        norm=mpl.colors.LogNorm(vmin=1e-2, vmax=1e6),
        cmap=cmap
    )
    
    # Add colorbar and labels
    cbar = plt.colorbar(im)
    cbar.set_label('Dust mass density [M$_{\odot}$/kpc$^2$]')
    
    plt.xlabel('X [Mpc]')
    plt.ylabel('Z [Mpc]')
    
    if title:
        plt.title(title)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    return hist, bins_x, bins_y

def compile_halo_data(dm_type, halo_properties, indices):
    """
    Compile basic halo data for a set of indices.
    
    Parameters:
    -----------
    dm_type : str
        Dark matter type label (for logging)
    halo_properties : h5py.File
        Halo properties file
    indices : np.ndarray
        Indices of halos to compile data for
        
    Returns:
    --------
    list
        List of HaloData objects
    """
    halos = []
    
    for idx in indices:
        pos = np.array([
            halo_properties['Xc'][idx],
            halo_properties['Yc'][idx],
            halo_properties['Zc'][idx]
        ])
        
        mass = halo_properties['Mass_200crit'][idx] * UNITMASS
        radius = halo_properties['R_200crit'][idx]
        structure_type = halo_properties['Structuretype'][idx]
        
        # Create partial HaloData with fields that don't require processing particles
        halo = HaloData(
            index=idx,
            pos=pos,
            mass=mass,
            radius=radius,
            structure_type=structure_type,
            gas_coords=None,
            gas_mass=None,
            dust_mass=None,
            metallicity=None,
            stellar_mass=None
        )
        
        halos.append(halo)
    
    logger.info(f"Compiled basic data for {len(halos)} {dm_type} halos")
    
    return halos

def print_summary_statistics(cdm_results, wdm_results):
    """
    Print summary statistics comparing CDM and WDM halos.
    
    Parameters:
    -----------
    cdm_results : list
        List of dictionaries with CDM halo results
    wdm_results : list
        List of dictionaries with WDM halo results
    """
    # Convert lists to numpy arrays for easier computation
    cdm_halo_masses = np.array([r['halo_mass'] for r in cdm_results])
    # wdm_halo_masses = np.array([r['halo_mass'] for r in wdm_results])
    
    cdm_gas_masses = np.array([r['gas_mass'] for r in cdm_results])
    # wdm_gas_masses = np.array([r['gas_mass'] for r in wdm_results])
    
    cdm_dust_masses = np.array([r['dust_mass'] for r in cdm_results])
    # wdm_dust_masses = np.array([r['dust_mass'] for r in wdm_results])
    
    cdm_metallicities = np.array([r['metallicity'] for r in cdm_results])
    # wdm_metallicities = np.array([r['metallicity'] for r in wdm_results])

    cdm_stellar_mass = np.array([r['Stellar_mass'] for r in cdm_results])
    
    # Count comparisons
    # count_cdm_more_dust = sum(cdm_dust_masses > wdm_dust_masses)
    # count_wdm_more_dust = sum(wdm_dust_masses > cdm_dust_masses)
    # count_equal_dust = sum(np.isclose(cdm_dust_masses, wdm_dust_masses, rtol=1e-2))
    
    # count_cdm_more_mass = sum(cdm_halo_masses > wdm_halo_masses)
    # count_wdm_more_mass = sum(wdm_halo_masses > cdm_halo_masses)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*50)
    logger.info(f"Total matched halos analyzed: {len(cdm_results)}")
    logger.info("\nComparison Counts:")
    # logger.info(f"Galaxies with more CDM dust than WDM: {count_cdm_more_dust}")
    # logger.info(f"Galaxies with more WDM dust than CDM: {count_wdm_more_dust}")
    # logger.info(f"Galaxies with equal dust content:      {count_equal_dust}")
    # logger.info(f"Galaxies with heavier CDM halos:       {count_cdm_more_mass}")
    # logger.info(f"Galaxies with heavier WDM halos:       {count_wdm_more_mass}")
    
    logger.info("\nHalo Masses:")
    logger.info(f"CDM mean halo mass:   {np.mean(cdm_halo_masses):.4e} Msun")
    # logger.info(f"WDM mean halo mass:   {np.mean(wdm_halo_masses):.4e} Msun")
    logger.info(f"CDM median halo mass: {np.median(cdm_halo_masses):.4e} Msun")
    # logger.info(f"WDM median halo mass: {np.median(wdm_halo_masses):.4e} Msun")
    
    logger.info("\nGas Masses:")
    logger.info(f"CDM mean gas mass:    {np.mean(cdm_gas_masses):.4e} Msun")
    # logger.info(f"WDM mean gas mass:    {np.mean(wdm_gas_masses):.4e} Msun")
    logger.info(f"CDM median gas mass:  {np.median(cdm_gas_masses):.4e} Msun")
    # logger.info(f"WDM median gas mass:  {np.median(wdm_gas_masses):.4e} Msun")
    
    logger.info("\nDust Masses:")
    logger.info(f"CDM mean dust mass:   {np.mean(cdm_dust_masses):.4e} Msun")
    # logger.info(f"WDM mean dust mass:   {np.mean(wdm_dust_masses):.4e} Msun")
    logger.info(f"CDM median dust mass: {np.median(cdm_dust_masses):.4e} Msun")
    # logger.info(f"WDM median dust mass: {np.median(wdm_dust_masses):.4e} Msun")
    
    logger.info("\nMetallicities:")
    logger.info(f"CDM mean metallicity: {np.mean(cdm_metallicities):.4e}")
    # logger.info(f"WDM mean metallicity: {np.mean(wdm_metallicities):.4e}")

    logger.info("\nStellar mass:")
    logger.info(f"CDM mean stellar mass: {np.mean(cdm_stellar_mass):.4e}")


    logger.info("="*50 + "\n")


def main():
    """Main execution function."""
    # Ensure output directory exists
    ensure_dir(CONFIG['figures_dir'])
    
    logger.info("Starting halo comparison analysis")
    logger.info(f"CDM simulation path: {CONFIG['cdm_path']}")
    logger.info(f"WDM simulation path: {CONFIG['wdm_path']}")
    logger.info(f"Snapshot number: {CONFIG['snap_number']}")
    logger.info(f"Mass range: {CONFIG['min_mass']:.2e} - {CONFIG['max_mass']:.2e} Msun")
    
    # Read halo catalogues
    logger.info("Reading CDM halo catalogue...")
    cdm_halo_properties, cdm_halo_group_info, cdm_particles = read_halo_catalogue(
        CONFIG['cdm_path'], CONFIG['snap_number']
    )
    
    logger.info("Reading WDM halo catalogue...")
    wdm_halo_properties, wdm_halo_group_info, wdm_particles = read_halo_catalogue(
        CONFIG['wdm_path'], CONFIG['snap_number']
    )
    
    # Read snapshots
    logger.info("Reading CDM snapshot...")
    cdm_snapshot = read_snapshot(CONFIG['cdm_path'], CONFIG['snap_number'])
    
    logger.info("Reading WDM snapshot...")
    wdm_snapshot = read_snapshot(CONFIG['wdm_path'], CONFIG['snap_number'])
    
    # Extract halo properties
    cdm_m200c = np.array(cdm_halo_properties['Mass_200crit'], dtype=np.float64) * UNITMASS
    cdm_r200c = np.array(cdm_halo_properties['R_200crit'], dtype=np.float64)
    cdm_structure_type = np.array(cdm_halo_properties['Structuretype'], dtype=np.float32)
    cdm_x = np.array(cdm_halo_properties['Xc'], dtype=np.float64)
    cdm_y = np.array(cdm_halo_properties['Yc'], dtype=np.float64)
    cdm_z = np.array(cdm_halo_properties['Zc'], dtype=np.float64)
    cdm_positions = np.column_stack((cdm_x, cdm_y, cdm_z))
    
    wdm_m200c = np.array(wdm_halo_properties['Mass_200crit'], dtype=np.float64) * UNITMASS
    wdm_r200c = np.array(wdm_halo_properties['R_200crit'], dtype=np.float64)
    wdm_structure_type = np.array(wdm_halo_properties['Structuretype'], dtype=np.float32)
    wdm_x = np.array(wdm_halo_properties['Xc'], dtype=np.float64)
    wdm_y = np.array(wdm_halo_properties['Yc'], dtype=np.float64)
    wdm_z = np.array(wdm_halo_properties['Zc'], dtype=np.float64)
    wdm_positions = np.column_stack((wdm_x, wdm_y, wdm_z))
    
    # Find Milky Way-mass halos
    cdm_mw_indices = find_matching_halos(
        'CDM', cdm_positions, cdm_m200c, cdm_structure_type, 
        CONFIG['min_mass'], CONFIG['max_mass']
    )
    
    # wdm_mw_indices = find_matching_halos(
    #     'WDM', wdm_positions, wdm_m200c, wdm_structure_type,
    #     CONFIG['min_mass'], CONFIG['max_mass']
    # )
    
    # # Match halos between simulations
    # matches = match_halos_between_simulations(
    #     cdm_positions, wdm_positions, cdm_mw_indices, wdm_mw_indices,
    #     CONFIG['match_distance_threshold']
    # )
    
    # Analyze matched halos
    cdm_results = []
    wdm_results = []
    
    # for i, (cdm_idx, wdm_idx, distance) in enumerate(cdm_mw_indices):
    for i, (cdm_idx) in enumerate(cdm_mw_indices):
        logger.info(f"\nAnalyzing matched halo pair {i+1}/{len(cdm_mw_indices)}")
        logger.info(f"CDM index: {cdm_idx}")#, WDM index: {wdm_idx}, distance: {distance:.4f} Mpc")
        
        try:
            # Get halo properties
            cdm_halo_pos = cdm_positions[cdm_idx]
            cdm_halo_mass = cdm_m200c[cdm_idx]
            cdm_halo_rad = cdm_r200c[cdm_idx]
            
            # wdm_halo_pos = wdm_positions[wdm_idx]
            # wdm_halo_mass = wdm_m200c[wdm_idx]
            # wdm_halo_rad = wdm_r200c[wdm_idx]
            
            logger.info(f"CDM halo M200c: {cdm_halo_mass:.2e} Msun, R200c: {cdm_halo_rad*1000:.1f} kpc")
            # logger.info(f"WDM halo M200c: {wdm_halo_mass:.2e} Msun, R200c: {wdm_halo_rad*1000:.1f} kpc")
            
            # Process halo data
            cdm_gas_coords, cdm_gas_mass, cdm_dust_mass, cdm_metallicity, stellar_mass = process_halo_data(
                'CDM', cdm_snapshot, cdm_halo_pos, cdm_halo_rad
            )
            
            # wdm_gas_coords, wdm_gas_mass, wdm_dust_mass, wdm_metallicity = process_halo_data(
            #     'WDM', wdm_snapshot, wdm_halo_pos, wdm_halo_rad
            # )
            
            # Create dust maps
            cdm_output_path = os.path.join(CONFIG['figures_dir'], f'xz_cdm_halo_{i}.png')
            # wdm_output_path = os.path.join(CONFIG['figures_dir'], f'xz_wdm_halo_{i}.png')
            
            # create_dust_map(
            #     cdm_gas_coords, cdm_dust_mass, cdm_halo_rad, CONFIG['hist_bin_size'],
            #     cdm_output_path, f"CDM Dust Distribution - Halo {i}"
            # )
            
            # create_dust_map(
            #     wdm_gas_coords, wdm_dust_mass, wdm_halo_rad, CONFIG['hist_bin_size'],
            #     wdm_output_path, f"WDM Dust Distribution - Halo {i}"
            # )
            
            # Store results
            cdm_results.append({
                'halo_index': cdm_idx,
                'halo_mass': cdm_halo_mass,
                'gas_mass': np.sum(cdm_gas_mass),
                'dust_mass': np.sum(cdm_dust_mass),
                'metallicity': cdm_metallicity,
                'Stellar_mass': np.sum(stellar_mass)
            })
            
            # wdm_results.append({
            #     'halo_index': wdm_idx,
            #     'halo_mass': wdm_halo_mass,
            #     'gas_mass': np.sum(wdm_gas_mass),
            #     'dust_mass': np.sum(wdm_dust_mass),
            #     'metallicity': wdm_metallicity
            # })
            
            # Compare results
            # dust_ratio = np.sum(cdm_dust_mass) / np.sum(wdm_dust_mass)
            # mass_ratio = cdm_halo_mass / wdm_halo_mass
            
            # logger.info(f"Dust mass ratio (CDM/WDM): {dust_ratio:.2f}")
            # logger.info(f"Halo mass ratio (CDM/WDM): {mass_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing halo pair {i+1}: {str(e)}")
            continue
    
    # Print summary statistics
    
    print_summary_statistics(cdm_results, wdm_results)

    ##############################
    ##############################
    ##############################
    fig = plt.subplots()
    swift_halo_mass    = np.array([r['halo_mass'] for r in cdm_results])
    swift_stellar_mass = np.array([r['Stellar_mass'] for r in cdm_results])
    swift_gas_mass     = np.array([r['gas_mass'] for r in cdm_results])
    swift_dust_mass    = np.array([r['dust_mass'] for r in cdm_results])

    analytic_halo_mass, analytic_mean_sm = analytic_SMHM()
    folder_list = []
    for simulation_name in [ simulation_name for simulation_name in os.listdir('/fred/oz217/aussing/') if ( simulation_name.startswith("N2048_L65_sd") )]:
        folder_list.append(simulation_name)
    folder_list = np.sort(folder_list)
    CDM_halo_mass, CDM_stellar_mass = [], []
    WDM_halo_mass, WDM_stellar_mass = [], []

    for folder in folder_list:
        # print(folder)
        cdm_file_loc = f'/fred/oz217/aussing/{folder}/cdm/zoom/output/sn_010/'
        wdm_file_loc = f'/fred/oz217/aussing/{folder}/wdm_3.5/zoom/output/sn_005/'
        # stellar_mass_halo_mass(cdm_file_loc,'','blue',26)
        # stellar_mass_halo_mass(wdm_file_loc,'','orange',26)
        try:
            CDM_halo_mass_iter, CDM_stellar_mass_iter = MWA_stellar_halo_mass(cdm_file_loc,26)
        except:
            print(f'Missing file in {folder}')
        CDM_halo_mass.append(CDM_halo_mass_iter)
        CDM_stellar_mass.append(CDM_stellar_mass_iter)
        
        try:
            WDM_halo_mass_iter, WDM_stellar_mass_iter = MWA_stellar_halo_mass(wdm_file_loc,26)
        except:
            print(f'Missing file in {folder}')
        WDM_halo_mass.append(WDM_halo_mass_iter)
        WDM_stellar_mass.append(WDM_stellar_mass_iter)
    
    CDM_halo_mass = np.array(CDM_halo_mass)
    CDM_stellar_mass = np.array(CDM_stellar_mass)

    WDM_halo_mass = np.array(WDM_halo_mass)
    WDM_stellar_mass = np.array(WDM_stellar_mass)

    plt.scatter(CDM_halo_mass,CDM_stellar_mass,c='blue',label='Gadget-4 CDM')
    plt.scatter(WDM_halo_mass,WDM_stellar_mass,c='orange',label='Gadget-4 WDM')
    plt.scatter(swift_halo_mass,swift_stellar_mass,c='r',label='SWIFT CDM')
    plt.plot(analytic_halo_mass,analytic_mean_sm, label="Behroozi+ (2010)",color='k', ls='-.')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$M_{200c}$')
    plt.ylabel(r'$M_{*}$')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./post_sub_tests/combined_sm_hm.png',dpi=250)
    
    fig = plt.subplots()
    method_list = ['dtm']#,'rr','li_bf']
    cdm_sn = 'sn_010'
    wdm_sn = 'sn_005'
    for method in method_list:     
        # for sn in sn_list:
            # for dm in dm_list :          
        cdm_stel_dust_mass, cdm_sfr_dust_mass, cdm_gas_mass = [], [], []
        wdm_stel_dust_mass, wdm_sfr_dust_mass, wdm_gas_mass = [], [], []
        fig=plt.subplots()
        for folder in folder_list:
            print()
            print(f'Simulation -> {folder}')
            
            snap_num = 26
            num_bins = 80

            cdm_dust_mass_total, cdm_dust_mass_rad200, cdm_stellar_mass, cdm_sfr, cdm_gas_mass_200c = dust_mass('cdm',cdm_sn,'',folder,method,num_bins,snap_num)
            wdm_dust_mass_total, wdm_dust_mass_rad200, wdm_stellar_mass, wdm_sfr, wdm_gas_mass_200c = dust_mass('wdm_3.5',wdm_sn,'',folder,method,num_bins,snap_num)

            cdm_stel_dust_mass.append((cdm_stellar_mass,cdm_dust_mass_rad200))
            cdm_sfr_dust_mass.append((cdm_sfr,cdm_dust_mass_rad200))
            
            wdm_stel_dust_mass.append((wdm_stellar_mass,wdm_dust_mass_rad200))
            wdm_sfr_dust_mass.append((wdm_sfr,wdm_dust_mass_rad200))

            cdm_gas_mass.append(cdm_gas_mass_200c)
            wdm_gas_mass.append(wdm_gas_mass_200c)
        
        cdm_stel_dust_mass = np.array(cdm_stel_dust_mass)
        wdm_stel_dust_mass = np.array(wdm_stel_dust_mass)

        print('Log10 stellar masses e10 Msun')
        print(np.log10(wdm_stel_dust_mass))
        # print('Log10 dust masses e7 Msun')
        # print(np.log10(wdm_stel_dust_mass[:,1]).T)

        plt.scatter(np.log10(cdm_stel_dust_mass[:,0]),np.log10(cdm_stel_dust_mass[:,1]),c='blue', label='CDM SN=10\%')
        plt.scatter(np.log10(wdm_stel_dust_mass[:,0]),np.log10(wdm_stel_dust_mass[:,1]),c='orange', label='WDM SN=5\%')
        plt.scatter(np.log10(swift_stellar_mass),np.log10(swift_dust_mass),c='red', label='SWIFT CDM')
        
        log_stellar_mass = np.geomspace(9,11.5,100)+0.23                
        peeples_14_fit = 0.86*(log_stellar_mass) - 1.31
        skibba_11_bar_fit  = log_stellar_mass - 2.84
        skibba_11_nobar_fit  = log_stellar_mass - 3.06
        # cortese_masses = np.array([8.62 , 9.06 , 9.48 , 9.99 , 10.46, 10.91])+0.23
        # cortese_DTS = np.array(   [-2.19, -2.24, -2.39, -2.87, -3.68, -3.34])
        cortese_masses = np.array([ 9.48 , 9.99 , 10.46, 10.91])+0.23
        cortese_DTS = np.array(   [ -2.39, -2.87, -3.68, -3.34])

        plt.plot(log_stellar_mass,skibba_11_nobar_fit,c='k',ls='-.',label='Skibba+ (2011)') ## NO BAR
        plt.scatter(cortese_masses, cortese_masses+cortese_DTS, s=200,marker="+",c='k', label='Cortese+ (2012)')
        plt.plot(log_stellar_mass,peeples_14_fit,c='k',ls='--',label='Peeples+ (2014)')
        # plt.plot(log_stellar_mass,skibba_11_bar_fit,c='k',ls=':',label='Skibba+11 barred')
        
        plt.xlabel(r'M$_{*}$ [M$_{\odot}$]')
        plt.ylabel(r'M$_{{dust}}$ [M$_{\odot}$]')
        plt.legend(loc='upper left',fontsize='large')
        plt.tight_layout()
        plt.savefig(f'./post_sub_tests/stellar_dust_mass_{method}.png',dpi=250)
        
        fig=plt.subplots()
        cdm_gas_mass = np.array(cdm_gas_mass)
        wdm_gas_mass = np.array(wdm_gas_mass)

        plt.scatter(np.log10(cdm_gas_mass),np.log10(cdm_stel_dust_mass[:,1]),c='blue', label='CDM SN=10\%')
        plt.scatter(np.log10(wdm_gas_mass),np.log10(wdm_stel_dust_mass[:,1]),c='orange', label='WDM SN=5\%')
        plt.scatter(np.log10(swift_gas_mass),np.log10(swift_dust_mass),c='red', label='SWIFT CDM')
        log_gas_mass = np.geomspace(9,12,100)
        li_draine_01 = log_gas_mass+np.log10(0.00813)
        zubko_04_min = log_gas_mass+np.log10(0.00568)
        zubko_04_max = log_gas_mass+np.log10(0.00648)

        plt.plot(log_gas_mass,li_draine_01,color='k',ls=':',label='Li+Draine 01')
        plt.fill_between(log_gas_mass,zubko_04_min,zubko_04_max,alpha=0.3,color='k',label='Zubko+04')
        # plt.plot(log_gas_mass,zubko_04_max,c='k',ls='-.',label='Zubko+04 max')

        plt.xlabel(r'M$_{gas}$ [M$_{\odot}$]')
        plt.ylabel(r'M$_{dust}$ [M$_{\odot}$]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'./post_sub_tests/gas_dust_mass_{method}.png',dpi=250)
        

        # cdm_sfr_dust_mass  = np.array(cdm_sfr_dust_mass)
        # wdm_sfr_dust_mass  = np.array(wdm_sfr_dust_mass)

        # fig=plt.subplots()
        # sfr_space = np.logspace(-1,0.8)
        # da_cunha_10_dust_mass = 1.28e7 * sfr_space**1.11
        # plt.scatter(np.log10(cdm_sfr_dust_mass[:,0]),(cdm_sfr_dust_mass[:,1]),c='blue', label="CDM SN=10\%")
        # plt.scatter(np.log10(wdm_sfr_dust_mass[:,0]),(wdm_sfr_dust_mass[:,1]),c='orange', label='WDM SN=5\%')
        # plt.plot(sfr_space,da_cunha_10_dust_mass,c='k',ls='--', label='Da Cunha+10')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel("SFR")
        # plt.ylabel("Dust Mass")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f'./post_sub_tests/SFR_dust_mass_{method}.png',dpi=250)    
    ##############################
    ##############################
    ##############################

    logger.info("Analysis completed")

if __name__ == "__main__":
    main()
