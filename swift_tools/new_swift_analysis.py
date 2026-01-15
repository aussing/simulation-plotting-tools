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
# log_filename = f'halo_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_filename = f'claudes_halo_analysis.log'
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
# LITTLEH = 1  # Dimensionless Hubble parameter

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
    
    #### Addded to create SM-HM comparison to G4

    swift_stellar_masses = np.array([r['Stellar_mass'] for r in cdm_results])
    
    
    # logger.info("Analysis completed")

if __name__ == "__main__":
    main()
