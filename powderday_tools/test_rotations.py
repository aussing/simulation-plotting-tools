import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
from tqdm import trange
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from scipy.stats import bootstrap
from matplotlib import gridspec
import cmasher as cmr

def rotation_matrix(phi,omega,theta):
    u_x = np.cos(theta) * np.cos(omega)
    u_y = np.cos(theta) * np.sin(omega)
    u_z = -np.sin(theta)
    
    v_x = -np.cos(phi) * np.sin(omega) + np.sin(phi) * np.sin(theta) * np.cos(omega)
    v_y = np.cos(phi) * np.cos(omega) + np.sin(phi) * np.sin(theta) * np.sin(omega)
    v_z = np.sin(phi) * np.cos(theta)

    w_x = np.sin(phi) * np.sin(omega) + np.cos(phi) * np.sin(theta) * np.cos(omega)
    w_y = -np.sin(phi) * np.cos(omega) + np.cos(phi) * np.sin(theta) * np.sin(omega)
    w_z = np.cos(phi) * np.cos(theta)

    rot_mat = [[u_x,v_x,w_x],
               [u_y,v_y,w_y],
               [u_z,v_z,w_z]]

    return np.array(rot_mat)

def rotate_and_mask(halo_pos,extent,exclusion,dust_coords,roation_maxtrix):
    # rotate halo pos
    halo_pos_rot = np.matmul(halo_pos,roation_maxtrix) 
    # rotate coordinates 
    rot_pos = np.matmul(dust_coords,roation_maxtrix) 
    
    # calculate distances
    dists_rot = np.sqrt((rot_pos[:,0]-halo_pos_rot[0])**2+(rot_pos[:,1]-halo_pos_rot[1])**2+(rot_pos[:,2]-halo_pos_rot[2])**2) 
    # print(softening)
    # if axis == 'z':
    dist_axis = np.abs(rot_pos[:,2]-halo_pos_rot[2])
    dist_mask = (dists_rot<=extent) & (dist_axis>=exclusion)  # & (dists_rot>=softening)
    # if axis == 'y':
    #     dist_axis = rot_pos[:,1]-halo_pos_rot[1]
    #     dist_mask = (dists_rot<=extent) & (dist_axis>=softening) # & (dists_rot>=softening)
    # if axis == 'x':
    #     dist_axis = rot_pos[:,0]-halo_pos_rot[0]
    #     dist_mask = (dists_rot<=extent) & (dist_axis>=softening) # & (dists_rot>=softening)

    rad_pos_x = rot_pos[:,0][dist_mask]#-halo_pos[0]
    rad_pos_y = rot_pos[:,1][dist_mask]#-halo_pos[1]
    rad_pos_z = rot_pos[:,2][dist_mask]#-halo_pos[2]

    masked_coords = np.column_stack((rad_pos_x,rad_pos_y,rad_pos_z))

    return masked_coords,dist_mask

if __name__ =='__main__':

    halo_pos = np.array([0,0,0])
    extent = 5
    dust_coords = np.array([[1,1,1],[-1,-1,-1]])
    rot_mat_z = rotation_matrix(0,0,np.pi/4) # Apply to xy
    new_coords = np.matmul(dust_coords,rot_mat_z)
    
    print(np.round(new_coords,4))