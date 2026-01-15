import numpy as np
import h5py 

def read_snap_header(sim_snap):
    
    box_size = np.float32(sim_snap['Header'].attrs['BoxSize'])
    mass_table = sim_snap['Header'].attrs['MassTable']
    num_part_total = sim_snap['Header'].attrs['NumPart_Total']
    redshift = sim_snap['Header'].attrs['Redshift']

    header = [box_size,mass_table,num_part_total,redshift]
    print(header)
    
    return header

def read_snap_params(sim_snap):
    little_h = sim_snap['Parameters'].attrs['HubbleParam']
    Omega_0 = sim_snap['Parameters'].attrs['Omega0']
    Omega_B = sim_snap['Parameters'].attrs['OmegaBaryon']
    Omega_L = sim_snap['Parameters'].attrs['OmegaLambda']
    # sigma_8 = sim_snap['Parameters'].attrs['Sigma8']

    params = [little_h,Omega_0,Omega_B,Omega_L]
    print(params)
    
    return params


def read_snap_data(sim_snap):
    dm_coords = np.array(sim_snap['PartType1']['Coordinates'], dtype=np.float64)
    dm_vels = np.array(sim_snap['PartType1']['Velocities'], dtype=np.float64)
    dm_ids = np.array(sim_snap['PartType1']['ParticleIDs'], dtype=np.float64)

    data = [dm_coords,dm_vels,dm_ids]
    return data

def write_new_file(new_file,header,params,data):

    with h5py.File(new_file,'w') as ds:
        ds.create_group('Header')
        ds['Header'].attrs['BoxSize']       = header[0]
        ds['Header'].attrs['MassTable']     = header[1]
        ds['Header'].attrs['NumPart_Total'] = header[2]
        ds['Header'].attrs['Redshift']      = header[3]

        ds.create_group('Parameters')
        ds['Parameters'].attrs['HubbleParam'] = params[0]
        ds['Parameters'].attrs['Omega0']      = params[1]
        ds['Parameters'].attrs['OmegaBaryon'] = params[2]
        ds['Parameters'].attrs['OmegaLambda'] = params[3]

        ds1 = ds.create_group('PartType1')
        ds1.create_dataset('Coordinates',dtype=np.float32,data=data[0])
        ds1.create_dataset('Velocities',dtype=np.float32,data=data[1])
        ds1.create_dataset('ParticleIDs',dtype=np.float32,data=data[2])


if __name__ == "__main__":
    i_file = 0
    sim_snap = f'/path/to/data/snapshot_{str(i_file).zfill(3)}.hdf5'
    sim_snap = h5py.File(sim_snap, 'r')
    header = read_snap_header(sim_snap)
    params = read_snap_params(sim_snap)
    data = read_snap_data(sim_snap)

    new_file = f'/path/to/new/data/snapshot_{str(i_file).zfill(3)}_converted.hdf5'
    write_new_file(new_file,header,params,data)
    
    
