import numpy as np

def vel_mag(u, v):
    '''
    - Velocity magnitude
    - Return U
    - Auther: Jia Cheng Hu
    '''
    return np.sqrt(u**2+v**2)

def vel_mean(u, v):
    '''
    - Mean of the velocity vector field
    - Return u, v
    - Auther: Caddie Zhang, Jia Cheng Hu
    '''
    return np.nanmean(u, axis=2), np.nanmean(v, axis=2)

def vel_mag_mean(u, v):
    '''
    - Mean of the velocity magnitude
    - Return U
    - Auther: Caddie Zhang, Jia Cheng Hu
    '''
    return np.nanmean(np.sqrt(u**2+v**2), axis=2)

def vel_rms(u, v):
    '''
    - Root mean square of the velocity
    - Return u_rms, v_rms
    - Auther: Caddie Zhang, Jia Cheng Hu
    '''
    return np.nanstd(u, axis=2), np.nanstd(v, axis=2)

def kinetic_energy(u, v):
    '''
    - Tubulent kinetic energy of the field
    - Return u_rms, v_rms
    - Auther: Caddie Zhang, Jia Cheng Hu
    '''
    urms, vrms = vel_rms(u, v)
    return urms**2+vrms**2

def vel_covar(u, v):
    '''
    - Tubulent flutuation coverance of the field
    - Return u_rms, v_rms
    - Auther: Caddie Zhang, Jia Cheng Hu
    '''
    uavg, vavg = vel_mean(u, v)
    uavg3d = uavg[:, :, np.newaxis]
    vavg3d = vavg[:, :, np.newaxis]
    return np.nanmean(((u-uavg3d)*(v-vavg3d)), axis=2)
