"""A set of functions for analysis of turbulent quantities.
"""

import numpy as np

def vel_mag(u, v):
    """Returns the velocity magnitude (scalar) field.

    Parameters
    ----------

    Returns
    -------

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return np.sqrt(u**2+v**2)

def vel_mean(u, v):
    """Calculates the mean of the velocity field.

    Parameters
    ----------

    Returns
    -------

    Author(s)
    ---------
    Caddie Zhang, Jia Cheng Hu
    """
    return np.nanmean(u, axis=2), np.nanmean(v, axis=2)

def vel_mag_mean(u, v):
    """Calculates the mean of the velocity magnitude.

    Parameters
    ----------

    Returns
    -------

    Author(s)
    ---------
    Caddie Zhang, Jia Cheng Hu
    """
    return np.nanmean(np.sqrt(u**2+v**2), axis=2)

def vel_rms(u, v):
    """Calculates the root mean square of the velocity field.

    Parameters
    ----------

    Returns
    -------

    Author(s)
    ---------
    Caddie Zhang, Jia Cheng Hu
    """
    return np.nanstd(u, axis=2), np.nanstd(v, axis=2)

def kinetic_energy(u, v):
    """Calculates the turbulent kinetic energy.

    Parameters
    ----------

    Returns
    -------

    Author(s)
    ---------
    Caddie Zhang, Jia Cheng Hu
    """
    urms, vrms = vel_rms(u, v)
    return urms**2+vrms**2

def vel_covar(u, v):
    """Calculates the covariance of turbulent fluctuations.

    Parameters
    ----------

    Returns
    -------
    (u_rms, v_rms) : `tuple`

    Author(s)
    ---------
    Caddie Zhang, Jia Cheng Hu
    """
    uavg, vavg = vel_mean(u, v)
    uavg3d = uavg[:, :, np.newaxis]
    vavg3d = vavg[:, :, np.newaxis]
    return np.nanmean(((u-uavg3d)*(v-vavg3d)), axis=2)
