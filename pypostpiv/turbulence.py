"""A set of functions for analysis of turbulent quantities.
"""

def kinetic_energy(field):
    """Calculates turbulent kinetic energy.

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return 0.5*(field.rms()**2).fsum(0)

def covariance(field):
    """Calculates the covariance of turbulent fluctuations.

    Author(s)
    ---------
    Jia Cheng Hu
    """
    field_fluc = field - field.fmean()
    return (field_fluc.u(0)*field_fluc.u(1)).fmean()
