"""
A set of functions for analysis of turbulent quantities.
"""

def turbulent_kinetic_energy(field):
    """
    Calculates turbulent kinetic energy.

    Parameters
    ----------
    field : Field2D

    Returns
    -------
    Field2D

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return 0.5*(field.rms()**2).fsum(0)

def reynolds_shear_stress(field):
    """
    Calculates the Reynolds shear stress of turbulent fluctuations.

    Parameters
    ----------
    field : Field2D

    Returns
    -------
    Field2D

    Author(s)
    ---------
    Jia Cheng Hu
    """
    field_fluc = field - field.fmean()
    return (field_fluc.u(0)*field_fluc.u(1)).fmean()
