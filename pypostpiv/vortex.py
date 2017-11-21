"""A set of functions for analysis of vortex dynamics
"""

def vorticity(field, method=None):
    """Compute the vorticity of the field

    Parameters
    ----------
    field : Field2D
    method: str, optional
        A method for calculation of voritcity. Valid options are:
        'circulation' - 
        ...

    Returns
    -------
    Field2D

    Author(s)
    ---------
    Jia Cheng Hu
    """
    if field.ftype() == 'vector':
        if method == 'circulation':
            u = field.u(0)
            v = field.u(1)
            dL = field.dL

            cir0 = u[:,2:,2:]  + 2*u[:,1:-1,2:]  + u[:,:-2,2:]
            cir1 = v[:,2:,2:]  + 2*v[:,2:,1:-1]  + v[:,2:,:-2]
            cir2 = u[:,2:,:-2] + 2*u[:,1:-1,:-2] + u[:,:-2,:-2]
            cir3 = v[:,:-2,2:] + 2*v[:,:-2,1:-1] + v[:,:-2,:-2]

            new_field = (cir0+cir1-cir2-cir3)/8/dL
            new_field.x = field.x[1:-1,1:-1]
            new_field.y = field.y[1:-1,1:-1]
            return new_field
        else:
            return field.u(1).ddx(method) - field.u(0).ddy(method)
    else:
        assert()

def lambda2(field, method=None):
    """
    Computes the lambda2 criterion of the field.

    Parameters
    ----------
    field : Field2D
    method: str, optional
        A method for calculation of the lambda2 criterion. Valid options are:
        ...

    Returns
    -------
    Field2D

    Author(s)
    ---------
    Jia Cheng Hu
    """
    ddx = field.ddx(method)
    ddy = field.ddy(method)
    return (ddx.u(0)+ddy.u(1))**2 - 4*(ddx.u(0)*ddy.u(1) - ddy.u(0)*ddx.u(1))
