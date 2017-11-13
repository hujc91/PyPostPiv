"""A set of functions for analysis of vortex dynamics
"""

def vorticity(field, method=None):
    """Compute the vorticity of the field

    Author(s)
    ---------
    Jia Cheng Hu
    """
    if field.ftype() == '2D Vector':
        return field.u(1).ddx(method) - field.u(0).ddy(method)
    else:
        assert()
