"""
Old implementation
"""

import h5py
#import numpy as np
from . import utilities
from . import turbulence

class Project2D2C:
    """A class describing a 2 dimensional, 2 component vector field.

    Author(s)
    ---------
    Jia Cheng Hu
    """

    def __init__(self, hdf5_file_path):
        f_handle = h5py.File(hdf5_file_path)

        self.data = utilities.load_from_hdf5(f_handle)
        f_handle.close()

    def list_data(self):
        """Lists all the data within the project.

        Author(s)
        ---------
        Jia Cheng Hu
        """
        utilities.print_dict_struct(self.data)

    def get_grid_size(self, cam_num=0):
        """Returns the grid size of the velocity field.

        Parameters
        ----------

        Returns
        -------
        (dx, dy) : `tuple`

        Author(s)
        ---------
        Jia Cheng Hu
        """
        sub_dict = self.data['cam_'+str(cam_num)]['grid_size']
        return sub_dict['dx'], sub_dict['dy']

    def get_velocity(self, cam_num=0):
        """Returns a velocity field.

        Parameters
        ----------

        Returns
        -------
        (x, y, u, v) : `tuple`

        Author(s)
        ---------
        Jia Cheng Hu
        """
        sub_dict = self.data['cam_'+str(cam_num)]['velocity']
        return sub_dict['x'], sub_dict['y'], sub_dict['u'], sub_dict['v']

    def get_vel_mag(self, cam_num=0):
        """Returns the  magnitude of a velocity field.

        Parameters
        ----------

        Returns
        -------
        (x, y, U) : `tuple`

        Author(s)
        ---------
        Jia Cheng Hu
        """
        # Check if the data is already computed, if not compute it
        if 'vel_mag' not in self.data['cam_'+str(cam_num)].keys():
            for key, value in self.data.items():
                vel = value['velocity']
                u_mag = turbulence.vel_mag(vel['u'], vel['v'])
                value['vel_mag'] = {'x':vel['x'], 'y':vel['y'], 'U':u_mag}

        sub_dict = self.data['cam_'+str(cam_num)]['vel_mag']
        return sub_dict['x'], sub_dict['y'], sub_dict['U']

    def get_vel_mean(self, cam_num=0):
        """Returns the mean of a velocity field.

        Parameters
        ----------

        Returns
        -------
        (x, y, u, v) : `tuple`

        Author(s)
        ---------
        Jia Cheng Hu
        """
        # Check if the data is already computed, if not compute it
        if 'vel_mean' not in self.data['cam_'+str(cam_num)].keys():
            for key, value in self.data.items():
                vel = value['velocity']
                u, v = turbulence.vel_mean(vel['u'], vel['v'])
                value['vel_mean'] = {'x':vel['x'], 'y':vel['y'], 'u':u, 'v':v}

        sub_dict = self.data['cam_'+str(cam_num)]['vel_mean']
        return sub_dict['x'], sub_dict['y'], sub_dict['u'], sub_dict['v']

    def get_vel_mag_mean(self, cam_num=0):
        """Returns the mean magnitude of a velocity field.

        Parameters
        ----------

        Returns
        -------
        (x, y, U) : `tuple`

        Author(s)
        ---------
        Jia Cheng Hu
        """

        # Check if the data is already computed, if not compute it
        if 'vel_mag_mean' not in self.data['cam_'+str(cam_num)].keys():
            for key, value in self.data.items():
                vel = value['velocity']
                U = turbulence.vel_mag_mean(vel['u'], vel['v'])
                value['vel_mag_mean'] = {'x':vel['x'], 'y':vel['y'], 'U':U}

        sub_dict = self.data['cam_'+str(cam_num)]['vel_mag_mean']
        return sub_dict['x'], sub_dict['y'], sub_dict['U']

    def get_turb_rms(self, cam_num=0):
        """Returns the RMS turbulence level.

        Parameters
        ----------

        Returns
        -------
        (x, y, u, v) : `tuple`

        Author(s)
        ---------
        Jia Cheng Hu
        """
        # Check if the data is already computed. If not, compute it.
        if 'turb_rms' not in self.data['cam_'+str(cam_num)].keys():
            for key, value in self.data.items():
                vel = value['velocity']
                u, v = turbulence.vel_rms(vel['u'], vel['v'])
                value['vel_rms'] = {'x':vel['x'], 'y':vel['y'], 'u':u, 'v':v}

        sub_dict = self.data['cam_'+str(cam_num)]['vel_rms']
        return sub_dict['x'], sub_dict['y'], sub_dict['u'], sub_dict['v']

    def get_turb_ke(self, cam_num=0):
        """Returns turbulent kinetic energy.

        Parameters
        ----------

        Returns
        -------
        (x, y, ke) : `tuple`
        Author(s)
        ---------
        Jia Cheng Hu
        """
        # Check if the data is already computed, if not compute it
        if 'turb_ke' not in self.data['cam_'+str(cam_num)].keys():
            for key, value in self.data.items():
                vel = value['velocity']
                ke = turbulence.kinetic_energy(vel['u'], vel['v'])
                value['turb_ke'] = {'x':vel['x'], 'y':vel['y'], 'ke':ke}

        sub_dict = self.data['cam_'+str(cam_num)]['turb_ke']
        return sub_dict['x'], sub_dict['y'], sub_dict['ke']

    def get_turb_covar(self, cam_num=0):
        """Returns turbulence covariance.

        Parameters
        ----------

        Returns
        -------
        (x, y, uv) : `tuple`

        Author(s)
        ---------
        Jia Cheng Hu
        """
        # Check if the data is already computed, if not computed it
        if 'turb_covar' not in self.data['cam_'+str(cam_num)].keys():
            for key, value in self.data.items():
                vel = value['velocity']
                uv = turbulence.vel_covar(vel['u'], vel['v'])
                value['turb_covar'] = {'x':vel['x'], 'y':vel['y'], 'uv':uv}

        sub_dict = self.data['cam_'+str(cam_num)]['turb_covar']
        return sub_dict['x'], sub_dict['y'], sub_dict['uv']
