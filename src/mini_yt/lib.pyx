# distutils: language = c++

import numpy as np

cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def obtain_relative_velocity_vector(
        data,
        field_names = (("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")),
        bulk_vector = "bulk_velocity"
    ):
    units = data[field_names[0]].units
    raise AttributeError