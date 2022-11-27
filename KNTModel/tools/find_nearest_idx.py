import numpy as np
from numba import njit
from numba import i8, f8

@njit(i8(f8, f8[:]))
def find_nearest_idx(x, array) -> int:
    """
    Find the closest value in array to x and return its index.
    
    Parameters
    ----------
    x : scalar
        value you are looking for
    array : 1 demension numpy array
        array of candidate values
    
    Returns
    -------
    int
        the index for the element closest to x in the array
    """
    return np.abs(array - x).argmin()