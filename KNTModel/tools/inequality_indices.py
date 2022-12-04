import numpy as np
from numba import njit, prange, f8
from .find_nearest_idx import find_nearest_idx

@njit(f8[:,:](f8[:],f8[:]))
def lorenz_curve(fx, x_dist):
    # Check if the sum of the distribution is unity
    if np.sum(x_dist) != 1:
        x_dist = x_dist / np.sum(x_dist)
    # Sort the data
    sorted_idx = fx.argsort()
    x_dist = x_dist[sorted_idx]
    fx.sort()
    # Calculate the cumulative share in aggregate earnings
    fx_contrib = fx * x_dist
    cum_fx_share = np.cumsum(fx_contrib)/np.sum(fx_contrib)
    # Calculate the cumulative share in total samples
    cum_N_share = np.cumsum(x_dist)/np.sum(x_dist)
    # Combine the two series into an array (and insert 0 in the first column)
    N = len(cum_fx_share)
    lorenz_curve = np.zeros((2, N+1))
    for n in prange(N):
        lorenz_curve[0, n+1] = cum_N_share[n]
        lorenz_curve[1, n+1] = cum_fx_share[n]
    return lorenz_curve

@njit(f8(f8[:],f8[:]))
def gini_index(fx, x_dist):
    # Prepare the Lorenz curve
    lc = lorenz_curve(fx, x_dist)
    # Numerical integration for the area below the Lorenz curve
    area_below_lc = 0.
    for i in range(lc.shape[-1] -1):
        area_below_lc += (lc[1, i] + lc[1, i+1]) * (lc[0, i+1] - lc[0, i]) * 0.5
    # Calculate the Gini coefficient
    gini_index = (0.5 -  area_below_lc) / 0.5
    return gini_index