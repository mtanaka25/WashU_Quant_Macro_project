import numpy as np
from ...tools import gini_index
from numba import njit, prange
from numba import types, f8, i8

@njit(types.Tuple((f8[:,:,:,:], f8[:,:,:,:]))
      (f8[:,:,:], f8[:,:,:], i8[:], f8[:,:], f8[:,:,:], f8[:,:,:],
       i8[:,:,:], i8[:,:,:], i8[:,:,:], i8[:,:,:]))
def x_shock_simulation(init_dist_H,
                       init_dist_N,
                       x_idx_path,
                       trans_prob_z,
                       default_prob,
                       purchase_prob,
                       a_star_HR_idx,
                       a_star_HD_idx,
                       a_star_NP_idx,
                       a_star_NN_idx,
                       ):
    # simulation horizon
    T = len(x_idx_path)
    # Get the number of grid points
    N_a, N_z, N_x = init_dist_N.shape
    # Prepare array in which the distribution will be saved
    irf_dist_H = np.zeros((N_a, N_z, N_x, T+1))
    irf_dist_N = np.zeros((N_a, N_z, N_x, T+1))
    # Save the initial distribuion in these arrays
    irf_dist_H[:, :, :, 0] = init_dist_H
    irf_dist_N[:, :, :, 0] = init_dist_N
    # iterate the economy
    for t in range(T):
        xt_idx = x_idx_path[t]
        for zt_idx in prange(N_z):
            for a_pre_idx in range(N_a):
                for z_pre_idx in range(N_z):
                    for x_pre_idx in range(N_x):
                        H_pre = irf_dist_H[a_pre_idx, z_pre_idx, x_pre_idx, t]
                        N_pre = irf_dist_N[a_pre_idx, z_pre_idx, x_pre_idx, t]
                        Pz = trans_prob_z[z_pre_idx, zt_idx]
                        Pd = default_prob[a_pre_idx, z_pre_idx, x_pre_idx]
                        Pp = purchase_prob[a_pre_idx, z_pre_idx, x_pre_idx]
                        irf_dist_H[a_star_HR_idx[a_pre_idx, z_pre_idx, x_pre_idx], zt_idx, xt_idx, t+1] += (1-Pd) * Pz * H_pre
                        irf_dist_N[a_star_HD_idx[a_pre_idx, z_pre_idx, x_pre_idx], zt_idx, xt_idx, t+1] += Pd * Pz * H_pre
                        irf_dist_H[a_star_NP_idx[a_pre_idx, z_pre_idx, x_pre_idx], zt_idx, xt_idx, t+1] += Pp * Pz * N_pre
                        irf_dist_N[a_star_NN_idx[a_pre_idx, z_pre_idx, x_pre_idx], zt_idx, xt_idx, t+1] += (1-Pp) * Pz * N_pre
    return irf_dist_H, irf_dist_N

@njit(f8(f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:]))
def population_mean(x_H,
                    x_N,
                    distribution_H,
                    distribution_N):
    ave = 0.
    N_a, N_z, N_x = distribution_N.shape
    for a_idx in range(N_a):
        for z_idx in range(N_z):
            for x_idx in range(N_x):
                ave += x_H[a_idx, z_idx, x_idx] * distribution_H[a_idx, z_idx, x_idx]
                ave += x_N[a_idx, z_idx, x_idx] * distribution_N[a_idx, z_idx, x_idx]
    return ave

@njit(f8(f8[:,:,:], f8[:,:,:]))
def group_mean(x, distribution):
    ave = 0.
    total = np.sum(distribution)
    N_a, N_z, N_x = distribution.shape
    for a_idx in range(N_a):
        for z_idx in range(N_z):
            for x_idx in range(N_x):
                ave += x[a_idx, z_idx, x_idx] * distribution[a_idx, z_idx, x_idx] / total
    return ave

@njit(types.Tuple((f8[:], f8[:], f8[:], f8[:], f8[:]))
      (f8[:,:,:,:], f8[:,:,:,:], f8[:], f8[:,:,:], f8[:,:,:], f8[:,:,:]))
def calc_aggregate_irfs(irf_dist_H,
                        irf_dist_N,
                        a_grid,
                        rm,
                        default_prob,
                        purchase_prob,
                        ):
    # Obtain matrix size
    N_a, N_z, N_x, T = irf_dist_N.shape
    # Prepare auxiliary matrix for a
    a_array = np.zeros((N_a, N_z, N_x))
    for a_idx in prange(N_a):
        a_i = a_grid[a_idx]
        a_array[a_idx, :, :] = a_i
    # prepare arrays
    H_share_vec = np.zeros((T, ))
    ave_a_vec = np.zeros((T, ))
    ave_rm_vec = np.zeros((T, ))
    ave_Pd_vec = np.zeros((T, ))
    ave_Pp_vec = np.zeros((T, ))
    # Calculation
    for t in range(T):
        H_share_vec[t] = np.sum(irf_dist_H[:, :, :, t])
        ave_a_vec[t] = population_mean(a_array, a_array, irf_dist_H[:, :, :, t], irf_dist_N[:, :, :, t])
        ave_rm_vec[t] = group_mean(rm, irf_dist_H[:, :, :, t])
        ave_Pd_vec[t] = group_mean(default_prob, irf_dist_H[:, :, :, t])
        ave_Pp_vec[t] = group_mean(purchase_prob, irf_dist_N[:, :, :, t])
    return H_share_vec, ave_a_vec, ave_rm_vec, ave_Pd_vec, ave_Pp_vec

@njit(f8(f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:]))
def calc_population_gini(fx_H, fx_N, dist_H, dist_N):
    fx_H, fx_N = fx_H.flatten(), fx_N.flatten()
    dist_H, dist_N = dist_H.flatten(), dist_N.flatten()
    N_H, N_N = len(fx_H), len(fx_N)
    N = N_H + N_N
    fx = np.zeros((N, ))
    dist = np.zeros((N, ))
    for n in prange(N):
        if n < N_H:
            fx[n], dist[n] = fx_H[n], dist_H[n]
        else:
            fx[n], dist[n] = fx_N[n-N_H], dist_N[n-N_H]
    gini = gini_index(fx = fx, x_dist = dist)
    return gini

@njit(f8(f8[:,:,:], f8[:,:,:]))
def calc_group_gini(fx, dist):
    fx, dist = fx.flatten(), dist.flatten()
    gini = gini_index(fx = fx, x_dist = dist)
    return gini

@njit(types.Tuple((f8[:], f8[:], f8[:], f8[:]))
      (f8[:,:,:,:], f8[:,:,:,:], f8[:], f8[:,:,:], f8[:,:,:], f8[:,:,:]))
def calc_inequality_irfs(irf_dist_H,
                         irf_dist_N,
                         a_grid,
                         rm,
                         default_prob,
                         purchase_prob,
                         ):
    # Obtain matrix size
    N_a, N_z, N_x, T = irf_dist_N.shape
    # Prepare auxiliary matrix for a
    a_array = np.zeros((N_a, N_z, N_x))
    for a_idx in prange(N_a):
        a_i = a_grid[a_idx]
        a_array[a_idx, :, :] = a_i
    # prepare arrays
    gini_a_vec = np.zeros((T, ))
    gini_rm_vec = np.zeros((T, ))
    gini_Pd_vec = np.zeros((T, ))
    gini_Pp_vec = np.zeros((T, ))
    # Calculation
    for t in range(T):
        gini_a_vec[t] = calc_population_gini(a_array, a_array, irf_dist_H[:, :, :, t], irf_dist_N[:, :, :, t])
        gini_rm_vec[t] = calc_group_gini(rm, irf_dist_H[:, :, :, t])
        gini_Pd_vec[t] = calc_group_gini(default_prob, irf_dist_H[:, :, :, t])
        gini_Pp_vec[t] = calc_group_gini(purchase_prob, irf_dist_N[:, :, :, t])
    return gini_a_vec, gini_rm_vec, gini_Pd_vec, gini_Pp_vec