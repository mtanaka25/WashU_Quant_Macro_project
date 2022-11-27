import numpy as np
from numba import njit, prange
from numba import types, f8, i8, b1
from ..tools import find_nearest_idx

@njit(types.Tuple((f8[:, :], f8[:, :], b1))
      (f8[:,:], f8[:,:], f8[:,:],
       i8[:,:], i8[:,:], i8[:,:], f8[:], i8, f8))
def get_stationary_dist(trans_prob_z,
                        default_prob,
                        purchase_prob,
                        a_star_H_idx,
                        a_star_NP_idx,
                        a_star_NN_idx,
                        a_grid,
                        max_iter,
                        tol):
    trans_prob_z =  np.ascontiguousarray(trans_prob_z)
    # Get the number of grid points
    N_z = trans_prob_z.shape[0]
    N_a = default_prob.shape[0]
    # find the index satsfying a=0
    a0_idx = find_nearest_idx(0, a_grid)
    # initialize the distribution
    each_density = 1/ (N_z *  N_a * 2)
    pop_H = np.ones((N_a, N_z)) * each_density
    pop_N = np.ones((N_a, N_z)) * each_density
    # initialize the while loop
    diff, iteration = tol +1., 0
    while (iteration < max_iter) & (diff > tol):
        pop_H_pre, pop_N_pre = np.copy(pop_H), np.copy(pop_N)
        # reset the updated density
        pop_H *= 0.
        pop_N *= 0.
        # draw z' and the Gumbel shock
        H_to_H = ((1 - default_prob) * pop_H_pre) @ trans_prob_z
        H_to_N = (default_prob * pop_H_pre) @ trans_prob_z
        N_to_H = (purchase_prob * pop_N_pre) @ trans_prob_z
        N_to_N = ((1 - purchase_prob) * pop_N_pre) @ trans_prob_z
        # applying the optimal asset holdings
        for a_idx in prange(N_a):
            for z_idx in prange(N_z):
                pop_H[a_star_H_idx[a_idx, z_idx], z_idx] += H_to_H[a_idx, z_idx]
                pop_N[a0_idx, z_idx] += H_to_N[a_idx, z_idx]
                pop_H[a_star_NP_idx[a_idx, z_idx], z_idx] += N_to_H[a_idx, z_idx]
                pop_N[a_star_NN_idx[a_idx, z_idx], z_idx] += N_to_N[a_idx, z_idx]
        # Convergence check
        diff_pop_H = max(np.abs(pop_H - pop_H_pre).flatten())
        diff_pop_N = max(np.abs(pop_N - pop_N_pre).flatten())
        diff = max([diff_pop_H, diff_pop_N])
        # Progress the iteration counter
        iteration += 1
    flag = (diff > tol)
    return pop_H, pop_N, flag