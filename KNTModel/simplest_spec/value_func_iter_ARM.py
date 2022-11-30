import numpy as np
from numba import njit, types
from numba import f8, i8, b1
from .value_func import *
from ..tools import find_nearest_idx

@njit(types.Tuple((f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], i8[:,:,:], i8[:,:,:], i8[:,:,:], i8[:,:,:], b1))
      (f8[:], f8[:,:], f8[:], f8[:,:], f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8,
       f8, f8, f8, f8, f8, f8, f8, f8, i8, f8))
def value_func_iter(z_grid, trans_prob_z, x_grid, trans_prob_x, a_grid,
                    beta, alpha, sigma, gamma, d, ph, h_star, h_eps, c_d,
                    c_hat_d, delta, a_d_max, lmbd, r, theta, kappaH,
                    kappaN, max_iter, tol):
    # find the largest index satisfying a_d_max
    a_d_max_idx = find_nearest_idx(a_d_max, a_grid)
    if a_grid[a_d_max_idx] > a_d_max:
        a_d_max_idx -= 1
    # Prepare initial guesses for values and mortgage rates
    N_a, N_z, N_x = len(a_grid), len(z_grid), len(x_grid),
    VH_init  = np.zeros((N_a, N_z, N_x))
    VN_init  = np.zeros((N_a, N_z, N_x))
    rm_init = np.ones((N_a, N_z, N_x)) * 1.1 * r
    # Prepare arrays to save intermediate values
    V_HR_t, V_HD_t = np.zeros((N_a, N_z, N_x)), np.zeros((N_a, N_z, N_x))
    V_NP_t, V_NN_t = np.zeros((N_a, N_z, N_x)), np.zeros((N_a, N_z, N_x))
    # Prepare arrays to save optimal policy
    a_star_HR = np.zeros((N_a, N_z, N_x), dtype = np.int64)
    a_star_HD = np.zeros((N_a, N_z, N_x), dtype = np.int64)
    a_star_NP = np.zeros((N_a, N_z, N_x), dtype = np.int64)
    a_star_NN = np.zeros((N_a, N_z, N_x), dtype = np.int64)
    # initialize the while loop
    diff, iteration = tol + 1., 0
    VH_t, VN_t = np.copy(VH_init), np.copy(VN_init)
    rm_tm1 = np.copy(rm_init)
    while (iteration < max_iter) & (diff  > tol):
        # Load the values in the previous loop
        VH_tp1, VN_tp1 = np.copy(VH_t), np.copy(VN_t)
        rm_t = np.copy(rm_tm1)
        for a_idx, a in enumerate(a_grid):
            for z_idx, z in enumerate(z_grid):
                for x_idx, x in enumerate(x_grid):
                    # Homeowners
                    V_HR_t[a_idx, z_idx, x_idx], a_star_HR[a_idx, z_idx, x_idx] = \
                        V_HR(z = z, a = a, a_prime_vec = a_grid,
                             h = h_star, ph = ph, rm = rm_t[:, z_idx, x_idx],
                             r = r, beta = beta, alpha = alpha, sigma = sigma,
                             gamma = gamma, d = d, trans_prob_z_vec = trans_prob_z[z_idx, :],
                             trans_prob_x_vec = trans_prob_x[x_idx, :],
                             V_H_prime = VH_tp1)
                    if a_idx <= a_d_max_idx:
                        V_HD_t[a_idx, z_idx, x_idx] = \
                            V_HD(z = z, h = h_eps, c_d = c_d, c_hat_d = c_hat_d,
                                 delta = delta, beta = beta, alpha = alpha,
                                 sigma = sigma, gamma = gamma,
                                 trans_prob_z_vec = trans_prob_z[z_idx, :],
                                 trans_prob_x_vec = trans_prob_x[x_idx, :],
                                 V_N_prime = VN_tp1[a_idx, :, :])
                        a_star_HD[a_idx, z_idx, x_idx] = a_idx
                    else:
                        V_HD_t[a_idx, z_idx, x_idx] = \
                            V_HD(z = z, h = h_eps, c_d = c_d, c_hat_d = c_hat_d,
                                 delta = delta, beta = beta, alpha = alpha,
                                 sigma = sigma, gamma = gamma,
                                 trans_prob_z_vec = trans_prob_z[z_idx, :],
                                 trans_prob_x_vec = trans_prob_x[x_idx, :],
                                 V_N_prime = VN_tp1[a_d_max_idx, :, :])
                        a_star_HD[a_idx, z_idx, x_idx] = a_d_max_idx
                    # Non-homeowners
                    V_NP_t[a_idx, z_idx, x_idx], a_star_NP[a_idx, z_idx, x_idx] = \
                        V_NP(z = z, a = a, a_prime_vec = a_grid,
                             h = h_star, ph = ph, lmbd = lmbd, r = r, beta = beta,
                             alpha = alpha, sigma = sigma, gamma = gamma,
                             d = d, trans_prob_z_vec = trans_prob_z[z_idx, :],
                             trans_prob_x_vec = trans_prob_x[x_idx, :],
                             V_H_prime = VH_tp1)
                    V_NN_t[a_idx, z_idx, x_idx], a_star_NN[a_idx, z_idx, x_idx] = \
                        V_NN(z = z, a = a, a_prime_vec = a_grid,
                             h = h_eps, r = r, beta = beta, alpha = alpha,
                             sigma = sigma, gamma = gamma,
                             trans_prob_z_vec = trans_prob_z[z_idx, :],
                             trans_prob_x_vec = trans_prob_x[x_idx, :],
                             V_N_prime = VN_tp1)
        # Expected value for homeowners (considering the Gumbel disturbance)
        VH_t = V_H(V_HR_t, V_HD_t, kappaH)
        # Expected value for non-homeowners (considering the Gumbel disturbance)
        VN_t = V_N(V_NP_t, V_NN_t, kappaN)
        # Default probabilities for homeowners (considering the Gumbel disturbance)
        D_prob = default_prob(V_HR_t, V_HD_t, kappaH)
        # Purchase probabilities for non-homeowners (considering the Gumbel disturbance)
        P_prob = purchase_prob(V_NP_t, V_NN_t, kappaN)
        # Calculate the mortgage rate in the previous period
        for a_idx in range(N_a):
            for z_idx in range(N_z):
                for x_idx, x in enumerate(x_grid):
                    rm_tm1[a_idx, z_idx, x_idx] = \
                        mortgage_rate(a_prime_idx = a_idx,
                                      z_idx = z_idx,
                                      x_idx = x_idx,
                                      x = x,
                                      r = r,
                                      theta = theta,
                                      trans_prob_z = trans_prob_z,
                                      trans_prob_x = trans_prob_x,
                                      default_prob_array = D_prob)
        # Convergence check
        diff_VH = max(np.abs((VH_tp1 - VH_t).flatten()))
        diff_VN = max(np.abs((VN_tp1 - VN_t).flatten()))
        diff_rm = max(np.abs((rm_t - rm_tm1).flatten()))
        diff = max([diff_VH, diff_VN, diff_rm])
        # Progress the iteration counter
        iteration += 1
    # Check whether or not the converged result was obtained
    flag = (diff > tol)
    return VH_t, VN_t, rm_t, D_prob, P_prob, a_star_HR, a_star_HD, a_star_NP, a_star_NN, flag