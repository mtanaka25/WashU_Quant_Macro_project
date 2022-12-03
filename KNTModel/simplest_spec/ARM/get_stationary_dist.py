import numpy as np
from numba import njit, prange
from numba import types, f8, i8, b1

@njit(types.Tuple((f8[:, :, :], f8[:, :, :], b1))
      (f8[:,:], f8[:,:,:], f8[:,:,:], i8[:,:,:],
       i8[:,:,:], i8[:,:,:], i8[:,:,:], i8, i8, f8))
def get_distribution_under_specific_x(trans_prob_z,
                                      default_prob,
                                      purchase_prob,
                                      a_star_HR_idx,
                                      a_star_HD_idx,
                                      a_star_NP_idx,
                                      a_star_NN_idx,
                                      fixed_x_idx,
                                      max_iter,
                                      tol):
    # The following is pure computational stuff (for efficient usage of memory)
    trans_prob_z =  np.ascontiguousarray(trans_prob_z)
    # Get the number of grid points
    N_a = default_prob.shape[0]
    N_z = trans_prob_z.shape[0]
    N_x = default_prob.shape[-1]
    # initialize the distribution (start with uniform)
    each_density = 1/ (2 * N_a * N_z)
    pop_H = np.zeros((N_a, N_z, N_x))
    pop_N = np.zeros((N_a, N_z, N_x))
    pop_H[:, :, fixed_x_idx] = each_density
    pop_N[:, :, fixed_x_idx] = each_density
    # initialize the while loop
    diff, iteration = tol +1., 0
    while (iteration < max_iter) & (diff > tol):
        pop_H_pre, pop_N_pre = np.copy(pop_H), np.copy(pop_N)
        # reset the updated density
        pop_H *= 0.
        pop_N *= 0.
        for zp_idx in prange(N_z):
            for a_idx in prange(N_a):
                for z_idx in prange(N_z):
                    H_pre = pop_H_pre[a_idx, z_idx, fixed_x_idx]
                    N_pre = pop_N_pre[a_idx, z_idx, fixed_x_idx]
                    Pz = trans_prob_z[z_idx, zp_idx]
                    Pd = Pd = default_prob[a_idx, z_idx, fixed_x_idx]
                    Pp = purchase_prob[a_idx, z_idx, fixed_x_idx]
                    pop_H[a_star_HR_idx[a_idx, z_idx, fixed_x_idx], zp_idx, fixed_x_idx] += (1-Pd) * Pz * H_pre
                    pop_N[a_star_HD_idx[a_idx, z_idx, fixed_x_idx], zp_idx, fixed_x_idx] += Pd * Pz * H_pre
                    pop_H[a_star_NP_idx[a_idx, z_idx, fixed_x_idx], zp_idx, fixed_x_idx] += Pp * Pz * N_pre
                    pop_N[a_star_NN_idx[a_idx, z_idx, fixed_x_idx], zp_idx, fixed_x_idx] += (1-Pp) * Pz * N_pre
        # Convergence check
        diff_pop_H = max(np.abs(pop_H - pop_H_pre).flatten())
        diff_pop_N = max(np.abs(pop_N - pop_N_pre).flatten())
        diff = max([diff_pop_H, diff_pop_N])
        # Progress the iteration counter
        iteration += 1
    flag = (diff > tol)
    return pop_H, pop_N, flag

@njit(f8[:, :]
      (f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:],
       i8[:,:,:], i8[:,:,:], i8[:,:,:], i8[:,:,:]))
def get_joint_transition_matrix(trans_prob_z,
                                trans_prob_x,
                                default_prob,
                                purchase_prob,
                                a_star_HR_idx,
                                a_star_HD_idx,
                                a_star_NP_idx,
                                a_star_NN_idx):
    # Get the number of grid points
    N_h = 2
    N_a = default_prob.shape[0]
    N_z = trans_prob_z.shape[0]
    N_x = trans_prob_x.shape[0]
    # Prepare some auxiliary values
    N_zx = N_z * N_x
    N_azx = N_a * N_z * N_x
    N = N_h * N_a * N_z * N_x
    # define a helper function
    def cumulative_idx(h_idx, a_idx, z_idx, x_idx):
        return N_azx * h_idx + N_zx * a_idx + N_x * z_idx + x_idx
    # Prepare arrays to save the transiton probabilities
    trans_mat = np.zeros((N, N))
    for z_prime_idx in range(N_z):
        for x_prime_idx in range(N_x):
            for a_idx in range(N_a):
                for z_idx in range(N_z):
                    for x_idx in range(N_x):
                        a_prime_idx_HH = a_star_HR_idx[a_idx, z_idx, x_idx]
                        a_prime_idx_HN = a_star_HD_idx[a_idx, z_idx, x_idx]
                        a_prime_idx_NH = a_star_NP_idx[a_idx, z_idx, x_idx]
                        a_prime_idx_NN = a_star_NN_idx[a_idx, z_idx, x_idx]
                        row_H = cumulative_idx(0, a_idx, z_idx, x_idx)
                        row_N = cumulative_idx(1, a_idx, z_idx, x_idx)
                        col_HH = cumulative_idx(0, a_prime_idx_HH, z_prime_idx, x_prime_idx)
                        col_HN = cumulative_idx(1, a_prime_idx_HN, z_prime_idx, x_prime_idx)
                        col_NH = cumulative_idx(0, a_prime_idx_NH, z_prime_idx, x_prime_idx)
                        col_NN = cumulative_idx(1, a_prime_idx_NN, z_prime_idx, x_prime_idx)
                        # Pick up probabilities
                        Pz = trans_prob_z[z_idx, z_prime_idx] # Prob of z to z_prime
                        Px = trans_prob_x[x_idx, x_prime_idx] # Prob of x to x_prime
                        Pd = default_prob[a_idx, z_idx, x_idx] # Prob of H to N
                        Pp = purchase_prob[a_idx, z_idx, x_idx] # Prob of N to H
                        trans_mat[row_H, col_HH] = (1 - Pd) * Pz * Px
                        trans_mat[row_H, col_HN] = Pd * Pz * Px
                        trans_mat[row_N, col_NH] = (1 - Pp) * Pz * Px
                        trans_mat[row_N, col_NN] = Pp * Pz * Px
    return trans_mat

#@njit(f8[:](f8[:, :], f8)) # numba does not work when eig() returns complex numbers.
def get_stationary_dist_by_eig(transition_matrix,
                               tol = 1E-5):
    eigval, eigvec = np.linalg.eig(transition_matrix)
    diff = np.abs(np.real(eigval) - 1.)
    count = np.sum((diff < tol))
    if count == 1:
        print('The system has a unique stationary distribution.')
    elif count == 0:
        print('The system is not stationary.')
    else:
        print('The system has multiple stationary distribuitions.')
        print(f'The transiton matrix has {count} eigenvalues of unity.')
    if count == 0:
        density, eigvals = np.zeros((eigvec.shape[0], 1))
    elif count == 1:
        eigvals = np.real(eigval[diff < tol])
        density = np.real(eigvec[:, (diff < tol)])
        density = density / sum(density)
        density.reshape[-1, 1]
    else:
        eigvals = np.real(eigval[diff < tol])
        density = np.real(eigvec[:, (diff < tol)])
        for i in range(count):
            density[:, i] = density[:, i] / sum(density[:, i])
    return density, eigvals, count

@njit(types.Tuple((f8[:, :, :], f8[:, :, :]))(f8[:], i8, i8, i8))
def convert_flatten_dist_to_array(flatten_distribution, N_a, N_z, N_x):
    # Prepare some auxiliary values
    N_zx = N_z * N_x
    N_azx = N_a * N_z * N_x
    # define a helper function
    def cumulative_idx(h_idx, a_idx, z_idx, x_idx):
        return N_azx * h_idx + N_zx * a_idx + N_x * z_idx + x_idx
    pop_H = np.zeros((N_a, N_z, N_x))
    pop_N = np.zeros((N_a, N_z, N_x))
    for a_idx in range(N_a):
        for z_idx in range(N_z):
                for x_idx in range(N_x):
                    idx_H = cumulative_idx(0, a_idx, z_idx, x_idx)
                    pop_H[a_idx, z_idx, x_idx] = flatten_distribution[idx_H]
                    idx_N = cumulative_idx(1, a_idx, z_idx, x_idx)
                    pop_N[a_idx, z_idx, x_idx] = flatten_distribution[idx_N]
    return pop_H, pop_N
    
@njit(types.Tuple((f8[:, :, :], f8[:, :, :], b1))
      (f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:],
       i8[:,:,:], i8[:,:,:], i8[:,:,:], i8[:,:,:],
       i8, f8))
def get_stationary_dist_by_iter(trans_prob_z,
                                trans_prob_x,
                                default_prob,
                                purchase_prob,
                                a_star_HR_idx,
                                a_star_HD_idx,
                                a_star_NP_idx,
                                a_star_NN_idx,
                                max_iter,
                                tol):
    # The following two lines are pure computational stuff (for efficient usage of memory)
    trans_prob_z =  np.ascontiguousarray(trans_prob_z)
    trans_prob_x =  np.ascontiguousarray(trans_prob_x)
    # Get the number of grid points
    N_a = default_prob.shape[0]
    N_z = trans_prob_z.shape[0]
    N_x = trans_prob_x.shape[0]
    # initialize the distribution (start with uniform)
    each_density = 1/ (2 * N_a * N_z * N_x) # 2 = homeowner or not
    pop_H = np.ones((N_a, N_z, N_x)) * each_density
    pop_N = np.ones((N_a, N_z, N_x)) * each_density
    # initialize the while loop
    diff, iteration = tol +1., 0
    while (iteration < max_iter) & (diff > tol):
        pop_H_pre, pop_N_pre = np.copy(pop_H), np.copy(pop_N)
        # reset the updated density
        pop_H *= 0.
        pop_N *= 0.
        # Update the distribution
        for zp_idx in prange(N_z):
            for xp_idx in prange(N_x):
                for a_idx in range(N_a):
                    for z_idx in range(N_z):
                        for x_idx in range(N_x):
                            H_pre = pop_H_pre[a_idx, z_idx, x_idx]
                            N_pre = pop_N_pre[a_idx, z_idx, x_idx]
                            Pd = default_prob[a_idx, z_idx, x_idx]
                            Pp = purchase_prob[a_idx, z_idx, x_idx]
                            Pz = trans_prob_z[z_idx, zp_idx]
                            Px = trans_prob_x[x_idx, xp_idx]
                            pop_H[a_star_HR_idx[a_idx, z_idx, x_idx], zp_idx, xp_idx] += (1-Pd) * Pz * Px * H_pre
                            pop_N[a_star_HD_idx[a_idx, z_idx, x_idx], zp_idx, xp_idx] += Pd * Pz * Px * H_pre
                            pop_H[a_star_NP_idx[a_idx, z_idx, x_idx], zp_idx, xp_idx] += Pp * Pz * Px * N_pre
                            pop_N[a_star_NN_idx[a_idx, z_idx, x_idx], zp_idx, xp_idx] += (1-Pp) * Pz * Px * N_pre
        # Convergence check
        diff_pop_H = max(np.abs(pop_H - pop_H_pre).flatten())
        diff_pop_N = max(np.abs(pop_N - pop_N_pre).flatten())
        diff = max([diff_pop_H, diff_pop_N])
        # Progress the iteration counter
        iteration += 1
    flag = (diff > tol)
    return pop_H, pop_N, flag