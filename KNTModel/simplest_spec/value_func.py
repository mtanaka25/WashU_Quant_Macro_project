import numpy as np
from numba import njit, types
from numba import f8, i8
from ..utility_fuc import *

@njit(types.Tuple((f8, i8))(f8, f8, f8[:], f8, f8, f8[:], f8, f8, f8, f8, f8, f8,
                          f8[:], f8[:], f8[:,:,:]))
def V_HR(z, a, a_prime_vec, h, ph, rm, r, beta, alpha, sigma, gamma, d,
         trans_prob_z_vec, trans_prob_x_vec, V_H_prime):
    # transition probability matirix
    # (from today' z and x to each pair of (z', x'))
    joint_trans_prob = np.zeros((len(trans_prob_z_vec),
                                 len(trans_prob_x_vec)))
    for z_idx, prob_z in enumerate(trans_prob_z_vec):
        for x_idx, prob_x in enumerate(trans_prob_x_vec):
            joint_trans_prob[z_idx, x_idx] = prob_z * prob_x
    # prepare an array to temporarily store the possible values
    possible_values = np.zeros((len(a_prime_vec), ))
    # flag indicating consumption cannot be positive
    flag = False
    # calculate value for each possible a_prime
    for a_prime_idx, a_prime in enumerate(a_prime_vec):
        # mortgage payment
        mortgage = rm[a_prime_idx] * (1 - d) * ph * h
        # calculate the consumption as the residual of budget constraint
        c = cons(z = z,
                 a = a,
                 r = r,
                 a_prime = a_prime,
                 down_pay = 0.,
                 mortgage_pay = mortgage)
        if c <= 0:
            # if c is negative, give penalty
            flow_u = utility(c = 1E-5,
                             h = 1E-5,
                             alpha = alpha,
                             sigma = sigma,
                             gamma = gamma)
            if a_prime_idx == 0:
                # if c is negative with smallest a', stop calculation
                flag = True
                break
        else:
            # flow utility
            flow_u = utility(c = c,
                             h = h,
                             alpha = alpha,
                             sigma = sigma,
                             gamma = gamma)
        # expected value
        expected_u = joint_trans_prob * V_H_prime[a_prime_idx, :, :]
        expected_u = np.sum(expected_u)
        # store the value of choosing the specific a_prime
        possible_values[a_prime_idx] = flow_u + beta * expected_u
    if flag:
        # if c is always negative, give penalty value. In that case, argmax is zero.
        return flow_u, 0
    else:
        # return max and argmax
        return max(possible_values), np.argmax(possible_values)

@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8[:], f8[:], f8[:,:]))
def V_HD(z, h, c_d, c_hat_d, delta, beta, alpha, sigma, gamma,
         trans_prob_z_vec, trans_prob_x_vec, V_N_prime):
    # transition probability matirix
    # (from today' z and x to each pair of (z', x'))
    joint_trans_prob = np.zeros((len(trans_prob_z_vec),
                                 len(trans_prob_x_vec)))
    for z_idx, prob_z in enumerate(trans_prob_z_vec):
        for x_idx, prob_x in enumerate(trans_prob_x_vec):
            joint_trans_prob[z_idx, x_idx] = prob_z * prob_x
    # consumption is income net of default cost
    c = min([delta * (z - c_d), c_hat_d])
    # flow utility
    flow_u = utility(c = c,
                     h = h,
                     alpha = alpha,
                     sigma = sigma,
                     gamma = gamma)
    # expected value
    expected_u = joint_trans_prob * V_N_prime[:, :]
    expected_u = np.sum(expected_u)
    # store the value of choosing the specific a_prime
    # return the value of default
    return flow_u + beta * expected_u

@njit(types.Tuple((f8, i8))(f8, f8, f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8,
                          f8[:], f8[:], f8[:,:,:]))
def V_NP(z, a, a_prime_vec, h, ph, lmbd, r, beta, alpha, sigma, gamma, d,
         trans_prob_z_vec, trans_prob_x_vec, V_H_prime):
    # transition probability matirix
    # (from today' z and x to each pair of (z', x'))
    joint_trans_prob = np.zeros((len(trans_prob_z_vec),
                                 len(trans_prob_x_vec)))
    for z_idx, prob_z in enumerate(trans_prob_z_vec):
        for x_idx, prob_x in enumerate(trans_prob_x_vec):
            joint_trans_prob[z_idx, x_idx] = prob_z * prob_x
    # down payment
    down_pay = d * ph * h
    # prepare an array to temporarily store the possible values
    possible_values = np.zeros((len(a_prime_vec), ))
    # flag indicating consumption cannot be positive
    flag = False
    # calculate value for each possible a_prime
    for a_prime_idx, a_prime in enumerate(a_prime_vec):
        # calculate the consumption as the residual of budget constraint
        c = cons(z = z,
                 a = a,
                 r = r,
                 a_prime = a_prime,
                 down_pay = down_pay,
                 mortgage_pay = 0.) - lmbd
        if c <= 0:
            # if c is negative, give penalty
            flow_u = utility(c = 1E-5,
                             h = 1E-5,
                             alpha = alpha,
                             sigma = sigma,
                             gamma = gamma)
            if a_prime_idx == 0:
                # if c is negative with smallest a', stop calculation
                flag = True
                break
        else:
            # flow utility
            flow_u = utility(c = c,
                             h = h,
                             alpha = alpha,
                             sigma = sigma,
                             gamma = gamma)
        # expected value
        expected_u = joint_trans_prob * V_H_prime[a_prime_idx, :, :]
        expected_u = np.sum(expected_u)
        # store the value of choosing the specific a_prime
        possible_values[a_prime_idx] = flow_u + beta * expected_u
    if flag:
        # if c is always negative, give penalty value. In that case, argmax is zero.
        return flow_u, 0
    else:
        # return max and argmax
        return max(possible_values), np.argmax(possible_values)

@njit(types.Tuple((f8, i8))(f8, f8, f8[:], f8, f8, f8, f8, f8, f8,
                          f8[:], f8[:], f8[:,:,:]))
def V_NN(z, a, a_prime_vec, h, r, beta, alpha, sigma, gamma,
         trans_prob_z_vec, trans_prob_x_vec, V_N_prime):
    # transition probability matirix
    # (from today' z and x to each pair of (z', x'))
    joint_trans_prob = np.zeros((len(trans_prob_z_vec),
                                 len(trans_prob_x_vec)))
    for z_idx, prob_z in enumerate(trans_prob_z_vec):
        for x_idx, prob_x in enumerate(trans_prob_x_vec):
            joint_trans_prob[z_idx, x_idx] = prob_z * prob_x
    # prepare an array to temporarily store the possible values
    possible_values = np.zeros((len(a_prime_vec), ))
    # flag indicating consumption cannot be positive
    flag = False
    # calculate value for each possible a_prime
    for a_prime_idx, a_prime in enumerate(a_prime_vec):
        # calculate the consumption as the residual of budget constraint
        c = cons(z = z,
                 a = a,
                 r = r,
                 a_prime = a_prime,
                 down_pay = 0.,
                 mortgage_pay = 0.)
        if c <= 0:
            # if c is negative, give penalty
            flow_u = utility(c = 1E-5,
                             h = 1E-5,
                             alpha = alpha,
                             sigma = sigma,
                             gamma = gamma)
            if a_prime_idx == 0:
                # if c is negative with smallest a', stop calculation
                flag = True
                break
        else:
            # flow utility
            flow_u = utility(c = c,
                             h = h,
                             alpha = alpha,
                             sigma = sigma,
                             gamma = gamma)
        # expected value
        expected_u = joint_trans_prob * V_N_prime[a_prime_idx, :, :]
        expected_u = np.sum(expected_u)
        # store the value of choosing the specific a_prime
        possible_values[a_prime_idx] = flow_u + beta * expected_u
    if flag:
        # if c is always negative, give penalty value. In that case, argmax is zero.
        return flow_u, 0
    else:
        # return max and argmax
        return max(possible_values), np.argmax(possible_values)

@njit([f8[:,:,:](f8[:,:,:], f8[:,:,:], f8),
       f8[:,:,:,:](f8[:,:,:,:], f8[:,:,:,:], f8)])
def V_H(V_HR, V_HD, kappa):
    return V_HD + kappa * np.log(np.exp((V_HR - V_HD)/kappa) + 1)

@njit(f8[:,:,:](f8[:,:,:], f8[:,:,:], f8))
def V_N(V_NP, V_NN, kappa):
    return V_NN + kappa * np.log(np.exp((V_NP - V_NN)/kappa) + 1)

@njit([f8[:,:,:](f8[:,:,:], f8[:,:,:], f8),
       f8[:,:,:,:](f8[:,:,:,:], f8[:,:,:,:], f8)])
def default_prob(V_HR, V_HD, kappa):
    return 1 / (1 + np.exp((V_HR - V_HD)/kappa))

@njit(f8[:,:,:](f8[:,:,:], f8[:,:,:], f8))
def purchase_prob(V_NP, V_NN, kappa):
    return np.exp((V_NP - V_NN)/kappa) / (1 + np.exp((V_NP - V_NN)/kappa))

@njit(f8(i8, i8, i8, f8, f8, f8, f8[:,:], f8[:,:], f8[:,:,:]))
def mortgage_rate(a_prime_idx, z_idx, x_idx, x, r, theta,
                  trans_prob_z, trans_prob_x, default_prob_array):
    # calculate the default probablity
    default_prob = 0.
    for z_prime_idx in range(trans_prob_z.shape[1]):
        for x_prime_idx in range(trans_prob_x.shape[1]):
            default_prob += (
                trans_prob_z[z_idx, z_prime_idx]
                * trans_prob_x[x_idx, x_prime_idx]
                * default_prob_array[a_prime_idx, z_prime_idx, x_prime_idx]
            )
    if default_prob == 1:
        # If default probability is 1, rm is not defined. So use a value very close to 1.
        default_prob *= 0.999
    return ((1 + r * x) - theta * default_prob)/(1 - default_prob) - 1