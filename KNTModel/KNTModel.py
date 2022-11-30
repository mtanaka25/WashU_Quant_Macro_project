import numpy as np
from .tools import find_nearest_idx

class KNTModel:
    def __init__(self, z_grid, trans_prob_z, x_grid, trans_prob_x,
                 beta, alpha, sigma, gamma, d, h_star, h_eps,
                 c_d, c_hat_d, a_d_max, delta, lmbd, r, theta,
                 kappaH, kappaN, a_ranage, N_a
                 ):
        # Prepare the grid for a
        a_grid = np.linspace(a_ranage[0], a_ranage[1], N_a)
        # Ensure a_grid has a = 0
        idx = find_nearest_idx(0, a_grid)
        a_grid[idx] = 0.
        # Store the parameters as the instance attributes
        self.z_grid, self.trans_prob_z = z_grid, trans_prob_z
        self.x_grid, self.trans_prob_x = x_grid, trans_prob_x
        self.N_z, self.N_x, self.N_a = len(z_grid), len(x_grid), N_a
        self.a_grid = a_grid
        self.beta, self.alpha, self.sigma = beta, alpha, sigma
        self.gamma, self.d, = gamma, d,
        self.h_star, self.h_eps, self.c_d = h_star, h_eps, c_d
        self.a_d_max, self.c_hat_d, self.delta, self.lmbd = a_d_max, c_hat_d, delta, lmbd
        self.r, self.theta, self.kappaH, self.kappaN = r, theta, kappaH, kappaN