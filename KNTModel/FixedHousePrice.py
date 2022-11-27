from .KNTModel import *
from .tools import StopWatch, find_nearest_idx
from .simplest_spec import get_stationary_dist
from .default_params import max_iter_def, tol_def
import numpy as np
class FixedHousePrice(KNTModel):
    def __init__(self, z_grid, trans_prob_z, x_grid, trans_prob_x,
                 beta, alpha, sigma, gamma, d, h_star, h_eps,
                 c_d, r, theta, kappa, a_ranage, N_a, ph
                 ):
        super().__init__(z_grid = z_grid,
                         trans_prob_z = trans_prob_z,
                         x_grid = x_grid,
                         trans_prob_x = trans_prob_x,
                         beta = beta,
                         alpha = alpha,
                         sigma = sigma,
                         gamma = gamma,
                         d = d,
                         h_star = h_star,
                         h_eps = h_eps,
                         c_d = c_d,
                         r = r,
                         theta = theta,
                         kappa = kappa,
                         a_ranage = a_ranage,
                         N_a = N_a)
        self.ph = ph
    
    def value_func_iter(self):
        pass
    
    def get_stationary_dist(self,
                            max_iter = max_iter_def,
                            tol = tol_def):
        x1_idx = find_nearest_idx(1., self.x_grid)
        print('Starting to calculate the stationary distribution (under x = 1)...',
              flush = True)
        stopwatch = StopWatch()
        result = get_stationary_dist(trans_prob_z = self.trans_prob_z,
                                     default_prob = self.probD[:, :, x1_idx],
                                     purchase_prob = self.probP[:, :, x1_idx],
                                     a_star_H_idx = self.a_star_H_idx[:, :, x1_idx],
                                     a_star_NP_idx = self.a_star_NP_idx[:, :, x1_idx],
                                     a_star_NN_idx = self.a_star_NN_idx[:, :, x1_idx],
                                     a_grid = self.a_grid,
                                     max_iter = max_iter,
                                     tol = tol)
        stopwatch.stop()
        if result[-1]==True:
            print('Failed to obtain the stationary distribution. Try again with more max_iter.')
        # Unpack and store the result
        self.density_H, self.density_N, _ = result
