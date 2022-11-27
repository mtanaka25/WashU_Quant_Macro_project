from .KNTModel import *

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
    
    