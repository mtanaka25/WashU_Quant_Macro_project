from .KNTModel import *
from .default_params import *

class VariableHousePrice(KNTModel):
    def __init__(self, z_grid, trans_prob_z, x_grid, trans_prob_x,
                 ph_grid, trans_prob_ph,
                 beta, alpha, sigma, gamma, d, h_star, h_eps,
                 c_d, r, theta, kappaH, kappaN, a_ranage, N_a,
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
                         kappaH = kappaH,
                         kappaN = kappaN,
                         a_ranage = a_ranage,
                         N_a = N_a)
        self.ph_grid, self.trans_prob_ph = ph_grid, trans_prob_ph
        
class ARM_VariableHousePrice(VariableHousePrice):
    def __init__(self,
                 z_grid,
                 trans_prob_z,
                 x_grid,
                 trans_prob_x,
                 ph_grid,
                 trans_prob_ph,
                 beta = beta_def,
                 alpha = alpha_def,
                 sigma = sigma_def,
                 gamma = gamma_def,
                 d = d_def,
                 h_star = h_star_def,
                 h_eps = h_eps_def,
                 c_d = c_d_def,
                 r = r_def,
                 theta = theta_def,
                 kappaH = kappaH_def,
                 kappaN = kappaN_def,
                 a_ranage = a_range_def,
                 N_a = N_a_def,
                 ph = ph_def
                 ):
        super().__init__(z_grid = z_grid,
                         trans_prob_z = trans_prob_z,
                         x_grid = x_grid,
                         trans_prob_x = trans_prob_x,
                         ph_grid = ph_grid,
                         trans_prob_ph = trans_prob_ph,
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
                         kappaH = kappaH,
                         kappaN = kappaN,
                         a_ranage = a_ranage,
                         N_a = N_a,
                         ph = ph)
        print('ARM_VariableHousePrice class is not implemented yet.')

class FRM_VariableHousePrice(VariableHousePrice):
    def __init__(self,
                 z_grid,
                 trans_prob_z,
                 x_grid,
                 trans_prob_x,
                 ph_grid,
                 trans_prob_ph,
                 beta = beta_def,
                 alpha = alpha_def,
                 sigma = sigma_def,
                 gamma = gamma_def,
                 d = d_def,
                 h_star = h_star_def,
                 h_eps = h_eps_def,
                 c_d = c_d_def,
                 r = r_def,
                 theta = theta_def,
                 kappaH = kappaH_def,
                 kappaN = kappaN_def,
                 a_ranage = a_range_def,
                 N_a = N_a_def,
                 ph = ph_def
                 ):
        super().__init__(z_grid = z_grid,
                         trans_prob_z = trans_prob_z,
                         x_grid = x_grid,
                         trans_prob_x = trans_prob_x,
                         ph_grid = ph_grid,
                         trans_prob_ph = trans_prob_ph,
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
                         kappaH = kappaH,
                         kappaN = kappaN,
                         a_ranage = a_ranage,
                         N_a = N_a,
                         ph = ph)
        print('FRM_VariableHousePrice class is not implemented yet.')
