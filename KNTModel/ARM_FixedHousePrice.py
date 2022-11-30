from .default_params import *
from .FixedHousePrice import FixedHousePrice
from .simplest_spec.value_func_iter_ARM import value_func_iter
from .tools import StopWatch

class ARM_FixedHousePrice(FixedHousePrice):
    def __init__(self,
                 z_grid,
                 trans_prob_z,
                 x_grid,
                 trans_prob_x,
                 beta = beta_def,
                 alpha = alpha_def,
                 sigma = sigma_def,
                 gamma = gamma_def,
                 d = d_def,
                 h_star = h_star_def,
                 h_eps = h_eps_def,
                 c_d = c_d_def,
                 c_hat_d = c_hat_d_def,
                 delta = delta_def,
                 a_d_max = a_d_max_def,
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
                         beta = beta,
                         alpha = alpha,
                         sigma = sigma,
                         gamma = gamma,
                         d = d,
                         h_star = h_star,
                         h_eps = h_eps,
                         c_d = c_d,
                         c_hat_d = c_hat_d,
                         delta = delta,
                         a_d_max = a_d_max,
                         r = r,
                         theta = theta,
                         kappaH = kappaH,
                         kappaN = kappaN,
                         a_ranage = a_ranage,
                         N_a = N_a,
                         ph = ph)
    
    def value_func_iter(self,
                        max_iter = max_iter_def,
                        tol = tol_def):
        print('Starting the value function iteration...', flush=True)
        stopwatch = StopWatch()
        result = value_func_iter(z_grid = self.z_grid,
                                 trans_prob_z = self.trans_prob_z,
                                 x_grid = self.x_grid,
                                 trans_prob_x = self.trans_prob_x,
                                 a_grid = self.a_grid,
                                 beta = self.beta,
                                 alpha = self.alpha,
                                 sigma = self.sigma,
                                 gamma = self.gamma,
                                 d = self.d,
                                 ph = self.ph,
                                 h_star = self.h_star,
                                 h_eps = self.h_eps,
                                 c_d = self.c_d,
                                 c_hat_d = self.c_hat_d,
                                 delta = self.delta,
                                 a_d_max = self.a_d_max,
                                 r = self.r,
                                 theta = self.theta,
                                 kappaH = self.kappaH,
                                 kappaN = self.kappaN,
                                 max_iter = max_iter,
                                 tol = tol)
        stopwatch.stop()
        if result[-1]==True:
            print('Value iteration failed to converge. Try again with more max_iter.')
        # Unpack and store the result
        self.ValueH, self.ValueN, self.rm, self.probD, self.probP,\
        self.a_star_HR_idx, self.a_star_HD_idx, self.a_star_NP_idx, self.a_star_NN_idx, _ = result
 