from .FixedHousePrice import FixedHousePrice
from .default_params import *
from .simplest_spec.FRM import *
from .tools import StopWatch, plot_distribution

class FRM_FixedHousePrice(FixedHousePrice):
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
                 lmbd = lmbd_def,
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
                         lmbd = lmbd,
                         r = r,
                         theta = theta,
                         kappaH = kappaH,
                         kappaN = kappaN,
                         a_ranage = a_ranage,
                         N_a = N_a,
                         ph = ph)
        self.isFRM = True
    
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
                                 lmbd = self.lmbd,
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

    def get_dist_under_specific_x(self,
                                  fixed_x = 1.,
                                  max_iter = max_iter_def,
                                  tol = tol_def):
        x1_idx = find_nearest_idx(fixed_x, self.x_grid)
        print(f'Starting to calculate the distribution under x = {fixed_x}...',
              flush = True)
        stopwatch = StopWatch()
        result = get_distribution_under_specific_x(
            trans_prob_z = self.trans_prob_z,
            default_prob = self.probD[:, :, x1_idx, :],
            purchase_prob = self.probP[:, :, x1_idx],
            a_star_HR_idx = self.a_star_HR_idx[:, :, x1_idx, :],
            a_star_HD_idx = self.a_star_HD_idx[:, :, x1_idx, :],
            a_star_NP_idx = self.a_star_NP_idx[:, :, x1_idx],
            a_star_NN_idx = self.a_star_NN_idx[:, :, x1_idx],
            max_iter = max_iter,
            tol = tol)
        stopwatch.stop()
        if result[-1]==True:
            print('Failed to obtain the stationary distribution. Try again with more max_iter.')
        # Unpack and store the result
        self.conditional_density_H, self.conditional_density_N, _ = result
    
    def get_stationary_dist_by_iter(self,
                                    max_iter = max_iter_def,
                                    tol = tol_def):
        print('Starting to calculate the stationary distribution...',
              flush = True)
        stopwatch = StopWatch()
        result = get_stationary_dist_by_iter(
            trans_prob_z = self.trans_prob_z,
            trans_prob_x = self.trans_prob_x,
            default_prob = self.probD,
            purchase_prob = self.probP,
            a_star_HR_idx = self.a_star_HR_idx,
            a_star_HD_idx = self.a_star_HD_idx,
            a_star_NP_idx = self.a_star_NP_idx,
            a_star_NN_idx = self.a_star_NN_idx,
            max_iter = max_iter,
            tol = tol)
        stopwatch.stop()
        if result[-1]==True:
            print('Failed to obtain the stationary distribution. Try again with more max_iter.')
        # Unpack and store the result
        self.density_H, self.density_N, _ = result
    
    def get_stationary_dist_by_eig(self,
                                   tol = 1E-5):
        stopwatch = StopWatch()
        trans_mat = get_joint_transition_matrix(
            trans_prob_z = self.trans_prob_z,
            trans_prob_x = self.trans_prob_x,
            default_prob = self.probD,
            purchase_prob = self.probP,
            a_star_HR_idx = self.a_star_HR_idx,
            a_star_HD_idx = self.a_star_HD_idx,
            a_star_NP_idx = self.a_star_NP_idx,
            a_star_NN_idx = self.a_star_NN_idx,
            )
        dist, eigs, count = get_stationary_dist_by_eig(trans_mat, tol)
        if count == 1:
            popH, popN = convert_flatten_dist_to_array(
                dist, N_a = self.N_a, N_z = self.N_z, N_x = self.N_x)
            self.density_H, self.density_N = popH, popN
        else:
            for i in range(count):
                popH, popN = convert_flatten_dist_to_array(
                    dist[:, i], N_a = self.N_a, N_z = self.N_z, N_x = self.N_x)
                self.__dict__[f'density_H_{i+1}'] = popH
                self.__dict__[f'density_N_{i+1}'] = popN
        stopwatch.stop()
        self.joint_transition_mat = trans_mat
        self.eigenvalues = eigs
        
    def plot_stationary_distribution(self,
                                     homeownership = 'H',
                                     density_name = None,
                                     fixed_axis = 2,
                                     fixed_state_id = 0,
                                     savefig = True,
                                     zlim = [0, 1.05],
                                     fname = 'stationary_dist.png'
                                     ):
        if homeownership == 'H':
            density = self.density_H
            title = 'Homeowners'
            if np.isscalar(fixed_axis)|np.isscalar(fixed_state_id):
                print('For homeowners in FRM economy, fixed_axis and fixed_state_id should be 2-dimension lists/tuples.')
        else:
            density = self.density_N
            title = 'Non-Homeowners'
            if not (np.isscalar(fixed_axis) & np.isscalar(fixed_state_id)):
                print('For non-homeowners in FRM economy, fixed_axis and fixed_state_id should be scalars.')
        if type(density_name) != type(None):
            density = self.__dict__[density_name]
            title = density_name
        if homeownership == 'H':
            fixed_axis = tuple(fixed_axis)
            if fixed_axis == (0, 1):
                density2plot = density[fixed_state_id[0], fixed_state_id[1], :, :]
                x, y = self.a_grid, self.x_grid
                xlabel, ylabel = '$x$', '$\\bar{x}$'
                title += ' ($a = a_{' + f'{fixed_state_id[0]}' + '}$ and $z = z_{' + f'{fixed_axis[1]}'+ '}$)'
            elif fixed_axis == (0, 2):
                density2plot = density[fixed_state_id[0], :, fixed_state_id[1], :]
                x, y = self.z_grid, self.x_grid
                xlabel, ylabel = '$z$', '$\\bar{x}$'
                title += ' ($a = a_{' + f'{fixed_state_id[0]}' + '}$ and $x = x_{' + f'{fixed_axis[1]}'+ '}$)'
            elif fixed_axis == (0, 3) or (0, -1):
                density2plot = density[fixed_state_id[0], :, :, fixed_state_id[1]]
                x, y = self.z_grid, self.x_grid
                xlabel, ylabel = '$z$', '$x$'
                title += ' ($a = a_{' + f'{fixed_state_id[0]}' + '}$ and $\\bar{x} = \\bar{x}_{' + f'{fixed_axis[1]}'+ '}$)'
            elif fixed_axis == (1, 2):
                density2plot = density[:, fixed_state_id[0],fixed_state_id[1], :]
                x, y = self.a_grid, self.x_grid
                xlabel, ylabel = '$a$', '$\\bar{x}$'
                title += ' ($z = z_{' + f'{fixed_state_id[0]}' + '}$ and $x = x_{' + f'{fixed_axis[1]}'+ '}$)'
            elif fixed_axis == (1, 3) or (1, -1):
                density2plot = density[:, fixed_state_id[0], :, fixed_state_id[1]]
                x, y = self.a_grid, self.x_grid
                xlabel, ylabel = '$a$', '$x$'
                title += ' ($z = z_{' + f'{fixed_state_id[0]}' + '}$ and $\\bar{x} = \\bar{x}_{' + f'{fixed_axis[1]}'+ '}$)'
            else:
                density2plot = density[:, :, fixed_state_id[0], fixed_state_id[1]]
                x, y = self.a_grid, self.x_grid
                xlabel, ylabel = '$a$', '$z$'
                title += ' ($x = x_{' + f'{fixed_state_id[0]}' + '}$ and $\\bar{x} = \\bar{x}_{' + f'{fixed_axis[1]}'+ '}$)'
        else:
            if fixed_axis == 0:
                density2plot = density[fixed_state_id, :, :]
                x, y = self.z_grid, self.x_grid
                xlabel, ylabel = 'z', 'x'
                title += ' ($a = a_{' + f'{fixed_state_id}' + '}$)'
            elif fixed_axis == 1:
                density2plot = density[:, fixed_state_id, :]
                x, y = self.a_grid, self.x_grid
                xlabel, ylabel = 'a', 'x'
                title += ' ($z = z_{' + f'{fixed_state_id}' + '}$)'
            else:
                density2plot = density[:, :, fixed_state_id]
                x, y = self.a_grid, self.z_grid
                xlabel, ylabel = 'a', 'z'
                title += ' ($x = x_{' + f'{fixed_state_id}' + '}$)'
        plot_distribution(x = x,
                          y = y,
                          density = density2plot,
                          xlabel = xlabel,
                          ylabel = ylabel,
                          zlabel = '',
                          title = title,
                          zlim = zlim,
                          savefig = savefig,
                          fname = fname
                          )