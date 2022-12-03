from KNTModel import *
from KNTModel.tools import AR1_process
from KNTModel.model_comparison import compare_irfs
import numpy as np
import matplotlib.pyplot as plt

N_z     = 10    # number of grid points
rho_z   = 0.9   # AR1 coefficient
sig_z   = 0.1   # size of exogenous shock
Omega_z = 4     # range of grid

# N_x     = 5   # number of grid points
# rho_x   = 0.8 # AR1 coefficient
# sig_x   = 0.5 # size of exogenous shock
# Omega_x = 3   # range of grid
N_x     = 5   # number of grid points
rho_x   = 0.9 # AR1 coefficient
sig_x   = 0.1 # size of exogenous shock
Omega_x = 3   # range of grid


x_path = np.array([2, 2] + [2] * 98)

# Generate the instance for the income process
lnz_process = AR1_process(rho = rho_z,
                          sig = sig_z,
                          varname = 'lnz')
# Discretize by Tauchen's method
lnz_process.discretize(method = 'Tauchen',
                       N = N_z,
                       Omega = Omega_z,
                       is_write_out_result = False)
# Get the grid and the transition probabilities
z_grid = np.exp(lnz_process.lnz_grid)
trans_prob_z = lnz_process.trans_mat

print(z_grid)



# Generate the instance for the aggregate factor process
lnx_process = AR1_process(rho = rho_x,
                          sig = sig_x,
                          varname = 'lnx')
# Discretize by Tauchen's method
lnx_process.discretize(method = 'Tauchen',
                       N = N_x,
                       Omega = Omega_x,
                       is_write_out_result = False)
# Get the grid and the transition probabilities
x_grid = np.exp(lnx_process.lnx_grid)
trans_prob_x = lnx_process.trans_mat

print(x_grid)


# ## Step 2. Solve the adjustable-rate model

# Generate the instance for the adjustable-rate economy
ARM_Economy = ARM_FixedHousePrice(z_grid = z_grid,
                                  x_grid = x_grid,
                                  trans_prob_z = trans_prob_z,
                                  trans_prob_x = trans_prob_x,)
# Run the value iteration (it may take a few mins depending on # of grid points)
ARM_Economy.value_func_iter()
# Plot value function with respect to a
ARM_Economy.plot_value_func(homeownership = 'H',
                            axis = 0,
                            fixed_states =[(2, 2), (5, 2), (8, 2)],
                            fname ='ARM_value_fun_H_wrt_a.png')
ARM_Economy.plot_value_func(homeownership = 'N',
                            axis = 0,
                            fixed_states =[(2, 2), (5, 2), (8, 2)],
                            fname ='ARM_value_func_N_wrt_a.png')
# Plot saving function with respect to a
ARM_Economy.plot_saving_func(type_of_household = 'HR',
                             axis = 0,
                             fixed_states =[(2, 2), (5, 2), (8, 2)],
                             fname ='ARM_saving_func_HR_wrt_a.png')
ARM_Economy.plot_saving_func(type_of_household = 'NP',
                             axis = 0,
                             fixed_states =[(2, 2), (5, 2), (8, 2)],
                             fname ='ARM_saving_func_NP_wrt_a.png')
ARM_Economy.plot_saving_func(type_of_household = 'NN',
                             axis = 0,
                             fixed_states =[(2, 2), (5, 2), (8, 2)],
                             fname ='ARM_saving_func_NN_wrt_a.png')
# Plot default probability with respect to a
ARM_Economy.plot_default_prob(axis = 0,
                              fixed_states =[(2, 4), (5, 4), (8, 4)],
                              fname ='ARM_prob_D_wrt_a.png')
# Plot probability of buying house with respect to a
ARM_Economy.plot_purchase_prob(axis = 0,
                              fixed_states =[(2, 2), (5, 2), (8, 2)],
                              fname ='ARM_prob_P_wrt_a.png')
# Plot mortgage rate with respect to a
ARM_Economy.plot_mortgage_rate(axis = 0,
                               fixed_states =[(2, 2), (5, 2), (8, 2)],
                               fname ='ARM_mortgage_rate.png')
# Plot mortgage rate with respect to x
ARM_Economy.plot_mortgage_rate(axis = 2,
                               fixed_states =[(0, 5), (12, 5), (25, 5), (37, 5), (50, 5)],
                               fname ='ARM_mortgage_rate.png')
# Solve for the stationary distribution
ARM_Economy.get_stationary_dist_by_iter()
ARM_Economy.plot_homeownership(fname = 'ARM_homeownership.png')
ARM_Economy.plot_stationary_distribution(homeownership = 'H',
                                         fixed_axis = 2,
                                         fixed_state_id = 2,
                                         zlim = None,
                                         fname = 'ARM_stationary_dist_H.png')
ARM_Economy.plot_stationary_distribution(homeownership = 'N',
                                         fixed_axis = 2,
                                         fixed_state_id = 2,
                                         zlim = None,
                                         fname = 'ARM_stationary_dist_N.png')

ARM_Economy.x_shock_simulation(pre_shock_x_idx = 2, x_idx_path = x_path)
plt.close('all')

# ## Step 3. Solve the (partially) fixed-rate model

# Generate the instance for the (partially) fixed-rate economy
FRM_Economy = FRM_FixedHousePrice(z_grid = z_grid,
                                  x_grid = x_grid,
                                  trans_prob_z = trans_prob_z,
                                  trans_prob_x = trans_prob_x,)
# Run the value iteration (it may take a few mins depending on # of grid points)
FRM_Economy.value_func_iter()
# Plot value function with respect to a
FRM_Economy.plot_value_func(homeownership = 'H',
                            axis = 0,
                            fixed_states =[(2, 2, 2), (5, 2, 2), (8, 2, 2)],
                            fname ='FRM_value_func_homeowner_wrt_a.png')
FRM_Economy.plot_value_func(homeownership = 'N',
                            axis = 0,
                            fixed_states =[(2, 2), (5, 2), (8, 2)],
                            fname ='FRM_value_func_nonhomeowner_wrt_a.png')
# Plot saving function with respect to a
FRM_Economy.plot_saving_func(type_of_household = 'HR',
                             axis = 0,
                             fixed_states =[(2, 2, 2), (5, 2, 2), (8, 2, 2)],
                             fname ='FRM_saving_func_HR_wrt_a.png')
FRM_Economy.plot_saving_func(type_of_household = 'NP',
                             axis = 0,
                             fixed_states =[(2, 2), (5, 2), (8, 2)],
                             fname ='FRM_saving_func_NP_wrt_a.png')
FRM_Economy.plot_saving_func(type_of_household = 'NN',
                             axis = 0,
                             fixed_states =[(2, 2), (5, 2), (8, 2)],
                             fname ='FRM_saving_func_NP_wrt_a.png')
# Plot default probability with respect to a
FRM_Economy.plot_default_prob(axis = 0,
                              fixed_states =[(2, 4, 4), (5, 4, 4), (8, 4, 4)],
                              fname ='FRM_prob_D_wrt_a.png')
FRM_Economy.plot_default_prob(axis = 4,
                              fixed_states =[(2, 4, 2), (15, 4, 2), (28, 4, 2)],
                              fname ='FRM_prob_D_wrt_own_z.png')
# Plot probability of buying house with respect to a
FRM_Economy.plot_purchase_prob(axis = 0,
                               fixed_states =[(2, 2), (5, 2), (8, 2)],
                               fname ='FRM_prob_P_wrt_a.png')
# Solve for the stationary distribution
FRM_Economy.get_stationary_dist_by_iter()
FRM_Economy.plot_homeownership(fname = 'FRM_homeownership.png')
FRM_Economy.plot_stationary_distribution(homeownership = 'H',
                                         fixed_axis = (2, 3),
                                         fixed_state_id = (2, 2),
                                         zlim = None,
                                         fname = 'FRM_stationary_dist_H.png')
FRM_Economy.plot_stationary_distribution(homeownership = 'N',
                                         fixed_axis = 2,
                                         fixed_state_id = 2,
                                         zlim = None,
                                         fname = 'FRM_stationary_dist_N.png')
FRM_Economy.x_shock_simulation(pre_shock_x_idx = 2, x_idx_path = x_path)
plt.close('all')

# ## Step 4. Model comparison
compare_irfs(ARM_Economy, FRM_Economy, T_max = 20)
t = 1
sum([np.sum(ARM_Economy.irf_dist_H[:, :, 4, t]), np.sum(ARM_Economy.irf_dist_N[:, :, 4, t])])

