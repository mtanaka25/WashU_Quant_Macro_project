from .KNTModel import *
from .tools import StopWatch, find_nearest_idx, multiple_line_plot, plot_distribution
from .simplest_spec import get_stationary_dist, get_stationary_dist_conditional_on_x
from .default_params import max_iter_def, tol_def
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class FixedHousePrice(KNTModel):
    def __init__(self, z_grid, trans_prob_z, x_grid, trans_prob_x,
                 beta, alpha, sigma, gamma, d, h_star, h_eps,
                 c_d, r, theta, kappaH, kappaN, a_ranage, N_a, ph
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
        self.ph = ph
    
    def get_stationary_dist_conditional_on_x(self,
                                             fixed_x = 1.,
                                             max_iter = max_iter_def,
                                             tol = tol_def):
        x1_idx = find_nearest_idx(fixed_x, self.x_grid)
        print(f'Starting to calculate the stationary distribution (under x = {fixed_x})...',
              flush = True)
        stopwatch = StopWatch()
        result = get_stationary_dist_conditional_on_x(
            trans_prob_z = self.trans_prob_z,
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
        self.conditional_density_H, self.conditional_density_N, _ = result
    
    def get_stationary_dist(self,
                            max_iter = max_iter_def,
                            tol = tol_def):
        x1_idx = find_nearest_idx(1., self.x_grid)
        print('Starting to calculate the stationary distribution...',
              flush = True)
        stopwatch = StopWatch()
        result = get_stationary_dist(trans_prob_z = self.trans_prob_z,
                                     trans_prob_x = self.trans_prob_x,
                                     default_prob = self.probD,
                                     purchase_prob = self.probP,
                                     a_star_H_idx = self.a_star_H_idx,
                                     a_star_NP_idx = self.a_star_NP_idx,
                                     a_star_NN_idx = self.a_star_NN_idx,
                                     a_grid = self.a_grid,
                                     max_iter = max_iter,
                                     tol = tol)
        stopwatch.stop()
        if result[-1]==True:
            print('Failed to obtain the stationary distribution. Try again with more max_iter.')
        # Unpack and store the result
        self.density_H, self.density_N, _ = result
    
    def plot_value_func(self,
                        homeownership = 'H',
                        axis = 0,
                        fixed_states = ((0, 0)),
                        savefig = True,
                        fname = 'fig_values.png'
                        ):
        if homeownership == 'H':
            V = self.ValueH
            ylabel = '$V^H$'
        else:
            V = self.ValueN
            ylabel = '$V^N$'
        # conut the number of series to be plotted
        if type(fixed_states[0]) == int:
            N_lines = 1
        else:
            N_lines = len(fixed_states)
        # take x-axis data
        if axis == 0:
            x = self.a_grid
            xlabel = '$a$'
        elif axis == 1:
            x = self.z_grid
            xlabel = '$z$'
        else:
            x = self.x_grid
            xlabel = '$x$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if axis == 0:
            for n in range(N_lines):
                data2plot[n, :] = V[:, fixed_states[n][0], fixed_states[n][1]]
                labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        elif axis == 1:
            for n in range(N_lines):
                data2plot[n, :] = V[fixed_states[n][0], :, fixed_states[n][1]]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        else:
            for n in range(N_lines):
                data2plot[n, :] = V[fixed_states[n][0], fixed_states[n][1], :]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ z_{'+ f'{fixed_states[n][1]}' +'}$')
        # Plot the data
        multiple_line_plot(x, data2plot,
                           x_label = xlabel, y_label = ylabel,
                           labels = labels,
                           savefig = savefig, fname = fname)
    
    def plot_saving_func(self,
                        type_of_household = 'H',
                        axis = 0,
                        fixed_states = ((0, 0)),
                        savefig = True,
                        fname = 'fig_saving.png'
                        ):
        if type_of_household == 'H':
            A = self.a_grid[self.a_star_H_idx]
            ylabel = "${a'}^{H}$"
        elif type_of_household == 'NP':
            A = self.a_grid[self.a_star_NP_idx]
            ylabel = "${a'}^{N,P}$"
        else:
            A = self.a_grid[self.a_star_NN_idx]
            ylabel = "${a'}^{N,N}$"
        # conut the number of series to be plotted
        if type(fixed_states[0]) == int:
            N_lines = 1
        else:
            N_lines = len(fixed_states)
        # take x-axis data
        if axis == 0:
            x = self.a_grid
            xlabel = '$a$'
        elif axis == 1:
            x = self.z_grid
            xlabel = '$z$'
        else:
            x = self.x_grid
            xlabel = '$x$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if axis == 0:
            for n in range(N_lines):
                data2plot[n, :] = A[:, fixed_states[n][0], fixed_states[n][1]]
                labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        elif axis == 1:
            for n in range(N_lines):
                data2plot[n, :] = A[fixed_states[n][0], :, fixed_states[n][1]]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        else:
            for n in range(N_lines):
                data2plot[n, :] = A[fixed_states[n][0], fixed_states[n][1], :]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ z_{'+ f'{fixed_states[n][1]}' +'}$')
        # Plot the data
        multiple_line_plot(x, data2plot,
                           x_label = xlabel, y_label = ylabel,
                           labels = labels,
                           plot_45_degree_line = True,
                           savefig = savefig, fname = fname)
    
    def plot_default_prob(self,
                          axis = 0,
                          fixed_states = ((0, 0)),
                          savefig = True,
                          fname = 'default_prob.png'
                          ):
        # conut the number of series to be plotted
        if type(fixed_states[0]) == int:
            N_lines = 1
        else:
            N_lines = len(fixed_states)
        # take x-axis data
        if axis == 0:
            x = self.a_grid
            xlabel = '$a$'
        elif axis == 1:
            x = self.z_grid
            xlabel = '$z$'
        else:
            x = self.x_grid
            xlabel = '$x$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if axis == 0:
            for n in range(N_lines):
                data2plot[n, :] = self.probD[:, fixed_states[n][0], fixed_states[n][1]]
                labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        elif axis == 1:
            for n in range(N_lines):
                data2plot[n, :] = self.probD[fixed_states[n][0], :, fixed_states[n][1]]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        else:
            for n in range(N_lines):
                data2plot[n, :] = self.probD[fixed_states[n][0], fixed_states[n][1], :]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ z_{'+ f'{fixed_states[n][1]}' +'}$')
        # Plot the data
        multiple_line_plot(x, data2plot*100,
                           x_label = xlabel, y_label = 'default prob(%)',
                           ylim = [-0.5, 100.5],
                           labels = labels,
                           savefig = savefig, fname = fname)
    
    def plot_purchase_prob(self,
                          axis = 0,
                          fixed_states = ((0, 0)),
                          savefig = True,
                          fname = 'purchase_prob.png'
                          ):
        # conut the number of series to be plotted
        if type(fixed_states[0]) == int:
            N_lines = 1
        else:
            N_lines = len(fixed_states)
        # take x-axis data
        if axis == 0:
            x = self.a_grid
            xlabel = '$a$'
        elif axis == 1:
            x = self.z_grid
            xlabel = '$z$'
        else:
            x = self.x_grid
            xlabel = '$x$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if axis == 0:
            for n in range(N_lines):
                data2plot[n, :] = self.probP[:, fixed_states[n][0], fixed_states[n][1]]
                labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        elif axis == 1:
            for n in range(N_lines):
                data2plot[n, :] = self.probP[fixed_states[n][0], :, fixed_states[n][1]]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        else:
            for n in range(N_lines):
                data2plot[n, :] = self.probP[fixed_states[n][0], fixed_states[n][1], :]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ z_{'+ f'{fixed_states[n][1]}' +'}$')
        # Plot the data
        multiple_line_plot(x, data2plot*100,
                           x_label = xlabel, y_label = 'prob of buying house(%)',
                           ylim = [-0.5, 100.5],
                           labels = labels,
                           savefig = savefig, fname = fname)
    
    def plot_mortgage_rate(self,
                          axis = 0,
                          fixed_states = ((0, 0)),
                          savefig = True,
                          fname = 'mortgage_rate.png'
                          ):
        # conut the number of series to be plotted
        if type(fixed_states[0]) == int:
            N_lines = 1
        else:
            N_lines = len(fixed_states)
        # take x-axis data
        if axis == 0:
            x = self.a_grid
            xlabel = '$a$'
        elif axis == 1:
            x = self.z_grid
            xlabel = '$z$'
        else:
            x = self.x_grid
            xlabel = '$x$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if axis == 0:
            for n in range(N_lines):
                data2plot[n, :] = self.rm[:, fixed_states[n][0], fixed_states[n][1]]
                labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        elif axis == 1:
            for n in range(N_lines):
                data2plot[n, :] = self.rm[fixed_states[n][0], :, fixed_states[n][1]]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ x_{'+ f'{fixed_states[n][1]}' +'}$')
        else:
            for n in range(N_lines):
                data2plot[n, :] = self.rm[fixed_states[n][0], fixed_states[n][1], :]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and$ z_{'+ f'{fixed_states[n][1]}' +'}$')
        # Plot the data
        multiple_line_plot(x, data2plot*100,
                           x_label = xlabel, y_label = 'mortgage rate (%)',
                           labels = labels,
                           savefig = savefig, fname = fname)

    def plot_homeownership(self,
                            savefig = True,
                            fname = 'houseownership.png'
                            ):
        data = [np.sum(self.density_H), np.sum(self.density_N)]
        label = ['Homeowners', 'Non-homeowners']
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.bar(label, data, width=0.618)
        if savefig:
            plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
    
    def plot_stationary_distribution(self,
                                     homeownership = 'H',
                                     fixed_axis = 2,
                                     fixed_state_id = 0,
                                     savefig = True,
                                     zlim = [0, 1.05],
                                     fname = 'stationary_dist.png'
                                     ):
        if homeownership == 'H':
            density = self.density_H
            title = 'Homeowners'
        else:
            density = self.density_N
            title = 'Non-Homeowners'
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