from .KNTModel import KNTModel
from .tools import multiple_line_plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class FixedHousePrice(KNTModel):
    def __init__(self, z_grid, trans_prob_z, x_grid, trans_prob_x,
                 beta, alpha, sigma, gamma, d, h_star, h_eps,
                 c_d, c_hat_d, delta, a_d_max, lmbd, r, theta, kappaH,
                 kappaN, a_ranage, N_a, ph):
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
                         N_a = N_a)
        self.ph = ph

    def plot_value_func(self,
                        homeownership = 'H',
                        axis = 0,
                        fixed_states = ((0, 0, 0)),
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
        elif axis == 2:
            x = self.x_grid
            xlabel = '$x$'
        else:
            x = self.x_grid
            xlabel = '$\\bar{x}$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if self.isFRM & (homeownership == 'H'):
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = V[:, fixed_states[n][0], fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = V[fixed_states[n][0], :, fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 2:
                for n in range(N_lines):
                    data2plot[n, :] = V[fixed_states[n][0], fixed_states[n][1] :, fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = V[fixed_states[n][0], fixed_states[n][1], fixed_states[n][2], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $z_{'+ f'{fixed_states[n][1]}' +'}$, $x_{'+ f'{fixed_states[n][2]}' +'}$')
        else:
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = V[:, fixed_states[n][0], fixed_states[n][1]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = V[fixed_states[n][0], :, fixed_states[n][1]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = V[fixed_states[n][0], fixed_states[n][1], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $z_{'+ f'{fixed_states[n][1]}' +'}$')
        # Plot the data
        multiple_line_plot(x, data2plot,
                           x_label = xlabel, y_label = ylabel,
                           labels = labels,
                           savefig = savefig, fname = fname)
    
    def plot_saving_func(self,
                        type_of_household = 'HR',
                        axis = 0,
                        fixed_states = ((0, 0, 0)),
                        savefig = True,
                        fname = 'fig_saving.png'
                        ):
        if type_of_household == 'HR':
            A = self.a_grid[self.a_star_HR_idx]
            ylabel = "${a'}^{H,R}$"
        elif type_of_household == 'HD':
            A = self.a_grid[self.a_star_HD_idx]
            ylabel = "${a'}^{H,D}$"
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
        elif axis == 2:
            x = self.x_grid
            xlabel = '$x$'
        else:
            x = self.x_grid
            xlabel = '$\\bar{x}$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if self.isFRM & ((type_of_household == 'HR') | (type_of_household == 'HD')):
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = A[:, fixed_states[n][0], fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = A[fixed_states[n][0], :, fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 2:
                for n in range(N_lines):
                    data2plot[n, :] = A[fixed_states[n][0], fixed_states[n][1] :, fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = A[fixed_states[n][0], fixed_states[n][1], fixed_states[n][2], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $z_{'+ f'{fixed_states[n][1]}' +'}$, $x_{'+ f'{fixed_states[n][2]}' +'}$')
        else:
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = A[:, fixed_states[n][0], fixed_states[n][1]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = A[fixed_states[n][0], :, fixed_states[n][1]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = A[fixed_states[n][0], fixed_states[n][1], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $z_{'+ f'{fixed_states[n][1]}' +'}$')
        # Plot the data
        multiple_line_plot(x, data2plot,
                           x_label = xlabel, y_label = ylabel,
                           labels = labels,
                           plot_45_degree_line = True,
                           savefig = savefig, fname = fname)
    
    def plot_default_prob(self,
                          axis = 0,
                          fixed_states = ((0, 0, 0)),
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
        elif axis == 2:
            x = self.x_grid
            xlabel = '$x$'
        else:
            x = self.x_grid
            xlabel = '$\\bar{x}$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if self.isFRM:
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = self.probD[:, fixed_states[n][0], fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = self.probD[fixed_states[n][0], :, fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 2:
                for n in range(N_lines):
                    data2plot[n, :] = self.probD[fixed_states[n][0], fixed_states[n][1] :, fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = self.probD[fixed_states[n][0], fixed_states[n][1], fixed_states[n][2], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $z_{'+ f'{fixed_states[n][1]}' +'}$, $x_{'+ f'{fixed_states[n][2]}' +'}$')
        else:
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = self.probD[:, fixed_states[n][0], fixed_states[n][1]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = self.probD[fixed_states[n][0], :, fixed_states[n][1]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = self.probD[fixed_states[n][0], fixed_states[n][1], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $z_{'+ f'{fixed_states[n][1]}' +'}$')
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
                labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
        elif axis == 1:
            for n in range(N_lines):
                data2plot[n, :] = self.probP[fixed_states[n][0], :, fixed_states[n][1]]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
        else:
            for n in range(N_lines):
                data2plot[n, :] = self.probP[fixed_states[n][0], fixed_states[n][1], :]
                labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $z_{'+ f'{fixed_states[n][1]}' +'}$')
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
        elif axis == 2:
            x = self.x_grid
            xlabel = '$x$'
        else:
            x = self.x_grid
            xlabel = '$\\bar{x}$'
        # prepare the array to save data to be plotted
        data2plot = np.zeros([N_lines, len(x)])
        labels = []
        if self.isFRM:
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = self.rm[:, fixed_states[n][0], fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = self.rm[fixed_states[n][0], :, fixed_states[n][1], fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            elif axis == 2:
                for n in range(N_lines):
                    data2plot[n, :] = self.rm[fixed_states[n][0], fixed_states[n][1] :, fixed_states[n][2]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $x_{'+ f'{fixed_states[n][1]}' +'}$, $\\bar{x}_{'+ f'{fixed_states[n][2]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = self.rm[fixed_states[n][0], fixed_states[n][1], fixed_states[n][2], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$, $z_{'+ f'{fixed_states[n][1]}' +'}$, $x_{'+ f'{fixed_states[n][2]}' +'}$')
        else:
            if axis == 0:
                for n in range(N_lines):
                    data2plot[n, :] = self.rm[:, fixed_states[n][0], fixed_states[n][1]]
                    labels.append('$z_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            elif axis == 1:
                for n in range(N_lines):
                    data2plot[n, :] = self.rm[fixed_states[n][0], :, fixed_states[n][1]]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $x_{'+ f'{fixed_states[n][1]}' +'}$')
            else:
                for n in range(N_lines):
                    data2plot[n, :] = self.rm[fixed_states[n][0], fixed_states[n][1], :]
                    labels.append('$a_{' + f'{fixed_states[n][0]}' +'}$ and $z_{'+ f'{fixed_states[n][1]}' +'}$')
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
        print(f'Share of homeowners: {data[0]*100:.3f} %')
        label = ['Homeowners', 'Non-homeowners']
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.bar(label, data, width=0.618)
        if savefig:
            plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
    
