import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

def plot_distribution(x,
                      y,
                      density,
                      xlabel = None,
                      ylabel = None,
                      zlabel = None,
                      title = None,
                      xlim = None,
                      ylim = None,
                      zlim = None,
                      savefig = True,
                      fname = 'density.png'):
    Y, X = np.meshgrid(y, x)
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection = "3d")
    ax.plot_surface(X, Y, density, cmap = "Blues")
    if type(xlabel) != None:
        ax.set_xlabel(xlabel)
    if type(ylabel) != None:
        ax.set_ylabel(ylabel)
    if type(zlabel) != None:
        ax.set_zlabel(zlabel)
    if type(title) != None:
        ax.set_title(title)
    if type(xlim) != None:
        ax.set_xlim(xlim)
    if type(ylim) != None:
        ax.set_ylim(ylim)
    if type(zlim) != None:
        ax.set_zlim(zlim)
    if savefig:
        plt.savefig(fname, dpi = 100, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    