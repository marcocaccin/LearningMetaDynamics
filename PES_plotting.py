from __future__ import print_function, division

import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import cm


class Plot_energy_n_point:
    def __init__(self, ax, mlmodel, current_point):
        self.ax = ax
        self.mlmodel = mlmodel
        x = y = sp.linspace(0,2*sp.pi, 80)
        self.X, self.Y = sp.meshgrid(x, y)
        self.gridpts = sp.vstack((self.X.ravel(), self.Y.ravel())).T
        zmin, zmax = mlmodel.y.min() - 0.2, mlmodel.y.max() + 0.2
        
        Z = self.mlmodel.predict(self.gridpts)
        Z[Z > zmax] = zmax
        Z[Z < zmin] = zmin
        Z = Z.reshape(self.X.shape)
        self.meshplot = ax.pcolormesh(self.X, self.Y, Z, cmap = cm.RdBu, alpha = 0.8,
                                      edgecolors='None', vmin = -1., vmax = 0.5)
        
        self.scatterplot = ax.scatter(current_point[0], current_point[1], marker='h', 
                                      s = 400, c = 'g', alpha = 0.5)
        
        self.ax.set_xlim(x.min(), x.max())
        self.ax.set_ylim(y.min(), y.max())
        self.ax.set_title('ML Prediction')
        self.ax.set_xlabel(r'$\phi$')
        self.ax.set_ylabel(r'$\psi$')
        
        
    def update_current_point(self, current_point):
        self.scatterplot.set_offsets(current_point.ravel())
        return self
        
    
    def update_prediction(self):
        zmin, zmax = self.mlmodel.y.min() - 0.2, self.mlmodel.y.max() + 0.2
        Z = self.mlmodel.predict(self.gridpts)
        Z[Z > zmax] = zmax
        Z[Z < zmin] = zmin
        Z = Z.reshape(self.X.shape)
        self.meshplot.set_array(Z[:-1,:-1].ravel())
        return self
        
        
class Plot_datapts:
    def __init__(self, ax, mlmodel):
        self.ax = ax
        self.mlmodel = mlmodel
        self.ylim = self.xlim = (0, 2*sp.pi)
        self.scatterplot = ax.scatter(self.mlmodel.X_fit_[:,0], self.mlmodel.X_fit_[:,1], 
                                      c = self.mlmodel.y, marker = 'o', s = 50, 
                                      cmap = cm.RdBu, alpha = 0.8, edgecolors='none',
                                      vmin = -1., vmax = 0.5)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_title('Data Points')
        self.ax.set_xlabel(r'$\phi$')
        self.ax.set_ylabel(r'$\psi$')
        
        
    def update(self):
        xy = self.mlmodel.X_fit_
        z = self.mlmodel.y.ravel()
        self.scatterplot.set_offsets(xy)
        self.scatterplot.set_array(z)
        return self


# def draw_2Dreconstruction(ax, mlmodel, Xnew, X_grid):
#     
#     ax0, ax1 = ax
#     y_grid = mlmodel.predict(X_grid)
#     
#     y_grid[y_grid > mlmodel.y.max() + 0.2] = mlmodel.y.max() + 0.2
#     y_grid[y_grid < mlmodel.y.min() - 0.2] = mlmodel.y.min() - 0.2
#     
#     ax0.clear()
#     ax0.scatter(mlmodel.X_fit_[:,0], mlmodel.X_fit_[:,1], marker = 'o', s = 50, c = mlmodel.y, 
#                 cmap = cm.RdBu, alpha = .5, edgecolors='none')    
#     ax1.clear()
#     ax1.scatter(X_grid[:,0], X_grid[:,1], s = 200, c = y_grid, 
#                 cmap = cm.RdBu, alpha = 1, edgecolors='none', marker='s')
#     ax1.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)
#     
#     gradnew = mlmodel.predict_gradient(sp.atleast_2d(Xnew)).ravel()
#     gradnew /= LA.norm(gradnew)
#     ax1.arrow(Xnew[0], Xnew[1], gradnew[0], gradnew[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
# 
#     ax0.set_title('Data Points')
#     ax1.set_title('ML Prediction')
#     for axx in [ax0, ax1]:
#         axx.set_xlabel(r'$\phi$')
#         axx.set_ylabel(r'$\psi$')
#         axx.set_xlim(0, 2*sp.pi)
#         axx.set_ylim(0, 2*sp.pi)
# 
# 
# def draw_3Dreconstruction(fig, ax, mlmodel, X_grid):
#     
#     y_grid = mlmodel.predict(X_grid)
#     y_grid[y_grid > mlmodel.y.max() + 0.2] = mlmodel.y.max() + 0.2
#     y_grid[y_grid < mlmodel.y.min() - 0.2] = mlmodel.y.min() - 0.2
#     ax.clear()
#     
#     X0, X1 = sp.unique(X_grid[:,0]), sp.unique(X_grid[:,1])
#     X0, X1 = sp.meshgrid(X0, X1)
#     y_grid = y_grid.reshape(X0.shape)
#     surf = ax.plot_surface(X0, X1, y_grid, rstride=1, cstride=1, cmap=cm.RdBu,
#                            linewidth=0, antialiased=False)
#     ax.set_xlabel(r'$\phi$')
#     ax.set_ylabel(r'$\psi$')
#     ax.set_title('ML Prediction')
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02e'))
#     # fig.colorbar(surf, shrink=0.5, aspect=5)
#     ax.set_xlim(0, 2*sp.pi)
#     ax.set_ylim(0, 2*sp.pi)
# 
# 
# def Xplotgrid(Xmin, Xmax, n_features, n_pts):
#     X_grid = [sp.linspace(Xmin[i], Xmax[i], n_pts) for i in range(n_features)]
#     X_grid = sp.meshgrid(*X_grid)
#     return sp.vstack([x.ravel() for x in X_grid]).reshape(n_features,-1).T 
