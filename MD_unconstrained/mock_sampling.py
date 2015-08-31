#!/usr/bin/env python
from __future__ import print_function, division

import os
import scipy as sp
import scipy.linalg as LA
import scipy.spatial.distance as sp_dist
from ase import units
from sklearn.kernel_ridge import KernelRidge
# import pdb
from matplotlib import pyplot as plt
import PES_plotting as pl
from time import time as get_time


SAMPLE_WEIGHT = 1
def round_vector(vec, precision = 0.05):
    """
    vec: array_like, type real
    
    precision: real, > 0
    """
    return ((vec + 0.5 * precision) / precision).astype('int') * precision 


def update_model(colvars, mlmodel, grid_spacing, temperature, do_update=False):
    # get actual forces and potential energy of configuration
    ### ML IS HERE ###
    if not (mlmodel is None and not do_update):
        # Accumulate the new observation in the dataset
        coarse_colvars = round_vector(colvars, precision=grid_spacing)
        distance_from_data = sp_dist.cdist(
            sp.atleast_2d(coarse_colvars), mlmodel.X_fit_).ravel()
        # check if configuration has already occurred
        if distance_from_data.min() == 0.0:
            index = list(distance_from_data).index(0.0)
            mlmodel.y[index] = - units.kB * temperature * sp.log(sp.exp(-mlmodel.y[index] / (units.kB * temperature)) + SAMPLE_WEIGHT)
        else:
            mlmodel.accumulate_data(coarse_colvars, - units.kB * temperature * sp.log(SAMPLE_WEIGHT))
            cv = coarse_colvars.ravel()
            xx = sp.linspace(cv[0] - 2*grid_spacing, cv[0] + 2*grid_spacing, 5)
            yy = sp.linspace(cv[1] - 2*grid_spacing, cv[1] + 2*grid_spacing, 5)
            XX, YY = sp.meshgrid(xx, yy)
            near_bins = sp.vstack((XX.ravel(), YY.ravel())).T
            distance_from_data = sp_dist.cdist(sp.atleast_2d(near_bins), mlmodel.X_fit_)
            for distance, near_bin in zip(distance_from_data, near_bins):
                if distance.min() > 0.:
                    mlmodel.accumulate_data(near_bin, 0.)
        if do_update:
            # update ML potential with all the data contained in it.
            mlmodel.update_fit()
    return


def main():
    
    T = 300.0 # Simulation temperature
    dt = 1 * units.fs # MD timestep
    nsteps = 1000000 # MD number of steps
    lengthscale = 0.5 # KRR Gaussian width.
    gamma = 1 / (2 * lengthscale**2)
    grid_spacing = 0.1
    mlmodel = KernelRidge(kernel='rbf', 
                          gamma=gamma, gammaL = gamma/4, gammaU=2*gamma,
                           alpha=1.0e-2, variable_noise=False, max_lhood=False)
    anglerange = sp.arange(0, 2*sp.pi + grid_spacing, grid_spacing)
    X_grid = sp.array([[sp.array([x,y]) for x in anglerange]
                       for y in anglerange]).reshape((len(anglerange)**2, 2))
                           
    # Bootstrap from initial database? uncomment
    data_MD = sp.loadtxt('phi_psi_pot_md300.csv')
    colvars = data_MD[0,:2]
    PotEng = data_MD[0,2]
    KinEng = data_MD[0,3]
    
    # Prepare diagnostic visual effects.
    plt.close('all')
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(24, 13))
    
    # Zero-timestep evaluation and data files setup.
    print("START")
    mlmodel.accumulate_data(round_vector(data_MD[0,:2], precision=grid_spacing), 0.)
    print('Step %d | Energy per atom: Epot = %.3e eV  Ekin = %.3e eV (T = %3.0f K) Etot = %.7e eV' % (
        0, PotEng/22, KinEng/22, KinEng / (22 * 1.5 * units.kB), PotEng + KinEng))

    # MD Loop
    for istep, line in enumerate(data_MD[:nsteps]):
        colvars = line[:2]
        PotEng = line[2]
        KinEng = line[3]
        # Flush Cholesky decomposition of K
        if istep % 1000 == 0:
            mlmodel.Cho_L = None
            mlmodel.max_lhood = False
        print("Dihedral angles | phi = %.3f, psi = %.3f " % (colvars[0], colvars[1]))
        do_update = (istep % 1000 == 59)
        t = get_time()
        update_model(colvars, mlmodel, grid_spacing, T, do_update=do_update)        
        if do_update and mlmodel.max_lhood:
            mlmodel.max_lhood = False
        print("TIMER 002 | %.3f" % (get_time() - t))

        print('Step %d | Energy per atom: Epot = %.3e eV  Ekin = %.3e eV (T = %3.0f K) Etot = %.7e eV' % (
            istep, PotEng/22, KinEng/22, KinEng / (22 * 1.5 * units.kB), PotEng + KinEng))

        if istep % 1000 == 59:
            t = get_time()
            if 'datasetplot' not in locals():
                datasetplot = pl.Plot_datapts(ax[0], mlmodel)
            else:
                datasetplot.update()
            if hasattr(mlmodel, 'dual_coef_'):
                if 'my2dplot' not in locals():
                    my2dplot = pl.Plot_energy_n_point(ax[1], mlmodel, colvars.ravel())
                else:
                    my2dplot.update_prediction()
                    my2dplot.update_current_point(colvars.ravel())
            print("TIMER 003 | %.03f" % (get_time() - t))
            t = get_time()
            fig.canvas.draw()
            print("TIMER 004 | %.03f" % (get_time() - t))    
    return mlmodel

if __name__ == '__main__':
    ret = main()
