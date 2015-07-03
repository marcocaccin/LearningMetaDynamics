#!/usr/bin/env python
from __future__ import print_function, division

import os
import scipy as sp
import scipy.linalg as LA
import scipy.spatial.distance as sp_dist
from ase.atoms import Atoms
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.io
import theano
import theano.tensor as th
from lammps2py_interface import calc_lammps
from sklearn.gaussian_process import GaussianProcess
from sklearn.kernel_ridge import KernelRidge
import pdb
# import matplotlib
# matplotlib.use('GTK3Cairo')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from ase.md.langevin import Langevin
# from ase.calculators.cp2k import CP2K
# from ase.calculators.lj import LennardJones
# from ase.calculators.lammpsrun import LAMMPS


##### Levi-Civita symbol used for Theano cross product #####
eijk = sp.zeros((3,3,3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
#############################################################


##### Definition of functional groups for CVs #####
dihedral_atoms_phi = [4,6,8,14] # C(O)-N-C(a)-C(O)
dihedral_atoms_psi = [6,8,14,16] # N-C(a)-C(O)-N

fun_group_phi = range(6) + [7]
fun_group_psi = range(15,22)
#############################################################


def get_reciprocal(positions, dihedral_atoms):
    """Calculate dihedral angle
    """
    ii, jj, kk, ll = dihedral_atoms
    # vector 0->1, 1->2, 2->3 and their normalized cross products:
    a = positions[jj] - positions[ii]
    b = positions[kk] - positions[jj]
    c = positions[ll] - positions[kk]
    bxa = sp.cross(b, a)
    if sp.vdot(bxa, c) > 0:
        return True
    else:
        return False


##### Utility functions to be added to ase.Atoms Class #####
def phi_(self, dihedral_list=dihedral_atoms_phi):
    return self.get_dihedral(dihedral_list)


def psi_(self, dihedral_list=dihedral_atoms_psi):
    return self.get_dihedral(dihedral_list)


def set_phi_(self, phi):
    self.set_dihedral(dihedral_atoms_phi, phi, indices=fun_group_phi)


def set_psi_(self, psi):
    self.set_dihedral(dihedral_atoms_psi, psi, indices=fun_group_psi)


def colvars(self):
    s = sp.atleast_2d(sp.array([self.phi(), self.psi()]))
    return s


def rescale_velocities_(self, T_target):
    T = self.get_kinetic_energy() / (1.5 * units.kB * len(self))
    self.set_momenta(self.get_momenta() * (T_target / T)**0.5)


def thcross(x, y):
    result = th.as_tensor(th.dot(th.dot(eijk, y), x))
    return result


def th_dihedral_angle(r0,r1,r2,r3):
    bxa = thcross(r2 - r1, r1 - r0)
    bxa /= bxa.norm(2)
    cxb = thcross(r3 - r2, r2 - r1)
    cxb /= cxb.norm(2)
    angle = th.arccos(th.dot(bxa, cxb))
    return angle
    
def th_dihedral_angle_rec(r0,r1,r2,r3):
    # reciprocal of the angle
    bxa = thcross(r2 - r1, r1 - r0)
    bxa /= bxa.norm(2)
    cxb = thcross(r3 - r2, r2 - r1)
    cxb /= cxb.norm(2)
    angle = 2*sp.pi - th.arccos(th.dot(bxa, cxb))
    return angle


thr0,thr1, thr2, thr3 = th.dvectors('thr0', 'thr1', 'thr2', 'thr3')
angle = th_dihedral_angle(thr0,thr1, thr2, thr3)
angle_rec = th_dihedral_angle_rec(thr0,thr1, thr2, thr3)

grad_angle = th.grad(angle, [thr0,thr1, thr2, thr3])
grad_angle_rec = th.grad(angle_rec, [thr0,thr1, thr2, thr3])

# fun = theano.function([thr0, thr1, thr2, thr3], angle)
d_angle = theano.function([thr0, thr1, thr2, thr3], grad_angle, allow_input_downcast=True)
d_angle_rec = theano.function([thr0, thr1, thr2, thr3], grad_angle_rec, allow_input_downcast=True)


def ddihedralangle_dr(positions, dihedron):
    """
    dF / dr = dF / dx . dx / dr
    This is the second bit. r are coordinates of the atoms, x is collective variable
    F is variable of interest (potential energy in our case)
    """
    ii, jj, kk, ll = dihedron
    r0, r1, r2, r3 = positions[dihedron]
    reciprocal_angle = get_reciprocal(positions, dihedron)
    gradients = sp.zeros(positions.shape)
    if get_reciprocal:
        nonzerograds = d_angle_rec(r0, r1, r2, r3)
    else:
        nonzerograds = d_angle(r0, r1, r2, r3)
    gradients[dihedron] = nonzerograds
    return gradients


def dEml_dr(positions):
    result = ddihedralangle_dr(positions, dihedral_atoms_phi) + \
             ddihedralangle_dr(positions, dihedral_atoms_psi)
    return result


def get_constraint_forces(atoms, ml_model):
    pos = atoms.get_positions()
    ds_dr = sp.array([ddihedralangle_dr(pos, dihedral_atoms_phi),
                      ddihedralangle_dr(pos, dihedral_atoms_psi)])
    dUml_ds = ml_model.predict_gradient(sp.atleast_2d(atoms.colvars()))
    # dUml_ds[0][0] = 0
    forces = sp.dot(ds_dr.T, dUml_ds.ravel()).T
    return forces


def round_vector(vec, precision = 0.05):
    return ((vec + 0.5 * precision) / precision).astype('int') * precision 


def verletstep(atoms, mlmodel, f, dt, mixing=[1.0, 0.0], lammpsdata=None, do_update=True):
    p = atoms.get_momenta()
    p += 0.5 * dt * f
    atoms.set_positions(atoms.get_positions() +
                        dt * p / atoms.get_masses()[:,None])
    atoms.set_momenta(p)
    # get actual forces and potential energy of configuration
    pot_energy, f0 = calc_lammps(atoms, preloaded_data=lammpsdata)
    # Accumulate the new observation in the dataset
    coarse_colvars = round_vector(atoms.colvars())
    distance_from_data = sp_dist.cdist(sp.atleast_2d(coarse_colvars), mlmodel.X_fit_).ravel()
    # check if configuration has already occurred
    if distance_from_data.min() == 0.0:
        index = list(distance_from_data).index(0.0)
        # set value to minimum energy in the bin
        if pot_energy < mlmodel.y[index]: 
            mlmodel.y[index] = pot_energy
            newdata = True
        else:
            newdata = False
    else:
        mlmodel.accumulate_data(coarse_colvars, sp.array([pot_energy]))
        newdata = True
    # update ML potential if required
    if newdata:
        mlmodel.update_fit()
    # Get ML constraint forces if the model is fitted
    if hasattr(mlmodel, 'dual_coef_') and do_update: # and pot_energy < 0:
        fextra = - get_constraint_forces(atoms, mlmodel)
    else:
        if hasattr(mlmodel, 'dual_coef_'):
            print("ML model not fitted yet")
        fextra = 0
    # X_near = Xplotgrid([atoms.phi() - 0.2, atoms.psi() - 0.2], [atoms.phi() - 0.2, atoms.psi() - 0.2], 2, 10)
    # y_near_mean = mlmodel.predict(X_near).mean()
    # if pot_energy < y_near_mean:
    #             mix = 1.2
    #         else:
    #             mix = 0.8

    # Compose the actual and the ML forces together by mixing them accordingly
    f = (mixing[0] * f0 - mixing[1] * fextra)
    
    atoms.set_momenta(atoms.get_momenta() + 0.5 * dt * f)
    return pot_energy, f
    
    
def draw_2Dreconstruction(ax, mlmodel, Xnew, X_grid):
    
    ax0, ax1 = ax
    y_grid = mlmodel.predict(X_grid)
    
    y_grid[y_grid > mlmodel.y.max() + 0.2] = mlmodel.y.max() + 0.2
    y_grid[y_grid < mlmodel.y.min() - 0.2] = mlmodel.y.min() - 0.2
    
    ax0.clear()
    ax0.scatter(mlmodel.X_fit_[:,0], mlmodel.X_fit_[:,1], marker = 'o', s = 50, c = mlmodel.y, 
                cmap = cm.RdBu, alpha = .5, edgecolors='none')    
    ax1.clear()
    ax1.scatter(X_grid[:,0], X_grid[:,1], s = 200, c = y_grid, 
                cmap = cm.RdBu, alpha = 1, edgecolors='none', marker='s')
    ax1.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)
    
    gradnew = mlmodel.predict_gradient(sp.atleast_2d(Xnew)).ravel()
    gradnew /= LA.norm(gradnew)
    ax1.arrow(Xnew[0], Xnew[1], gradnew[0], gradnew[1], head_width=0.05, head_length=0.1, fc='k', ec='k')

    ax0.set_title('Data Points')
    ax1.set_title('ML Prediction')
    for axx in [ax0, ax1]:
        axx.set_xlabel(r'$\phi$')
        axx.set_ylabel(r'$\psi$')
        axx.set_xlim(0, 2*sp.pi)
        axx.set_ylim(0, 2*sp.pi)


def draw_3Dreconstruction(fig, ax, mlmodel, X_grid):
    
    y_grid = mlmodel.predict(X_grid)
    y_grid[y_grid > mlmodel.y.max() + 0.2] = mlmodel.y.max() + 0.2
    y_grid[y_grid < mlmodel.y.min() - 0.2] = mlmodel.y.min() - 0.2
    ax.clear()
    
    X0, X1 = sp.unique(X_grid[:,0]), sp.unique(X_grid[:,1])
    X0, X1 = sp.meshgrid(X0, X1)
    y_grid = y_grid.reshape(X0.shape)
    surf = ax.plot_surface(X0, X1, y_grid, rstride=1, cstride=1, cmap=cm.RdBu,
                           linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi$')
    ax.set_title('ML Prediction')
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02e'))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0, 2*sp.pi)
    ax.set_ylim(0, 2*sp.pi)


def Xplotgrid(Xmin, Xmax, n_features, n_pts):
    X_grid = [sp.linspace(Xmin[i], Xmax[i], n_pts) for i in range(n_features)]
    X_grid = sp.meshgrid(*X_grid)
    return sp.vstack([x.ravel() for x in X_grid]).reshape(n_features,-1).T 


def printenergy(atoms, epot, step=-1):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    ekin = atoms.get_kinetic_energy() / len(atoms)
    print('Step %d | Energy per atom: Epot = %.3e eV  Ekin = %.3e eV (T = %3.0f K) Etot = %.7e eV' % (step, epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


def initialise_env():
    # add convenience functions to calculate and set angles to Atoms object
    setattr(Atoms, 'set_phi', set_phi_)
    setattr(Atoms, 'set_psi', set_psi_)
    setattr(Atoms, 'phi', phi_)
    setattr(Atoms, 'psi', psi_)
    setattr(Atoms, 'rescale_velocities', rescale_velocities_)
    setattr(Atoms, 'colvars', colvars)
    
    if "ASE_CP2K_COMMAND" not in os.environ:
        os.environ['ASE_CP2K_COMMAND'] = '/Users/marcocaccin/Code/cp2k/cp2k/exe/Darwin-IntelMacintosh-gfortran/cp2k_shell.ssmp'
    if "LAMMPS_COMMAND" not in os.environ:
        os.environ["LAMMPS_COMMAND"] = '/Users/marcocaccin/Code/lammps/src/lmp_mpi'
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'


def main():
    
    T = 300.0
    dt = 1 * units.fs
    nsteps = 10000
    mixing = [1.0, .9]
    lengthscale = 0.6
    gamma = 1 / (2 * lengthscale**2)
    #     mlmodel = GaussianProcess(corr='squared_exponential', 
    #         # theta0=1e-1, thetaL=1e-4, thetaU=1e+2,
    #         theta0=1., 
    #         random_start=100, normalize=False, nugget=1.0e-2)
    mlmodel = KernelRidge(kernel='rbf', 
                          gamma=gamma, gammaL = gamma/4, gammaU=2*gamma,
                           alpha=1.0e-2, variable_noise=False, max_lhood=False)
    # data = sp.loadtxt('phi_psi_minener_coarse_1M_md.csv')
    # mlmodel.fit(data[:,:2], data[:,2])
    plt.close('all')
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(24, 13))
    X_grid = Xplotgrid(sp.array([0., 0.]), 2 * sp.pi * sp.array([1., 1.]), 2, 70)
    
    
    atoms = ase.io.read('myplum.xyz')
    with open('data.input', 'r') as file:
        lammpsdata = file.readlines()

    # Set temperature
    MaxwellBoltzmannDistribution(atoms, 0.5 * units.kB * T, force_temp=True)
    # Set total momentum to zero
    p = atoms.get_momenta()
    p -= p.sum(axis=0) / len(atoms)
    atoms.set_momenta(p)
    atoms.rescale_velocities(T)

    print("START")
    
    pot_energy, f = calc_lammps(atoms, preloaded_data=lammpsdata)
    mlmodel.accumulate_data(round_vector(atoms.colvars()), sp.array([pot_energy]))
    printenergy(atoms, pot_energy)
    # draw_3Dreconstruction(fig3d, ax3d, mlmodel, X_grid)
    # fig3d.canvas.draw()
    try:
        os.remove('atoms.traj')
    except:
        pass
    traj = ase.io.Trajectory("atoms.traj", 'a')
    traj.write(atoms)

    results, traj_buffer = [], []
    # teaching_points = sp.unique(sp.exp(sp.linspace(0, sp.log(nsteps), nsteps/10)).astype('int') + 1)
    teaching_points = sp.unique((sp.linspace(0, nsteps**(1/3), nsteps/20)**3).astype('int') + 1)

    for istep in range(nsteps):
        
        print("Dihedral angles | phi = %.3f, psi = %.3f " % (atoms.phi(), atoms.psi()))
        do_update = (istep > 100) # (istep in teaching_points) or (istep - nsteps == 1) # istep % 20 == 0 # 
        pot_energy, f = verletstep(atoms, mlmodel, f, dt, 
                                   mixing=mixing, lammpsdata=lammpsdata, 
                                   do_update = do_update)
        atoms.rescale_velocities(T)
        printenergy(atoms, pot_energy/atoms.get_number_of_atoms(), step=istep)
        if do_update:
            try:
                print("Lengthscale = %.3e, Noise = %.3e" % (1/(2 * mlmodel.gamma)**0.5, mlmodel.noise.mean()))
            except:
                print("")    
        #         if hasattr(mlmodel, 'dual_coef_') and do_update:
        #             draw_2Dreconstruction(ax, mlmodel, atoms.colvars().ravel(), X_grid)
        #             fig.canvas.draw()
        #             if False: # istep > 1000:
        #                 draw_3Dreconstruction(fig3d, ax3d, mlmodel, X_grid)
        #                 fig3d.canvas.draw()
        if istep % 10 == 0 and hasattr(mlmodel, 'dual_coef_'):
            draw_2Dreconstruction(ax, mlmodel, atoms.colvars().ravel(), X_grid)
            fig.canvas.draw()
        traj_buffer.append(atoms.copy())
        if istep % 100 == 0:
            for at in traj_buffer:
                traj.write(at, properties=['forces'])
            traj_buffer = []
        results.append(sp.array([atoms.phi(), atoms.psi(), pot_energy]))
    traj.close()
    print("FINISHED")
    # print("theta = %.5e" % mlmodel.theta_)
    sp.savetxt('results.csv', sp.array(results))
    sp.savetxt('mlmodel.dual_coef_.csv', mlmodel.dual_coef_)
    sp.savetxt('mlmodel.X_fit_.csv', mlmodel.X_fit_)
    sp.savetxt('mlmodel.y.csv', mlmodel.y)
    calc = None
    traj = ase.io.Trajectory("atoms.traj", "r")
    ats = []
    for atoms in traj:
        ats.append(atoms)
    ase.io.write("atomstraj.xyz", ats, format="extxyz")
    
    return mlmodel

if __name__ == '__main__':
    initialise_env()
    ret = main()
