#!/usr/bin/env python
from __future__ import print_function, division

import os
from matplotlib import pyplot as plt
import scipy as sp
from ase.atoms import Atoms
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.io
import theano
import theano.tensor as th
from lammps2py_interface import calc_lammps
from sklearn.gaussian_process import GaussianProcess
# from sklearn_additions import *
# import gpfit
# from ase.md.langevin import Langevin
# from ase.md.verlet import VelocityVerlet
# from ase.calculators.cp2k import CP2K
# from ase.calculators.lj import LennardJones
# from ase.calculators.lammpsrun import LAMMPS


##### Levi-Civita symbol used for Theano cross product #####
eijk = sp.zeros((3,3,3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
#############################################################


##### Definition of functional groups for CVs #####
dihedral_atoms_theta = [10,8,6,4]
dihedral_atoms_psi = [10,8,14,16]
fun_group_theta = range(6) + [7]
fun_group_psi = range(15,22)
#############################################################


##### Utility functions to be added to ase.Atoms Class #####
def psi_(self, dihedral_list=dihedral_atoms_psi):
    return self.get_dihedral(dihedral_list)


def theta_(self, dihedral_list=dihedral_atoms_theta):
    return self.get_dihedral(dihedral_list)


def set_theta_(self, theta):
    self.set_dihedral(dihedral_atoms_theta, theta, indices=fun_group_theta)


def set_psi_(self, psi):
    self.set_dihedral(dihedral_atoms_psi, psi, indices=fun_group_psi)


def colvars(self):
    s = sp.atleast_2d(sp.array([self.psi(), self.theta()]))
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


thr0,thr1, thr2, thr3 = th.dvectors('thr0', 'thr1', 'thr2', 'thr3')
angle = th_dihedral_angle(thr0,thr1, thr2, thr3)
gradangle = th.grad(angle, [thr0,thr1, thr2, thr3])
# fun = theano.function([thr0, thr1, thr2, thr3], angle)
dfun = theano.function([thr0, thr1, thr2, thr3], gradangle, allow_input_downcast=True)

    
def ddihedralangle_dr(positions, dihedron):
    """
    dF / dr = dF / dx . dx / dr
    This is the second bit. r are coordinates of the atoms, x is collective variable
    F is variable of interest (potential energy in our case)
    """
    
    gradients = sp.zeros(sp.shape(positions))
    ii, jj, kk, ll = dihedron
    nonzerograds = dfun(*positions[dihedron])
    gradients[dihedron] = nonzerograds
    return gradients


def dEml_dr(positions):
    result = ddihedralangle_dr(positions, dihedral_atoms_psi) + \
        0 * ddihedralangle_dr(positions, dihedral_atoms_theta)
    return result


def get_constraint_forces(atoms, ml_model):
    pos = atoms.get_positions()
    ds_dr = sp.array([ddihedralangle_dr(pos, dihedral_atoms_psi), 
                      ddihedralangle_dr(pos, dihedral_atoms_theta)])
    
    dUml_ds = ml_model.predict_gradient(sp.atleast_2d(atoms.colvars()))
    
    forces = sp.dot(ds_dr.T, dUml_ds.ravel()).T # sp.dot(dUml_ds, ds_dr)
    return forces


def verletstep(atoms, gp, f, dt, mixing=[1.0, 0.0], lammpsdata=None, do_update=True):
    p = atoms.get_momenta()
    p += 0.5 * dt * f
    atoms.set_positions(atoms.get_positions() +
                        dt * p / atoms.get_masses()[:,None])
    # print(atoms.get_positions())
    atoms.set_momenta(p)
    # get forces
    pot_energy, f0 = calc_lammps(atoms, preloaded_data=lammpsdata)
    
    if len(gp.X) > 100 and do_update:
        # update ML potential and calculate force
        gp.update_fit(atoms.colvars(), sp.array([pot_energy]))
        fextra = - get_constraint_forces(atoms, gp)
        fextra[dihedral_atoms_theta] = 0 
        # sum and rescale the two forces
        
        f = (mixing[0] * f0 - \
        mixing[1] * fextra)# * \
        # sp.sqrt((f0**2).sum(axis=1)).mean() / \
        # sp.sqrt((fextra**2).sum(axis=1)).mean())
    else:
        # keep on accumulating, it's not time yet
        gp.accumulate_data(atoms.colvars(), sp.array([pot_energy]))
        f = f0
    #     for b in range(22):
    #         atoms1 = atoms.copy()
    #         pos = atoms1.get_positions()
    #         pos[b,2] += 0.001
    #         atoms1.set_positions(pos) 
    #         pot1, f1 = calc_lammps(atoms1)
    #         dL = 0.5*(f0[b,2]+f1[b,2])*0.001
    #         dPot = pot1 - pot_energy
    #         dfrac = (dPot + dL) / dPot
    #         print("DEBUG | atom %03d, (dEpot+dL)/dEpot = %.5e" % ((b+1), dfrac))
    atoms.set_momenta(atoms.get_momenta() + 0.5 * dt * f)
    return pot_energy, f
    
    
def draw_2Dreconstruction(ax, gp, Xnew, X_grid):
    
    ax0, ax1 = ax
    y_grid, MSE_grid = gp.predict(X_grid, eval_MSE=True)
    
    y_grid[y_grid > gp.y.max() + 0.2] = gp.y.max() + 0.2
    y_grid[y_grid < gp.y.min() - 0.2] = gp.y.min() - 0.2
    
    ax0.clear()
    ax0.scatter(X_grid[:,0], X_grid[:,1], marker = 'h', s = 200, c = MSE_grid, 
                cmap = 'YlGnBu', alpha = 1, edgecolors='none')

    ax1.clear()
    ax1.scatter(X_grid[:,0], X_grid[:,1], s = 200, c = y_grid, 
                cmap = 'Spectral', alpha = 1, edgecolors='none')
    ax1.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)

    ax0.set_title('MSE')
    ax1.set_title('GP Prediction')
    
    for axx in [ax0, ax1]:
        axx.set_xlabel(r'$\psi$')
        axx.set_ylabel(r'$\theta$')
        axx.set_xlim(X_grid[:,0].min(), X_grid[:,0].max())
        axx.set_ylim(X_grid[:,1].min(), X_grid[:,1].max())   


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
    setattr(Atoms, 'set_theta', set_theta_)
    setattr(Atoms, 'set_psi', set_psi_)
    setattr(Atoms, 'theta', theta_)
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
    nsteps = 3000
    mixing = [1.0, .9]
    
    gp = GaussianProcess(corr='squared_exponential', 
        # theta0=1e-1, thetaL=1e-4, thetaU=1e+2,
        theta0=1., 
        random_start=100, normalize=False, nugget=1.0e-2)
    plt.close('all')
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    X_grid = Xplotgrid(sp.array([0., 0.]), 2 * sp.pi * sp.array([1., 1.]), 2, 50)
    
    
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
    gp.accumulate_data(atoms.colvars(), sp.array([pot_energy]))
    printenergy(atoms, pot_energy)

    try:
        os.remove('atoms.traj')
    except:
        pass
    traj = ase.io.Trajectory("atoms.traj", 'a')
    traj.write(atoms)

    results = []
    for istep in range(nsteps):
        print("Dihedral angles | theta = %.3f, psi = %.3f" % (atoms.theta(), atoms.psi()))
        
        pot_energy, f = verletstep(atoms, gp, f, dt, 
                                   mixing=mixing, lammpsdata=lammpsdata, 
                                   do_update = (istep % 10 == 0))
        atoms.rescale_velocities(T)
        # pos = atoms.get_positions() 
        # posdiff = sp.sum((pos - oldpos)**2, axis=1)**0.5
        # print("dR | max = %f, mean = %f" % (posdiff.max(), posdiff.mean()))
        # traj.append(atoms.copy())
        printenergy(atoms, pot_energy, step=istep)
        
        if hasattr(gp, 'D') and (istep % 10 == 0):
            draw_2Dreconstruction(ax, gp, atoms.colvars().ravel(), X_grid)
            fig.canvas.draw()
        if istep % 1 == 0:    
            traj.write(atoms, properties=['forces'])
        results.append(sp.array([atoms.theta(), atoms.psi(), pot_energy]))
    traj.close()
    print("FINISHED")
    print("theta = %.5e" % gp.theta_)
    sp.savetxt('results.csv', sp.array(results))
    calc = None
    traj = ase.io.Trajectory("atoms.traj", "r")
    ats = []
    for atoms in traj:
        ats.append(atoms)
    ase.io.write("atomstraj.xyz", ats, format="extxyz")

if __name__ == '__main__':
    initialise_env()
    main()
