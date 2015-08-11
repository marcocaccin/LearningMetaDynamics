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
from sklearn.kernel_ridge import KernelRidge
import pdb
from matplotlib import pyplot as plt
# from ase.md.langevin import Langevin
# from ase.calculators.cp2k import CP2K
# from ase.calculators.lj import LennardJones
# from ase.calculators.lammpsrun import LAMMPS
import PES_plotting as pl
from ignorance_field import IgnoranceField
# from MDPropagators import MLLangevin


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
    """
    In dihedral angle calculation, see if angle is the reciprocal or not.
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


##########################################################
#### Utility functions to be added to ase.Atoms Class ####
##########################################################

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


def get_temperature(self):
    """Return instantaneous canonical temperature of the system."""
    return self.get_kinetic_energy() / (1.5 * units.kB * len(self))


def rescale_velocities_(self, T_target):
    """Rescale momenta of the system to match canonical temperature T_target."""
    T = self.get_kinetic_energy() / (1.5 * units.kB * len(self))
    self.set_momenta(self.get_momenta() * (T_target / T)**0.5)

####################################
#### Theano functions from here ##########
# -- They need to be in the main code ####
# so that they are compiled only once -- #
##########################################

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
    Of the chain rule differentiation dF / dr = dF / dx . dx / dr , 
    this is the second bit. r are coordinates of the atoms, x is collective variable
    F is variable of interest (Potential energy, free energy...)
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


def get_constraint_forces(atoms, ml_model):
    """
    Atomic forces of a trained ML model given the configuration in atoms.
    """
    pos = atoms.get_positions()
    ds_dr = sp.array([ddihedralangle_dr(pos, dihedral_atoms_phi),
                      ddihedralangle_dr(pos, dihedral_atoms_psi)])
    dUml_ds = ml_model.predict_gradient(sp.atleast_2d(atoms.colvars()))
    forces = - sp.dot(ds_dr.T, dUml_ds.ravel()).T
    return forces


def get_extfield_forces(atoms, field):
    pos = atoms.get_positions()
    ds_dr = sp.array([ddihedralangle_dr(pos, dihedral_atoms_phi),
                      ddihedralangle_dr(pos, dihedral_atoms_psi)])
    dE_ds = field.get_forces(atoms.colvars().flatten())
    forces = sp.dot(ds_dr.T, dE_ds.ravel()).T
    return forces

    
def round_vector(vec, precision = 0.05):
    """
    vec: array_like, type real
    
    precision: real, > 0
    """
    return ((vec + 0.5 * precision) / precision).astype('int') * precision 


class MLVerlet:
    def __init__(self, atoms, timestep, temperature, do_vrescaling=True):
        self.atoms = atoms
        self.dt = timestep
        self.temp = temperature
        self.do_vrescaling = do_vrescaling
        
    def rescale_velocities(self):
        T = self.atoms.get_kinetic_energy() / (1.5 * units.kB * len(self.atoms))
        self.atoms.set_momenta(self.atoms.get_momenta() * (self.temp / T)**0.5)
        
    def step(self, forces, mlmodel, mixing=[1.0, 0.0], lammpsdata=None, do_update=False):
        
        atoms = self.atoms
        dt = self.dt
        p = atoms.get_momenta()
        f = forces[0] + forces[1]
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
            beta = 1 / (units.kB * self.temp)
            mlmodel.y[index] += - 1 / beta * sp.log(1 + sp.exp(- beta * (pot_energy - mlmodel.y[index])))
        else:
            mlmodel.accumulate_data(coarse_colvars, sp.array([pot_energy]))
            newdata = True
        # update ML potential if required
        if do_update:
            mlmodel.update_fit()
        # Get ML constraint forces if the model is fitted
        if hasattr(mlmodel, 'dual_coef_') and do_update: # and pot_energy < 0:
            fextra = - get_constraint_forces(atoms, mlmodel)
        else:
            if hasattr(mlmodel, 'dual_coef_'):
                print("ML model not fitted yet")
            fextra = sp.zeros(f0.shape)
        # X_near = Xplotgrid([atoms.phi() - 0.2, atoms.psi() - 0.2], [atoms.phi() - 0.2, atoms.psi() - 0.2], 2, 10)
        # y_near_mean = mlmodel.predict(X_near).mean()
        # if pot_energy < y_near_mean:
        #             mix = 1.2
        #         else:
        #             mix = 0.8
        
        # Compose the actual and the ML forces together by mixing them accordingly
        forces = [mixing[0] * f0, - mixing[1] * fextra]
        f = forces[0] + forces[1]
        atoms.set_momenta(atoms.get_momenta() + 0.5 * dt * f)
        # velocity rescale: rudimental thermostat
        if self.do_vrescaling:
            self.rescale_velocities()
        return pot_energy, [mixing[0] * f0, - mixing[1] * fextra]


class MLLangevin:
    """Langevin (constant N, V, T) molecular dynamics.

    atoms
        The list of atoms.
        
    dt
        The time step.

    temperature
        The desired temperature, in energy units (kB*T) 

    friction
        A friction coefficient, typically 1e-4 to 1e-2.

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    The temperature and friction are normally scalars, but in principle one
    quantity per atom could be specified by giving an array.
    
    Example: Langevin(atoms, 1.0*units.fs, 300*units.kB, 1.0e-2)
    """
    def __init__(self, atoms, timestep, temperature, friction, mlmodel, fixcm=True):
        self.atoms = atoms
        self.dt = timestep
        self.temp = temperature
        self.frict = friction
        self.masses = self.atoms.get_masses()
        self.fixcm = fixcm  # will the center of mass be held fixed?
        self.updatevars()
        
    def set_temperature(self, temperature):
        self.temp = temperature
        self.updatevars()

    def set_friction(self, friction):
        self.frict = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        # If the friction is an array some other constants must be arrays too.
        self._localfrict = hasattr(self.frict, 'shape')
        lt = self.frict * dt
        masses = self.masses
        sdpos = dt * sp.sqrt(self.temp / masses.reshape(-1) * (2.0/3.0 - 0.5 * lt) * lt)
        sdpos.shape = (-1, 1)
        sdmom = sp.sqrt(self.temp * masses.reshape(-1) * 2.0 * (1.0 - lt) * lt)
        sdmom.shape = (-1, 1)
        pmcor = sp.sqrt(3.0)/2.0 * (1.0 - 0.125 * lt)
        cnst = sp.sqrt((1.0 - pmcor) * (1.0 + pmcor))

        act0 = 1.0 - lt + 0.5 * lt * lt
        act1 = (1.0 - 0.5 * lt + (1.0/6.0) * lt * lt)
        act2 = 0.5 - (1.0/6.0) * lt + (1.0/24.0) * lt * lt
        c1 = act1 * dt / masses.reshape(-1)
        c1.shape = (-1, 1)
        c2 = act2 * dt * dt / masses.reshape(-1)
        c2.shape = (-1, 1)
        c3 = (act1 - act2) * dt
        c4 = act2 * dt
        del act1, act2
        if self._localfrict:
            # If the friction is an array, so are these
            act0.shape = (-1, 1)
            c3.shape = (-1, 1)
            c4.shape = (-1, 1)
            pmcor.shape = (-1, 1)
            cnst.shape = (-1, 1)
        self.sdpos = sdpos
        self.sdmom = sdmom
        self.c1 = c1
        self.c2 = c2
        self.act0 = act0
        self.c3 = c3
        self.c4 = c4
        self.pmcor = pmcor
        self.cnst = cnst
        self.natoms = self.atoms.get_number_of_atoms()
    
    
    def step(self, forces, mlmodel, extfield=None, mixing=[1.0, 0.0], lammpsdata=None, do_update=False):
        atoms = self.atoms
        p = self.atoms.get_momenta()
        f = sp.sum(forces, axis = 0)
        
        random1 = sp.random.standard_normal(size=(len(atoms), 3))
        random2 = sp.random.standard_normal(size=(len(atoms), 3))
        
        rrnd = self.sdpos * random1
        prnd = (self.sdmom * self.pmcor * random1 +
                self.sdmom * self.cnst * random2)
        
        if self.fixcm:
            rrnd = rrnd - sp.sum(rrnd, 0) / len(atoms)
            prnd = prnd - sp.sum(prnd, 0) / len(atoms)
            rrnd *= sp.sqrt(self.natoms / (self.natoms - 1.0))
            prnd *= sp.sqrt(self.natoms / (self.natoms - 1.0))
        
        self.atoms.set_positions(self.atoms.get_positions() +
                            self.c1 * p +
                            self.c2 * f + rrnd)
        p *= self.act0
        p += self.c3 * f + prnd
        self.atoms.set_momenta(p)
                      
        # get actual forces and potential energy of configuration
        pot_energy, forces = calc_lammps(atoms, preloaded_data=lammpsdata)
        forces = [forces]
        ### ML IS HERE ###
        # Accumulate the new observation in the dataset
        coarse_colvars = round_vector(self.atoms.colvars())
        distance_from_data = sp_dist.cdist(
            sp.atleast_2d(coarse_colvars), mlmodel.X_fit_).ravel()
        # check if configuration has already occurred
        if distance_from_data.min() == 0.0:
#         if newdata and do_update:
#             mlmodel.update_fit()
#             # set value to minimum energy in the bin
#             if pot_energy < mlmodel.y[index]: 
#                 mlmodel.y[index] = pot_energy
#                 do_update = True and do_update
#             else
#                 do_update = False
            index = list(distance_from_data).index(0.0)
            beta = 1 / self.temp
            mlmodel.y[index] += - 1 / beta * sp.log(
                1 + sp.exp(- beta * (pot_energy - mlmodel.y[index])))
            pot_energy = mlmodel.y[index]
        else:
            mlmodel.accumulate_data(coarse_colvars, pot_energy)
        if do_update:
            # update ML potential with all the data contained in it.
            mlmodel.update_fit()
        # Get ML constraint forces if the model is fitted
        if hasattr(mlmodel, 'dual_coef_') and pot_energy < 0:
            forces.append(get_constraint_forces(atoms, mlmodel))
        else:
            forces.append(sp.zeros(f0.shape))
        # X_near = Xplotgrid([atoms.phi() - 0.2, atoms.psi() - 0.2], [atoms.phi() - 0.2, atoms.psi() - 0.2], 2, 10)
        # y_near_mean = mlmodel.predict(X_near).mean()
        # if pot_energy < y_near_mean:
        #             mix = 1.2
        #         else:
        #             mix = 0.8

        # EXTERNAL FIELD IS HERE
        if extfield is not None:
            colvars = round_vector(self.atoms.colvars())
            extfield.update_cost(colvars, pot_energy) 
            extfield_forces = get_extfield_forces(self.atoms, extfield)
            extfield_forces /= sp.mean(map(LA.norm, extfield_forces))
            forces.append(extfield_forces)
        # Compose the actual and the ML forces together by mixing them accordingly
        # a [1,-1] mixing would result, in the perfect fitting limit, to a zero
        # mean field motion.
        forces = [m_i * f_i for m_i, f_i in zip(mixing, forces)]
        f = sp.sum(forces, axis = 0)
        self.atoms.set_momenta(self.atoms.get_momenta() + self.c4 * f)
        return pot_energy, forces


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
    setattr(Atoms, 'get_temperature', get_temperature)
    
    if "ASE_CP2K_COMMAND" not in os.environ:
        os.environ['ASE_CP2K_COMMAND'] = '/Users/marcocaccin/Code/cp2k/cp2k/exe/Darwin-IntelMacintosh-gfortran/cp2k_shell.ssmp'
    if "LAMMPS_COMMAND" not in os.environ:
        os.environ["LAMMPS_COMMAND"] = '/Users/marcocaccin/Code/lammps/src/lmp_mpi'
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'


def main():
    
    T = 300.0 # Simulation temperature
    dt = 1 * units.fs # MD timestep
    nsteps = 5000 # MD number of steps
    mixing = [1.0, -1.0, 0.5] # mixing weights for "real" and ML forces
    lengthscale = 0.6 # KRR Gaussian width.
    gamma = 1 / (2 * lengthscale**2)
    #     mlmodel = GaussianProcess(corr='squared_exponential', 
    #         # theta0=1e-1, thetaL=1e-4, thetaU=1e+2,
    #         theta0=1., 
    #         random_start=100, normalize=False, nugget=1.0e-2)
    mlmodel = KernelRidge(kernel='rbf', 
                          gamma=gamma, gammaL = gamma/4, gammaU=2*gamma,
                           alpha=5.0e-2, variable_noise=False, max_lhood=False)
    anglerange = np.arange(0, 6.3, 0.05)
    X_grid = np.array([[np.array([x,y]) for x in anglerange]
                       for y in anglerange]).reshape((126**2, 2))
    ext_field = IgnoranceField(X_grid, y_threshold=1.0e-1, cutoff = 3.)
                           
    # Bootstrap from initial database? uncomment
    # data = sp.loadtxt('phi_psi_minener_coarse_1M_md.csv') 
    # mlmodel.fit(data[:,:2], data[:,2])
    
    # Prepare diagnostic visual effects.
    plt.close('all')
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(24, 13))
    
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
    
    # Select MD propagator
    mdpropagator = MLLangevin(atoms, dt, T*units.kB, 1.0e-2, fixcm=True)
    # mdpropagator = MLVerlet(atoms, dt, T)

    # Zero-timestep evaluation and data files setup.
    print("START")
    pot_energy, f = calc_lammps(atoms, preloaded_data=lammpsdata)
    mlmodel.accumulate_data(round_vector(atoms.colvars()), pot_energy)
    printenergy(atoms, pot_energy)
    try:
        os.remove('atomstraj.xyz')
    except:
        pass
    traj = open("atomstraj.xyz", 'a')
    atoms.write(traj, format='extxyz')
    results, traj_buffer = [], []

    # When in the simulation to update the ML fit -- optional.
    teaching_points = sp.unique((sp.linspace(0, nsteps**(1/3), nsteps/20)**3).astype('int') + 1)

    # MD Loop
    for istep in range(nsteps):
        
        print("Dihedral angles | phi = %.3f, psi = %.3f " % (atoms.phi(), atoms.psi()))
        do_update = (istep % 10 == 9) # (istep in teaching_points) or (istep - nsteps == 1) # istep % 20 == 0 # 
        pot_energy, f = mdpropagator.step(f, mlmodel, extfield, mixing=mixing, lammpsdata=lammpsdata, do_update=do_update)

        # manual cooldown!!!
        if sp.absolute(atoms.get_kinetic_energy() / (1.5 * units.kB * atoms.get_number_of_atoms()) - T) > 50:
            atoms.rescale_velocities(T)

        printenergy(atoms, pot_energy/atoms.get_number_of_atoms(), step=istep)
        if do_update:
            try:
                print("Lengthscale = %.3e, Noise = %.3e" % (1/(2 * mlmodel.gamma)**0.5, mlmodel.noise.mean()))
            except:
                print("")
#         if 'datasetplot' not in locals():
#             datasetplot = pl.Plot_datapts(ax[0], mlmodel)
#         else:
#             datasetplot.update()
#         if hasattr(mlmodel, 'dual_coef_'):
#             if 'my2dplot' not in locals():
#                 my2dplot = pl.Plot_energy_n_point(ax[1], mlmodel, atoms.colvars().ravel())
#             else:
#                 my2dplot.update_prediction()
#                 my2dplot.update_current_point(atoms.colvars().ravel())
#         fig.canvas.draw()
#         # fig.canvas.print_figure('current.png')
        traj_buffer.append(atoms.copy())
        if istep % 100 == 0:
            for at in traj_buffer:
                atoms.write(traj, format='extxyz')
            traj_buffer = []
        results.append(sp.array([atoms.phi(), atoms.psi(), pot_energy]))
    traj.close()
    print("FINISHED")
    sp.savetxt('results.csv', sp.array(results))
    sp.savetxt('mlmodel.dual_coef_.csv', mlmodel.dual_coef_)
    sp.savetxt('mlmodel.X_fit_.csv', mlmodel.X_fit_)
    sp.savetxt('mlmodel.y.csv', mlmodel.y)
    calc = None
    
    return mlmodel

if __name__ == '__main__':
    initialise_env()
    # main() returns the up-to-date ML model.
    ret = main()
