from __future__ import division, print_function
import scipy as sp
import scipy.linalg as LA

import os 
from ase import units
from ase import Atoms
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import pickle as pkl

dihedral_atoms_phi = [4,6,8,14] # C(O)-N-C(a)-C(O)
dihedral_atoms_psi = [6,8,14,16] # N-C(a)-C(O)-N

fun_group_phi = range(6) + [7]
fun_group_psi = range(15,22)
#############################################################


##### Utility functions to be added to ase.Atoms Class #####


def phi_(self, dihedral_list=dihedral_atoms_phi):
    return self.get_dihedral(dihedral_list)


def psi_(self, dihedral_list=dihedral_atoms_psi):
    return self.get_dihedral(dihedral_list)


def colvars_(self):
    s = sp.atleast_2d(sp.array([self.phi(), self.psi()]))
    return s


def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = sp.linspace(min(x), max(x), resX)
    yi = sp.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = sp.meshgrid(xi, yi)
    return X, Y, Z



def round_vector(vec, precision = 0.1):
    return ((vec + 0.5 * precision)/ precision).astype('int') * precision

### CODE STARTS HERE ###

run_from_scratch = False
T = 300

if run_from_scratch:
    setattr(Atoms, 'phi', phi_)
    setattr(Atoms, 'psi', psi_)
    setattr(Atoms, 'colvars', colvars_)

    os.system('lmp_mpi < input_md')


    # load trajectory and get atomic positions into adata
    print("Reading positions from trajectory file...")
    data = []
    with open('lmp_md.xyz', 'r') as file:
        for i, line in enumerate(file.readlines()):
            if i % 31 > 8:
                data.append(line.split()[2:5])
    n_atoms = 22 

    print("Converting data...")
    data = sp.asarray(data).astype('float')
    data = data.reshape((len(data)/n_atoms, n_atoms, 3))

    # write potential energies to file
    print("Reading potential energies...")
    os.system('grep PotEng log.lammps | awk {\'print $3\'} > PotEng.md')
    energies = sp.loadtxt('PotEng.md')
    energies *= units.kcal / units.mol

    # write kinetic energies to file
    os.system('grep KinEng log.lammps | awk {\'print $6\'} > KinEng.md')
    kinengs = sp.loadtxt('KinEng.md')
    kinengs *= units.kcal / units.mol
    
    # now extract CVs from positions

    colvars = []
    print("Converting positions into collective variables...")
    for positions in data:
        atoms = Atoms(['H']*n_atoms, positions)
        colvars.append(atoms.colvars().flatten())
    colvars = sp.asarray(colvars)


    phipsi_pot = sp.hstack((colvars,energies[:,None], kinengs[:,None]))
    print("Saving data...")
    sp.savetxt('phi_psi_pot_md300.csv', phipsi_potkin)
else:
    data = sp.loadtxt('phi_psi_pot_md300.csv')
    colvars = data[:,:2]
    energies = data[:,2]
    kinengs = data[:,3]
    
colvars_r = round_vector(colvars)
energies_r = {}
kinengs_r = {}
for i, s in enumerate(colvars_r):
    if ('%f-%f' % (s[0], s[1])) in energies_r.keys():
        energies_r['%f-%f' % (s[0], s[1])].append(energies[i])
        kinengs_r['%f-%f' % (s[0], s[1])].append(kinengs[i])
    else:
        energies_r['%f-%f' % (s[0], s[1])] = [energies[i]]
        kinengs_r['%f-%f' % (s[0], s[1])] = [kinengs[i]]

colvars_2 = []
energies_mean = []
energies_min = []
n_confs = []
free_energies = []
ftilde = []
ftilde2 = []

meankin = 1.5 * 22 * units.kB * T
meane_plus_kin = []

beta = 1 / (units.kB * T)
for s, energy in energies_r.iteritems():
    energy = sp.array(energy)
    kin = sp.array(kinengs_r[s])
    colvars_2.append(sp.array(s.split('-')).astype('float'))
    energies_mean.append(energy.mean())
    energies_min.append(energy.min())
    n_confs.append(len(energy))
    free_energies.append(- units.kB * T * sp.log(sp.exp(-energy / (units.kB * T)).sum()))
    ftilde.append(units.kB * T * (- sp.log(n_confs[-1]) + sp.log(sp.mean(sp.exp(beta * (energy - energies_min[-1]))))))
    ftilde2.append(units.kB * T * (- sp.log(n_confs[-1]) + sp.log(sp.mean(sp.exp(beta * energy)))))

    meane_plus_kin.append(energies_mean[-1] + kin.mean() - meankin)
    
colvars_2 = sp.array(colvars_2)
n_confs = sp.array(n_confs)
energies_min = sp.array(energies_min)
energies_mean = sp.array(energies_mean)
free_energies = sp.array(free_energies)
ftilde = sp.array(ftilde)
ftilde2 = sp.array(ftilde2)

meane_plus_kin = sp.array(meane_plus_kin)

phi, psi = colvars_2[:,0], colvars_2[:,1]
phimin, phimax = phi.min(), phi.max()
psimin, psimax = psi.min(), psi.max()
phirange = phimax - phimin
psirange = psimax - psimin
aspect_ratio = psirange/phirange
print("Plotting trajectory...")

fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=energies_mean, marker = 's', s = 290,
                cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('energy_mean.png')
ax.clear()

fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=energies_min, marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('energy_min.png')
ax.clear()

fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=free_energies, marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('free_energy.png')
ax.clear()


fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=(energies_min - free_energies), marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('TS_min.png')
ax.clear()

fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=(energies_min + ftilde), marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('Emin_plus_Ftilde.png')
ax.clear()


fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=ftilde, marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('Ftilde_only.png')
ax.clear()

fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=ftilde2, marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('Ftilde2_only.png')
ax.clear()

fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=-units.kB * T * sp.log(n_confs), marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('F_from_N.png')
ax.clear()

fig, ax = plt.subplots(1,1,figsize=(10,10*aspect_ratio))
sc = ax.scatter(phi, psi, c=meane_plus_kin, marker = 's', s = 290,
           cmap = 'RdBu', alpha = .8, edgecolors='none')
ax.set_xlim(phimin, phimax)
ax.set_ylim(psimin, psimax)
plt.colorbar(sc, format='%.3e')
fig.savefig('meane_plus_kin.png')
ax.clear()
