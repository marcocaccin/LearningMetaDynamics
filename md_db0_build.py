from __future__ import division, print_function
import scipy as sp
import os 
from ase import units
from ase import Atoms
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

# ase Atoms functions to get collective variables for the studied molecule
dihedral_atoms_theta = [10,8,6,4]
dihedral_atoms_psi = [10,8,14,16]
fun_group_theta = range(6) + [7]
fun_group_psi = range(15,22)


def psi_(self, dihedral_list=dihedral_atoms_psi):
    return self.get_dihedral(dihedral_list)


def theta_(self, dihedral_list=dihedral_atoms_theta):
    return self.get_dihedral(dihedral_list)


def colvars(self):
    s = sp.array([self.psi(), self.theta()])
    return s


def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = sp.linspace(min(x), max(x), resX)
    yi = sp.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = sp.meshgrid(xi, yi)
    return X, Y, Z


setattr(Atoms, 'psi', psi_)
setattr(Atoms, 'theta', theta_)
setattr(Atoms, 'colvars', colvars)


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

# now extract CVs from positions

colvars = []
print("Converting positions into collective variables...")
for positions in data:
    atoms = Atoms(['H']*n_atoms, positions)
    colvars.append(atoms.colvars().flatten())
colvars = sp.asarray(colvars)


psi_theta_pot = sp.hstack((colvars,energies[:,None]))
print("Saving data...")
sp.savetxt('psi_theta_pot_100k_md.csv', psi_theta_pot)


print("Plotting trajectory...")
X, Y, Z = grid(colvars[:,0], colvars[:,1], energies)
plt.contourf(X, Y, Z, cmap='Spectral')
