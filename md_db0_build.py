from __future__ import division, print_function
import scipy as sp
import scipy.linalg as LA

import os 
from ase import units
from ase import Atoms
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

dihedral_atoms_phi = [4,6,8,14] # C(O)-N-C(a)-C(O)
dihedral_atoms_psi = [6,8,14,16] # N-C(a)-C(O)-N

fun_group_phi = range(6) + [7]
fun_group_psi = range(15,22)
#############################################################


##### Utility functions to be added to ase.Atoms Class #####
def get_my_dihedral(self, list):
    """Calculate dihedral angle.

    Calculate dihedral angle between the vectors list[0]->list[1]
    and list[2]->list[3], where list contains the atomic indexes
    in question.
    """

    # vector 0->1, 1->2, 2->3 and their normalized cross products:
    a = self.positions[list[1]] - self.positions[list[0]]
    b = self.positions[list[2]] - self.positions[list[1]]
    c = self.positions[list[3]] - self.positions[list[2]]
    bxa = sp.cross(b, a)
    bxa /= LA.norm(bxa)
    cxb = sp.cross(c, b)
    cxb /= LA.norm(cxb)
    angle = sp.vdot(bxa, cxb)
    angle = sp.arccos(angle)
    return angle


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


def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = sp.linspace(min(x), max(x), resX)
    yi = sp.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = sp.meshgrid(xi, yi)
    return X, Y, Z


# setattr(Atoms, 'get_my_dihedral', get_my_dihedral)
setattr(Atoms, 'phi', phi_)
setattr(Atoms, 'psi', psi_)
setattr(Atoms, 'colvars', colvars)

# os.system('lmp_mpi < input_md')


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


phipsi_pot = sp.hstack((colvars,energies[:,None]))
print("Saving data...")
sp.savetxt('psi_theta_pot_100k_md.csv', phipsi_pot)


print("Plotting trajectory...")
X, Y, Z = grid(colvars[:,0], colvars[:,1], energies)
plt.contourf(X, Y, Z, cmap='Spectral')
