import sys
import numpy as np
from numpy.random import standard_normal

"""
halfstep include all the thermostat stuff but keeps the force
calculation separate.
"""
def Langevin_halfstep_1of2(self, f):
    atoms = self.atoms
    p = self.atoms.get_momenta()

    random1 = standard_normal(size=(len(atoms), 3))
    random2 = standard_normal(size=(len(atoms), 3))

    if self.communicator is not None:
        self.communicator.broadcast(random1, 0)
        self.communicator.broadcast(random2, 0)

    rrnd = self.sdpos * random1
    prnd = (self.sdmom * self.pmcor * random1 +
            self.sdmom * self.cnst * random2)

    if self.fixcm:
        rrnd = rrnd - np.sum(rrnd, 0) / len(atoms)
        prnd = prnd - np.sum(prnd, 0) / len(atoms)
        rrnd *= np.sqrt(self.natoms / (self.natoms - 1.0))
        prnd *= np.sqrt(self.natoms / (self.natoms - 1.0))

        atoms.set_positions(atoms.get_positions() +
                        self.c1 * p +
                        self.c2 * f + rrnd)
    p *= self.act0
    p += self.c3 * f + prnd
    atoms.set_momenta(p)
    return


def Langevin_halfstep_2of2(self, f):
    self.atoms.set_momenta(self.atoms.get_momenta() + self.c4 * f)
    

def Verlet_halfstep_1of2(self, f):
    p = self.atoms.get_momenta()
    p += 0.5 * self.dt * f
    self.atoms.set_positions(self.atoms.get_positions() +
        self.dt * p / self.atoms.get_masses()[:,np.newaxis])
    # We need to store the momenta on the atoms before calculating
    # the forces, as in a parallel Asap calculation atoms may
    # migrate during force calculations, and the momenta need to
    # migrate along with the atoms.  For the same reason, we
    # cannot use self.masses in the line above.
    self.atoms.set_momenta(p)
    return

def Verlet_halfstep_2of2(self, f):
    self.atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * f)
    return
