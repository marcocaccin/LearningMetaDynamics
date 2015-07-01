from __future__ import print_function, division

import os
import scipy as sp
from ase import units


kcalmol2eV = units.kcal / units.mol


def replace_positions_lammps(atoms, datafile='data.input', preloaded_data=None):
    if preloaded_data is None:
        with open(datafile, 'r') as file:
            data = file.readlines()
    else:
        data = preloaded_data
        
    n = atoms.get_number_of_atoms()
    idx = 28
    # atom positions start on line idx
    for i, (line, position) in enumerate(zip(data[idx:idx+n], atoms.get_positions())):
        line = line.split()
        line[-3:] = position.astype('string')
        data[idx+i] = ' '.join(line) + '\n'
    
    # and write everything back
    with open(datafile, 'w') as file:
        file.writelines(data)
    file.close()

    
def load_lammps_forces(atoms, dumpfile='lmp_dump.xyz'):
    # WARNING: MUST declare "sort" option in LAMMPS input file for an ordered dump file.
    # Atoms will be ordered randomly otherwise
    with open(dumpfile, 'r') as file:
        data = file.readlines()
    n = atoms.get_number_of_atoms()
    fstrings = data[-n:]
    # print(*fstrings)
    adata = sp.asarray([s.strip().split() for s in fstrings]).astype('float')
    indices = adata[:,0].astype('int') - 1
    forces = adata[indices][:,-3:]
    # convert from kcal mol-1 A-1 to eV A-1    
    forces *=  kcalmol2eV
    # print ("Forces:\n", forces)
    return forces


def load_lammps_pote(atoms, log='log'):
    with open(log, 'r') as file:
        logo = file.readlines()

    for i, line in enumerate(logo):
        idx = line.find("PotEng")
        if idx > -1:
            column = idx
            row = i
            break

    line = logo[row].split()
    pot_energy = float(line[column + 2]) 
    pot_energy *= kcalmol2eV 
    return pot_energy


def calc_lammps(atoms, datafile='data.input', preloaded_data=None, dumpfile='lmp_dump.xyz'):
    lmp_exe = os.environ["LAMMPS_COMMAND"]
    # set the atoms coordinates
    replace_positions_lammps(atoms, datafile=datafile, preloaded_data=preloaded_data)
    # run the lammps executable with the given inputs
    os.system('%s < input > log' % lmp_exe)
    # load the results in
    pot_energy = load_lammps_pote(atoms)
    forces = load_lammps_forces(atoms, dumpfile=dumpfile)
    return pot_energy, forces
    