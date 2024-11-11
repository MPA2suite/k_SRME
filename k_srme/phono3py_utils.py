from ase import Atoms

from phonopy.structure.atoms import PhonopyAtoms

from phono3py.api_phono3py import Phono3py


def aseatoms2phonoatoms(atoms):
    phonoatoms = PhonopyAtoms(
        atoms.symbols, cell=atoms.cell, positions=atoms.positions, pbc=True
    )
    return phonoatoms


def aseatoms2phono3py(
    atoms, fc2_supercell, fc3_supercell, primitive_matrix=None, **kwargs
) -> Phono3py:
    unitcell = aseatoms2phonoatoms(atoms)
    return Phono3py(
        unitcell=unitcell,
        supercell_matrix=fc3_supercell,
        phonon_supercell_matrix=fc2_supercell,
        primitive_matrix=primitive_matrix,
        **kwargs,
    )


def phono3py2aseatoms(ph3: Phono3py) -> Atoms:
    phonopyatoms = ph3.unitcell
    atoms = Atoms(
        phonopyatoms.symbols,
        cell=phonopyatoms.cell,
        positions=phonopyatoms.positions,
        pbc=True,
    )

    if ph3.supercell_matrix is not None:
        atoms.info["fc3_supercell"] = ph3.supercell_matrix

    if ph3.phonon_supercell_matrix is not None:
        atoms.info["fc2_supercell"] = ph3.phonon_supercell_matrix

    if ph3.primitive_matrix is not None:
        atoms.info["primitive_matrix"] = ph3.primitive_matrix

    if ph3.mesh_numbers is not None:
        atoms.info["q_mesh"] = ph3.mesh_numbers

    # TODO : Non-default values and BORN charges to be added

    return atoms


def get_chemical_formula(ph3: Phono3py, mode="metal", **kwargs):
    unitcell = ph3.unitcell
    atoms = Atoms(
        unitcell.symbols, cell=unitcell.cell, positions=unitcell.positions, pbc=True
    )
    return atoms.get_chemical_formula(mode=mode, **kwargs)
