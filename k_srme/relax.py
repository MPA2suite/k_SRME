import warnings, os
from typing import Type
import numpy as np

from ase.io import read, write
from ase.io.vasp import read_vasp
from ase import Atoms
from ase.calculators.calculator import Calculator
import ase

from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter, ExpCellFilter, StrainFilter,FrechetCellFilter
from ase.optimize import BFGS, FIRE, MDMin, GPMin
from ase.spacegroup import get_spacegroup

from k_srme.utils import *


from pathlib import Path

NO_TILT_MASK=[True,True,True,False,False,False]

convert_ase_to_bar=1.602176634e-19/1e-30/1e5



def two_stage_relax(
    atoms,
    calculator : Calculator = None,
    fmax_stage1 : float = 1e-4,
    fmax_stage2 : float = 1e-4,
    steps_stage1 : int = 300,
    steps_stage2 : int = 300,
    enforce_symmetry : bool =True,
    symprec : float = 1e-5,
    allow_tilt : bool = False,
    Optimizer : Type[ase.optimize.optimize.Optimizer] = BFGS,
    Filter : Type[ase.filters.Filter] = FrechetCellFilter,
    filter_kwargs : dict | None = None,
    optim_kwargs : dict | None = None,
    log : str | Path | bool = True, # NOT WORKING FOR FILES FOR SYMMETRIES
    return_stage1 : bool = False,
    symprec_tests : list[float] = [1e-5,1e-4,1e-3,1e-1]
    ):
    
    


    if calculator is not None:
        atoms.calc=calculator
    else:
        if atoms.calc is None:
            raise ValueError("Atoms object does not have a calculator assigned")

    if filter_kwargs is None:
        _filter_kwargs = {}
    else:
        _filter_kwargs = filter_kwargs


    if optim_kwargs is None:
        _optim_kwargs = {}
    else:
        _optim_kwargs = optim_kwargs

    
    if log == False:
        ase_logfile=None
    elif log==True:
        ase_logfile='-'
    else:
        ase_logfile = log
    
    if "name" in atoms.info.keys():
        mat_name = atoms.info["name"]
    else:
        mat_name = f'{atoms.get_chemical_formula(mode="metal",empirical=True)}-{get_spacegroup(atoms,symprec=symprec).no}'


    

    tilt_mask=None
    if not allow_tilt:
        tilt_mask=NO_TILT_MASK

    input_cellpar=atoms.cell.cellpar().copy()

    log_message(f"\nRelaxing {mat_name}\n",output=log)
    log_message(f"Initial Energy {atoms.get_potential_energy()} ev",output=log)
    log_message(f"Initial Stress {atoms.get_stress()*convert_ase_to_bar} bar",output=log)
    log_message("Initial symmetry:",output=log)
    sym_init=log_symmetry(atoms, symprec, output=log)

    atoms.set_constraint(FixSymmetry(atoms))


    total_filter=Filter(atoms,mask=tilt_mask,**_filter_kwargs)
    dyn_stage1=Optimizer(total_filter,**_optim_kwargs,logfile=ase_logfile)
    dyn_stage1.run(fmax=fmax_stage1,steps=steps_stage1)



    log_message(f"After keeping symmetry stage 1 relax, energy {atoms.get_potential_energy()} ev",output=log)
    log_message(f"After keeping symmetry stage 1 relax, stress {atoms.get_stress()*convert_ase_to_bar} bar",output=log)

    cell_diff = (atoms.cell.cellpar() / input_cellpar - 1.0) * 100
    log_message("Stage 1 Cell         :", atoms.cell.cellpar(),output=log)
    log_message("Stage 1 Cell diff (%):", cell_diff,output=log)

    # We print out the initial symmetry groups
    log_message("After keeping symmetry stage 1 relax, symmetry:",output=log)
    sym_stage1=log_symmetry(atoms, symprec, output=log)



    if sym_stage1['number']!=sym_init['number']:
        warnings.warn(f"Symmetry is not kept during FixSymmetry relaxation of material {mat_name} in folder {os.getcwd()}")
        log_message(f"Symmetry is not kept during FixSymmetry relaxation of material {mat_name} in folder {os.getcwd()}",output=log)

    max_stress_stage1 = atoms.get_stress().reshape((2,3),order='C').max(axis=1)


    atoms_stage1=atoms.copy()
    atoms.constraints = None


    # Stage 2
    dyn_stage2=Optimizer(total_filter,**_optim_kwargs,logfile=ase_logfile)
    dyn_stage2.run(fmax=fmax_stage2,steps=steps_stage2)


    log_message("Stage 2 Energy", atoms.get_potential_energy()," ev",output=log)
    log_message("Stage 2 Stress",atoms.get_stress()*convert_ase_to_bar," bar",output=log)
    log_message("Stage 2 Symmetry:",output=log)
    sym_stage2=log_symmetry(atoms, symprec, output=log)
    cell_diff = (atoms.cell.cellpar() / input_cellpar - 1.0) * 100
    log_message(f"Stage 2 Cell         : {atoms.cell.cellpar()}",output=log)
    log_message(f"Stage 2 Cell diff (%): {cell_diff}\n",output=log)


    # Test symmetries with various symprec if stage2 is different
    sym_tests = {}
    if sym_init.number!=sym_stage2.number:
        for symprec_test in symprec_tests:
            log_message("Stage 2 Symmetry Test:",output=log)
            dataset_tests=log_symmetry(atoms, symprec_test, output=log)
            sym_tests[symprec_test]=dataset_tests.number

    max_stress_stage2 = atoms.get_stress().reshape((2,3),order='C').max(axis=1)

    atoms_stage2=atoms.copy()


    # compare symmetries and redirect to stage 1
    if sym_stage1.number!=sym_stage2.number and enforce_symmetry:
        redirected_to_symm = True
        atoms=atoms_stage1
        max_stress= max_stress_stage1
        sym_final = sym_stage1
        warnings.warn(f"Symmetry is not kept after deleting FixSymmetry constraint, redirecting to structure with symmetry of material {mat_name}, in folder {os.getcwd()}")
        log_message(f"Symmetry is not kept after deleting FixSymmetry constraint, redirecting to structure with symmetry of material {mat_name}, in folder {os.getcwd()}",output=log)

    else:
        redirected_to_symm = False
        sym_final = sym_stage2
        max_stress = max_stress_stage2
    

    # maximum residual stress component in for xx,yy,zz and xy,yz,xz components separately
    # result is a array of 2 elements
    reached_max_steps = dyn_stage1.step == steps_stage1 or dyn_stage2.step == steps_stage2

    relax_dict = {
        'max_stress': max_stress, 
        'reached_max_steps':reached_max_steps,
        'relaxed_space_group_number':sym_final.number,
        'broken_symmetry': sym_final.number != sym_init.number,
        'symprec_tests':sym_tests,
        'redirected_to_symm':redirected_to_symm}


    if return_stage1:
        # first return is final, second is stage 1, third is stage 2
        return_atoms = [atoms,atoms_stage1,atoms_stage2] 
    else:
        return_atoms = atoms

    return return_atoms, relax_dict

