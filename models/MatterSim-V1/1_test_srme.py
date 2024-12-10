import os
import datetime
import warnings
from typing import Literal, Any
from collections.abc import Callable
import traceback
from copy import deepcopy
from importlib.metadata import version
import json

import pandas as pd

from tqdm import tqdm

from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from ase import Atoms
from ase.io import read

from k_srme import aseatoms2str, two_stage_relax, ID, STRUCTURES, NO_TILT_MASK
from k_srme.utils import symm_name_map, get_spacegroup_number, check_imaginary_freqs
from k_srme.conductivity import (
    init_phono3py,
    get_fc2_and_freqs,
    get_fc3,
    calculate_conductivity,
)

from mattersim.forcefield import MatterSimCalculator

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")


# EDITABLE CONFIG
model_name = "MatterSim-V1"


# load the default checkpoint: 1M
calc = MatterSimCalculator(device="cuda")
checkpoint = "mattersim-v1.0.0-1m"
suffix = "1M"

# load the checkpoint of 5M parameters
# calc = MatterSimCalculator(device="cuda", load_path="mattersim-v1.0.0-5m")
# checkpoint = "mattersim-v1.0.0-5m"
# suffix="5M"


# Relaxation parameters
ase_optimizer: Literal["FIRE", "LBFGS", "BFGS"] = "FIRE"
ase_filter: Literal["frechet", "exp"] = "frechet"
if_two_stage_relax = True  # Use two-stage relaxation enforcing symmetries
max_steps = 300
force_max = 1e-4  # Run until the forces are smaller than this in eV/A

# Symmetry parameters
# symmetry precision for enforcing relaxation and conductivity calculation
symprec = 1e-5
# Enforce symmetry with during relaxation if broken
enforce_relax_symm = True
# Conductivity to be calculated if symmetry group changed during relaxation
conductivity_broken_symm = False
prog_bar = True
save_forces = True  # Save force sets to file


task_type = "LTC"  # lattice thermal conductivity
job_name = f"{model_name}-phononDB-{task_type}-{ase_optimizer}{'_2SR' if if_two_stage_relax else ''}_force{force_max}_sym{symprec}-{suffix}"
module_dir = os.path.dirname(__file__)
out_dir = f"{module_dir}/{datetime.datetime.now().strftime('%Y-%m-%d')}-{job_name}"
os.makedirs(out_dir, exist_ok=True)

out_path = f"{out_dir}/conductivity.json.gz"


timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
struct_data_path = STRUCTURES
print(f"\nJob {job_name} started {timestamp}")


print(f"Read data from {struct_data_path}")
atoms_list: list[Atoms] = read(struct_data_path, format="extxyz", index=":")

run_params = {
    "timestamp": timestamp,
    "k_srme_version": version("k_srme"),
    "model_name": model_name,
    # "checkpoint": checkpoint,
    "versions": {dep: version(dep) for dep in ("numpy", "torch")},
    "ase_optimizer": ase_optimizer,
    "ase_filter": ase_filter,
    "if_two_stage_relax": if_two_stage_relax,
    "max_steps": max_steps,
    "force_max": force_max,
    "symprec": symprec,
    "enforce_relax_symm": enforce_relax_symm,
    "conductivity_broken_symm": conductivity_broken_symm,
    # "slurm_array_task_count": slurm_array_task_count,
    # "slurm_array_job_id": slurm_array_job_id,
    "task_type": task_type,
    "job_name": job_name,
    "struct_data_path": os.path.basename(struct_data_path),
    "n_structures": len(atoms_list),
}

# if slurm_array_task_id == slurm_array_task_min:
if True:
    with open(f"{out_dir}/run_params.json", "w") as f:
        json.dump(run_params, f, indent=4)

atoms_list = atoms_list[:5]

# Set up the relaxation and force set calculation
filter_cls: Callable[[Atoms], Atoms] = {
    "frechet": FrechetCellFilter,
    "exp": ExpCellFilter,
}[ase_filter]
optim_cls: Callable[..., Optimizer] = {"FIRE": FIRE, "LBFGS": LBFGS}[ase_optimizer]


force_results: dict[str, dict[str, Any]] = {}
kappa_results: dict[str, dict[str, Any]] = {}


tqdm_bar = tqdm(atoms_list, desc="Conductivity calculation: ", disable=not prog_bar)

for atoms in tqdm_bar:
    mat_id = atoms.info[ID]
    init_info = deepcopy(atoms.info)
    mat_name = atoms.info["name"]
    mat_desc = f"{mat_name}-{symm_name_map[atoms.info['symm.no']]}"
    info_dict = {
        "desc": mat_desc,
        "name": mat_name,
        "initial_space_group_number": atoms.info["symm.no"],
        "errors": [],
        "error_traceback": [],
    }

    tqdm_bar.set_postfix_str(mat_desc, refresh=True)

    # Relaxation
    try:
        atoms.calc = calc
        if max_steps > 0:
            if not if_two_stage_relax:
                if enforce_relax_symm:
                    atoms.set_constraint(FixSymmetry(atoms))
                    filtered_atoms = filter_cls(atoms, mask=NO_TILT_MASK)
                else:
                    filtered_atoms = filter_cls(atoms)

                optimizer = optim_cls(filtered_atoms, logfile=f"{out_dir}/relax.log")
                optimizer.run(fmax=force_max, steps=max_steps)

                reached_max_steps = False
                if optimizer.step == max_steps:
                    reached_max_steps = True
                    print(
                        f"Material {mat_desc=}, {mat_id=} reached max step {max_steps=} during relaxation."
                    )

                # maximum residual stress component in for xx,yy,zz and xy,yz,xz components separately
                # result is a array of 2 elements
                max_stress = atoms.get_stress().reshape((2, 3), order="C").max(axis=1)

                atoms.calc = None
                atoms.constraints = None
                atoms.info = init_info | atoms.info

                symm_no = get_spacegroup_number(atoms, symprec=symprec)

                relax_dict = {
                    "structure": aseatoms2str(atoms),
                    "max_stress": max_stress,
                    "reached_max_steps": reached_max_steps,
                    "relaxed_space_group_number": symm_no,
                    "broken_symmetry": symm_no
                    != init_info["initial_space_group_number"],
                }

            else:
                atoms, relax_dict = two_stage_relax(
                    atoms,
                    fmax_stage1=force_max,
                    fmax_stage2=force_max,
                    steps_stage1=max_steps,
                    steps_stage2=max_steps,
                    Optimizer=optim_cls,
                    Filter=filter_cls,
                    allow_tilt=False,
                    log=f"{out_dir}/relax.log",
                    enforce_symmetry=enforce_relax_symm,
                )

                atoms.calc = None

    except Exception as exc:
        warnings.warn(f"Failed to relax {mat_name=}, {mat_id=}: {exc!r}")
        traceback.print_exc()
        info_dict["errors"].append(f"RelaxError: {exc!r}")
        info_dict["error_traceback"].append(traceback.format_exc())
        kappa_results[mat_id] = info_dict
        continue

    # Calculation of force sets
    try:
        ph3 = init_phono3py(atoms, log=False, symprec=symprec)

        ph3, fc2_set, freqs = get_fc2_and_freqs(
            ph3,
            calculator=calc,
            log=False,
            pbar_kwargs={"leave": False, "disable": not prog_bar},
        )

        imaginary_freqs = check_imaginary_freqs(freqs)
        freqs_dict = {"imaginary_freqs": imaginary_freqs, "frequencies": freqs}

        # if conductivity condition is met, calculate fc3
        ltc_condition = not imaginary_freqs and (
            not relax_dict["broken_symmetry"] or conductivity_broken_symm
        )

        if ltc_condition:
            ph3, fc3_set = get_fc3(
                ph3,
                calculator=calc,
                log=False,
                pbar_kwargs={"leave": False, "disable": not prog_bar},
            )

        else:
            fc3_set = []

        if save_forces:
            force_results[mat_id] = {"fc2_set": fc2_set, "fc3_set": fc3_set}

        if not ltc_condition:
            kappa_results[mat_id] = info_dict | relax_dict | freqs_dict
            continue

    except Exception as exc:
        warnings.warn(f"Failed to calculate force sets {mat_id}: {exc!r}")
        traceback.print_exc()
        info_dict["errors"].append(f"ForceConstantError: {exc!r}")
        info_dict["error_traceback"].append(traceback.format_exc())
        kappa_results[mat_id] = info_dict | relax_dict
        continue

    # Calculation of conductivity
    try:
        ph3, kappa_dict = calculate_conductivity(ph3, log=False)

    except Exception as exc:
        warnings.warn(f"Failed to calculate conductivity {mat_id}: {exc!r}")
        traceback.print_exc()
        info_dict["errors"].append(f"ConductivityError: {exc!r}")
        info_dict["error_traceback"].append(traceback.format_exc())
        kappa_results[mat_id] = info_dict | relax_dict | freqs_dict
        continue

    kappa_results[mat_id] = info_dict | relax_dict | freqs_dict | kappa_dict


df_kappa = pd.DataFrame(kappa_results).T
df_kappa.index.name = ID
df_kappa.reset_index().to_json(out_path)


if save_forces:
    force_out_path = f"{out_dir}/force_sets.json.gz"
    df_force = pd.DataFrame(force_results).T
    df_force = pd.concat([df_kappa, df_force], axis=1)
    df_force.index.name = ID
    df_force.reset_index().to_json(force_out_path)
