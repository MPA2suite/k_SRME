import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from phono3py.api_phono3py import Phono3py
from tqdm import tqdm

from k_srme import TEMPERATURES
from k_srme.benchmark import calculate_mode_kappa_TOT
from k_srme.phono3py_utils import aseatoms2phono3py, get_chemical_formula
from k_srme.utils import MODE_KAPPA_THRESHOLD, log_message

KAPPA_OUTPUT_NAME_MAP = {
    "weights": "grid_weights",
    "heat_capacity": "mode_heat_capacities",
}

CONDUCTIVITY_SIGMAS_LIST = [
    "kappa",
    "mode_kappa",
    "kappa_TOT_RTA",
    "kappa_P_RTA",
    "kappa_C",
    "mode_kappa_P_RTA",
    "mode_kappa_C",
    "gamma_isotope",
]


def calculate_fc2_set(
    ph3: Phono3py,
    calculator: Calculator,
    log: bool = True,
    pbar_kwargs: dict[str, Any] = {},
) -> np.ndarray:
    # calculate FC2 force set

    log_message(f"Computing FC2 force set in {get_chemical_formula(ph3)}.", output=log)

    forces = []
    nat = len(ph3.phonon_supercell)

    for sc in tqdm(
        ph3.phonon_supercells_with_displacements,
        desc=f"FC2 calculation: {get_chemical_formula(ph3)}",
        **pbar_kwargs,
    ):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc = calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.phonon_forces = force_set
    return force_set


def calculate_fc3_set(
    ph3: Phono3py,
    calculator: Calculator,
    log: bool = True,
    pbar_kwargs: dict[str, Any] = {},
) -> np.ndarray:
    # calculate FC3 force set

    log_message(f"Computing FC3 force set in {get_chemical_formula(ph3)}.", output=log)

    forces = []
    nat = len(ph3.supercell)

    for sc in tqdm(
        ph3.supercells_with_displacements,
        desc=f"FC3 calculation: {get_chemical_formula(ph3)}",
        **pbar_kwargs,
    ):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc = calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.forces = np.array(forces)
    return force_set


def init_phono3py(
    atoms: Atoms,
    log: str | Path | bool = True,
    symprec: float = 1e-5,
    displacement_distance: float = 0.03,
    **kwargs: Any,
) -> tuple[Phono3py, list[Any], list[Any]]:
    """Calculate fc2 and fc3 force lists from phono3py.

    Args:


    Raises:


    Returns:

    """
    if not log:
        log_level = 0
    elif log is not None:
        log_level = 1

    formula = atoms.get_chemical_formula(mode="metal")
    for key in ("fc2_supercell", "fc3_supercell", "q_mesh"):
        if key not in atoms.info:
            raise ValueError(
                f'{formula} "{key}" was not found in atoms.info when calculating force sets.'
            )

    # Initialise Phono3py object
    ph3 = aseatoms2phono3py(
        atoms,
        fc2_supercell=atoms.info["fc2_supercell"],
        fc3_supercell=atoms.info["fc3_supercell"],
        primitive_matrix="auto",
        symprec=symprec,
        log_level=log_level,
        **kwargs,
    )

    ph3.mesh_numbers = atoms.info["q_mesh"]

    ph3.generate_displacements(distance=displacement_distance)

    return ph3


def get_fc2_and_freqs(
    ph3: Phono3py,
    calculator: Calculator | None = None,
    log: str | Path | bool = True,
    pbar_kwargs: dict[str, Any] = {"leave": False},
) -> tuple[Phono3py, np.ndarray, np.ndarray]:
    if ph3.mesh_numbers is None:
        raise ValueError(
            '"mesh_number" was not found in phono3py object and was not provided as an argument when calculating phonons from phono3py object.'
        )

    if calculator is None:
        raise ValueError(
            f'{get_chemical_formula(ph3)} "calculator" was provided when calculating fc2 force sets.'
        )

    fc2_set = calculate_fc2_set(ph3, calculator, log=log, pbar_kwargs=pbar_kwargs)

    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_phonon_solver()

    freqs, eigvecs, grid = ph3.get_phonon_data()

    return ph3, fc2_set, freqs


def get_fc3(
    ph3: Phono3py,
    calculator: Calculator | None = None,
    log: str | Path | bool = True,
    pbar_kwargs: dict[str, Any] = {"leave": False},
) -> tuple[Phono3py, np.ndarray]:
    if calculator is None:
        raise ValueError(
            f'{get_chemical_formula(ph3)} "calculator" was provided when calculating fc3 force sets.'
        )

    fc3_set = calculate_fc3_set(ph3, calculator, log=log, pbar_kwargs=pbar_kwargs)

    ph3.produce_fc3(symmetrize_fc3r=True)

    return ph3, fc3_set


def load_force_sets(
    ph3: Phono3py, fc2_set: np.ndarray, fc3_set: np.ndarray
) -> Phono3py:
    ph3.phonon_forces = fc2_set
    ph3.forces = fc3_set
    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.produce_fc3(symmetrize_fc3r=True)

    return ph3


def calculate_conductivity(
    ph3: Phono3py,
    temperatures: np.ndarray = TEMPERATURES,
    log: str | Path | bool | None = None,
    **kwargs: Any,
) -> tuple[Phono3py, dict[str, np.ndarray]]:
    if not log:
        ph3._log_level = 0
    elif log is not None:
        ph3._log_level = 1

    ph3.init_phph_interaction(symmetrize_fc3q=False)

    ph3.run_thermal_conductivity(
        temperatures=temperatures,
        is_isotope=True,
        conductivity_type="wigner",
        boundary_mfp=1e6,
        **kwargs,
    )

    cond = ph3.thermal_conductivity

    kappa_dict = {}

    try:
        kappa_dict["kappa_TOT_RTA"] = deepcopy(cond.kappa_TOT_RTA[0])
        kappa_dict["kappa_P_RTA"] = deepcopy(cond.kappa_P_RTA[0])
        kappa_dict["kappa_C"] = deepcopy(cond.kappa_C[0])
        kappa_dict["weights"] = deepcopy(cond.grid_weights)
        kappa_dict["q_points"] = deepcopy(cond.qpoints)
        # kappa_dict["frequencies"] = deepcopy(cond.frequencies)
    except AttributeError as exc:
        warnings.warn(f"Phono3py conductivity does not have attribute: {exc}")

    try:
        mode_kappa_TOT = calculate_mode_kappa_TOT(
            deepcopy(cond.mode_kappa_P_RTA[0]),
            deepcopy(cond.mode_kappa_C[0]),
            deepcopy(cond.mode_heat_capacities),
        )
    except AttributeError as exc:
        warnings.warn(
            f"Calculate mode kappa tot failed in {get_chemical_formula(ph3)}: {exc}"
        )

    kappa_dict["mode_kappa_TOT"] = mode_kappa_TOT

    sum_mode_kappa_TOT = mode_kappa_TOT.sum(
        axis=tuple(range(1, mode_kappa_TOT.ndim - 1))
    ) / np.sum(kappa_dict["weights"])

    if np.all((sum_mode_kappa_TOT - kappa_dict["kappa_P_RTA"]) <= MODE_KAPPA_THRESHOLD):
        warnings.warn(
            f"Total mode kappa does not sum to total kappa. mode_kappa_TOT sum : {sum_mode_kappa_TOT}, kappa_TOT_RTA : {kappa_dict['kappa_P_RTA']}"
        )

    return ph3, kappa_dict
