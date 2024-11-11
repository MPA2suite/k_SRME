import io
import sys
import datetime
import warnings
import traceback

import numpy as np

import pandas as pd


from ase.io import read, write
from ase import Atoms
from ase.utils import atoms_to_spglib_cell

from spglib import get_symmetry_dataset

FREQUENCY_THRESHOLD = -1e-2
MODE_KAPPA_THRESHOLD = 1e-6

symm_name_map = {225: "rs", 186: "wz", 216: "zb"}


def aseatoms2str(atoms, format="extxyz"):
    buffer = io.StringIO()
    write(
        buffer,
        Atoms(atoms.symbols, positions=atoms.positions, cell=atoms.cell, pbc=True),
        format=format,
    )  # You can choose different formats (like 'cif', 'pdb', etc.)
    atoms_string = buffer.getvalue()
    return atoms_string


def str2aseatoms(atoms_string, format="extxyz"):
    buffer = io.StringIO(atoms_string)
    return read(buffer, format=format)


def log_message(*messages, output=True, sep=" ", **kwargs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_message = f"{timestamp} - {sep.join((str(m) for m in messages))}"

    if output is False:
        pass
    elif output is True:
        print(write_message, **kwargs)
        sys.stdout.flush()
    elif isinstance(output, str):
        # Write to the file specified by 'output'
        with open(output, "a") as file:
            file.write(write_message + "\n", **kwargs)
    elif hasattr(output, "write"):
        # Write to the file object
        output.write(write_message + "\n", **kwargs)
        output.flush()


class ImaginaryFrequencyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def check_imaginary_freqs(frequencies):
    try:
        if np.all(pd.isna(frequencies)):
            return True

        if np.any(frequencies[0, 3:] < 0):
            return True

        if np.any(frequencies[0, :3] < FREQUENCY_THRESHOLD):
            return True

        if np.any(frequencies[1:] < 0):
            return True
    except Exception as e:
        warnings.warn(f"Failed to check imaginary frequencies: {e!r}")
        warnings.warn(traceback.format_exc())

    return False


def get_spacegroup_number(atoms, symprec=1e-5):
    dataset = get_symmetry_dataset(atoms_to_spglib_cell(atoms), symprec=symprec)
    return dataset.number


def log_symmetry(atoms, symprec, output=True):
    dataset = get_symmetry_dataset(atoms_to_spglib_cell(atoms), symprec=symprec)

    log_message(
        "Symmetry: prec",
        symprec,
        "got symmetry group number",
        dataset.number,
        ", international (Hermann-Mauguin)",
        dataset.international,
        ", Hall ",
        dataset.hall,
        output=output,
    )

    return dataset
