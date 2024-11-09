import os


PKG_NAME = "k-srme"
__version__ = "1.0.0"

PKG_DIR = os.path.dirname(__file__)
# repo root directory if editable install, TODO: else the package dir
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"  # directory to store default data
DATA_DIR = f"{DATA_DIR}"
STRUCTURES_FILE = "phononDB-PBE-structures.extxyz"
STRUCTURES = f"{DATA_DIR}/{STRUCTURES_FILE}"

DFT_NAC_REF_FILE = "kappas_phononDB_PBE_NAC.json.gz"
DFT_NONAC_REF_FILE = "kappas_phononDB_PBE_noNAC.json.gz"
DFT_NAC_REF = f"{DATA_DIR}/{DFT_NAC_REF_FILE}"
DFT_NONAC_REF = f"{DATA_DIR}/{DFT_NONAC_REF_FILE}"

###
pkg_is_editable = True


######
TEMPERATURES = [300]
ID = "mp_id"

from .benchmark import *
from .data import glob2df

# TODO: remove these such that code is not dependent on phono3py
from .utils import *
from .relax import *
