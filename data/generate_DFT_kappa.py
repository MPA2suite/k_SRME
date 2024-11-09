import tarfile
import warnings
import os, sys
import autoWTE
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc
from copy import deepcopy
from phono3py.cui.load import load
import re

from ase.spacegroup import get_spacegroup

symm_no_q_mesh_map = {
    225 : [19,19,19], # rocksalt
    186 : [19,19,15], # wurtzite 
    216 : [19,19,19], # zincblende
}

symm_no_name_map = {
    225 : "rocksalt",
    186 : "wurtzite", 
    216 : "zincblende"
}

pbar = True

tar_dir = '/mnt/scratch2/q13camb_scratch/bp443/foundational_TC/release/phonondb-PBE-data/'
tar_list = ['phono3py_params_RS.tar.xz','phono3py_params_WZ.tar.xz','phono3py_params_ZB.tar.xz']

conductivity_output= "all"

atoms_list= []

dict_dft_nac_kappa = {}
dict_dft_nonac_kappa = {}

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib") 

slurm_array_task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", "debug")
slurm_array_task_min = int(os.getenv("SLURM_ARRAY_TASK_MIN", "0"))

nac_outpath = f"{conductivity_output}_kappas_phonondb_PBE_NAC_{slurm_array_job_id}/{slurm_array_task_id}.json.gz"
nonac_outpath = f"{conductivity_output}_kappas_phonondb_PBE_noNAC_{slurm_array_job_id}/{slurm_array_task_id}.json.gz"

os.makedirs(os.path.dirname(nac_outpath),exist_ok=True)
os.makedirs(os.path.dirname(nonac_outpath),exist_ok=True)

print(f"Output to {nac_outpath} and {nonac_outpath}")

count=0
count_list=list(range(1,104))
if slurm_array_job_id == "debug":
    count_list = count_list[:2]
elif slurm_array_task_count > 1:
    count_list = count_list[slurm_array_task_id - slurm_array_task_min::slurm_array_task_count]


mp_phonondb_id_file = "mp_id-phonondb_id.txt"
df_mp_id = pd.read_csv(mp_phonondb_id_file, sep=" ", header=None, names=["name", "symm.no", "mp-id"])
element_pattern = r'([A-Z][a-z]?)'
df_mp_id["element_list"] = df_mp_id["name"].apply(lambda x : sorted(re.findall(element_pattern, x)))


def get_mat_info(atoms):
    list_chemicals=sorted(np.unique(atoms.get_chemical_symbols()).tolist())

    # Regular expression to match elements with optional counts
    compound = df_mp_id[df_mp_id['element_list'].apply(lambda x: np.all(x==list_chemicals))]

    structure = compound[compound['symm.no'].apply(lambda x : int(x)) == int(get_spacegroup(atoms, symprec=1e-5).no) ]

    return structure.iloc[0].to_dict()





for tar_filename in tqdm(tar_list,desc="Tar files",disable= not pbar): 


    with tarfile.open(tar_dir+tar_filename, 'r:xz') as tar:
        
        # Read a specific file from the archive
        all_members = tar.getmembers()
        
        # Filter for directories
        files = [member for member in all_members if member.isfile()]
        
        for file_to_read in tqdm(files,leave=False,desc=f"Reading yaml files in tar file {tar_filename}",disable= not pbar) : 

            if file_to_read.name.split(".")[-1] != "yaml" :
                continue
            else:
                count += 1
                if count not in count_list:
                    continue


            file_content = tar.extractfile(file_to_read.name)
            
            if file_content is not None :
                ph3 = load(file_content,produce_fc=True,symmetrize_fc=True)

                atoms = autoWTE.phono3py2aseatoms(ph3)

                mat_info = {}

                info = get_mat_info(atoms)

                #mat_info["mp_id"] = info["mp-id"]
                mat_info["q_mesh"] = symm_no_q_mesh_map[ph3.symmetry._dataset.number]
                mat_info["name"] = info["name"]
                mat_info["symm.no"] = info["symm.no"]

                mat_id = info["mp-id"]

                print(mat_id,"\nNAC:\n")

                ph3, kappa_nac = autoWTE.calculate_conductivity_phono3py(
                    ph3,
                    q_mesh = symm_no_q_mesh_map[ph3.symmetry._dataset.number],
                    temperatures = autoWTE.BENCHMARK_TEMPERATURES,
                    log=True,
                    dict_output=conductivity_output,
                    nac_method = "Wang"
                )

                ph3.nac_params = None

                ph3, kappa_nonac = autoWTE.calculate_conductivity_phono3py(
                    ph3,
                    q_mesh = symm_no_q_mesh_map[ph3.symmetry._dataset.number],
                    temperatures = autoWTE.BENCHMARK_TEMPERATURES,
                    log=True,
                    dict_output=conductivity_output
                )

                
                print(kappa_nac["kappa_TOT_RTA"],"\nnoNAC:\n",kappa_nonac["kappa_TOT_RTA"])
                sys.stdout.flush()

                kappa_nac.update(mat_info)
                kappa_nonac.update(mat_info)

                dict_dft_nac_kappa[mat_id] = deepcopy(kappa_nac)
                dict_dft_nonac_kappa[mat_id] = deepcopy(kappa_nonac)

                
                del ph3
                del atoms
                del kappa_nonac
                del kappa_nac
                gc.collect()

            else:
                print(f"Could not read {file_to_read}")


df_dft_nac_kappa = pd.DataFrame(dict_dft_nac_kappa).T
df_dft_nonac_kappa = pd.DataFrame(dict_dft_nonac_kappa).T


df_dft_nac_kappa.index.name = autoWTE.BENCHMARK_ID
df_dft_nonac_kappa.index.name = autoWTE.BENCHMARK_ID

df_dft_nac_kappa.reset_index().to_json(nac_outpath)
df_dft_nonac_kappa.reset_index().to_json(nonac_outpath)
            

