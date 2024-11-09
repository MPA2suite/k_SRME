import tarfile
import warnings
import io
import os
import autoWTE
from ase.io import write
from tqdm import tqdm
from phono3py.cui.load import load
import numpy as np
import re
import ase

import pandas as pd
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

tar_dir = '/mnt/scratch2/q13camb_scratch/bp443/foundational_TC/release/phonondb-PBE-data/'
tar_list = ['phono3py_params_RS.tar.xz','phono3py_params_WZ.tar.xz','phono3py_params_ZB.tar.xz']

output_file = autoWTE.BENCHMARK_STRUCTURES_FILE

try:
    os.remove(output_file)
except OSError:
    pass

atoms_list= []

mp_phonondb_id_file = "mp_id-phonondb_id.txt"
df_mp_id = pd.read_csv(mp_phonondb_id_file, sep=" ", header=None, names=["name", "symm.no", "mp-id"])
element_pattern = r'([A-Z][a-z]?)'
df_mp_id["element_list"] = df_mp_id["name"].apply(lambda x : sorted(re.findall(element_pattern, x)))
df_mp_id["found"] = False

def get_mat_info(atoms):
    list_chemicals=sorted(np.unique(atoms.get_chemical_symbols()).tolist())

    # Regular expression to match elements with optional counts
    compound = df_mp_id[df_mp_id['element_list'].apply(lambda x: np.all(x==list_chemicals))]

    structure = compound[compound['symm.no'].apply(lambda x : int(x)) == int(get_spacegroup(atoms, symprec=1e-5).no) ]

    if len(structure) == 0:
        print(atoms,compound)
        return df_mp_id.iloc[0].to_dict()
    else:
        if len(structure) > 1:
            print(structure)
        df_mp_id.loc[structure.index,"found"] = True
        return structure.iloc[0].to_dict()


warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib") 

for tar_filename in tqdm(tar_list,desc="Tar files"): 


    with tarfile.open(tar_dir+tar_filename, 'r:xz') as tar:
        
        # Read a specific file from the archive
        all_members = tar.getmembers()
        
        # Filter for directories
        files = [member for member in all_members if member.isfile()]
        pbar = tqdm(files,leave=False,desc=f"Reading yaml files in tar file {tar_filename}")
        for file_to_read in  pbar: 

            if file_to_read.name.split(".")[-1] != "yaml" :
                continue

            file_content = tar.extractfile(file_to_read.name)
            
            if file_content is not None :
                ph3 = load(file_content,produce_fc=False,symmetrize_fc=False)
                atoms = autoWTE.phono3py2aseatoms(ph3)

                info = get_mat_info(atoms)

                pbar.set_postfix_str(info['name'])
                

                atoms.info["mp_id"] = info["mp-id"]

                atoms.info["q_mesh"] = symm_no_q_mesh_map[ph3.symmetry._dataset.number]
                atoms.info["name"] = info["name"]
                atoms.info["symm.no"] = info["symm.no"]


                write(output_file,atoms,format="extxyz",append=True)

            else:
                print(f"Could not read {file_to_read}")
            
