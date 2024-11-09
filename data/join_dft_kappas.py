from glob import glob
from autoWTE import calculate_kappa_ave,glob2df, BENCHMARK_ID, calculate_mode_kappa_TOT, BENCHMARK_DFT_NAC_REF,BENCHMARK_DFT_NONAC_REF
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import gc

module_dir = os.path.dirname(__file__)

conductivity_output = "all"
benchmark_drop_list = ["mode_kappa_TOT","kappa_C","mode_kappa_C","mode_kappa_P_RTA","kappa_TOT_RTA","kappa_P_RTA"]

nac_pattern = f"{conductivity_output}_kappas_phonondb_PBE_NAC_1060107/*.json.gz"
nonac_pattern = f"{conductivity_output}_kappas_phonondb_PBE_noNAC_1060107/*.json.gz"

nac_outpath = f"{conductivity_output}_kappas_phonondb_PBE_NAC.json.gz"
nonac_outpath = f"{conductivity_output}_kappas_phonondb_PBE_noNAC.json.gz"

if conductivity_output == "benchmark":
    nac_outpath = BENCHMARK_DFT_NAC_REF
    nonac_outpath = BENCHMARK_DFT_NONAC_REF


nac_files = sorted(glob(f"{module_dir}/{nac_pattern}"))
nonac_files = sorted(glob(f"{module_dir}/{nonac_pattern}"))

df_nac = glob2df(nac_pattern).set_index(BENCHMARK_ID)



df_nac["kappa_TOT_ave"] = df_nac["kappa_TOT_RTA"].apply(calculate_kappa_ave)
df_nac["mode_kappa_TOT"] = df_nac.apply(calculate_mode_kappa_TOT,axis=1)
df_nac["mode_kappa_TOT_ave"] = df_nac["mode_kappa_TOT"].apply(calculate_kappa_ave)

if conductivity_output == "benchmark" :
    df_nac = df_nac.drop(benchmark_drop_list, axis=1)

df_nac.reset_index().to_json(nac_outpath)

del df_nac
gc.collect()

# join noNAC files
df_nonac = glob2df(nonac_pattern).set_index(BENCHMARK_ID)

df_nonac["kappa_TOT_ave"] = df_nonac["kappa_TOT_RTA"].apply(calculate_kappa_ave)
df_nonac["mode_kappa_TOT"] = df_nonac.apply(calculate_mode_kappa_TOT,axis=1)
df_nonac["mode_kappa_TOT_ave"] = df_nonac["mode_kappa_TOT"].apply(calculate_kappa_ave)

if conductivity_output == "benchmark" :
    df_nonac = df_nonac.drop(benchmark_drop_list, axis=1)

df_nonac.reset_index().to_json(nonac_outpath)



