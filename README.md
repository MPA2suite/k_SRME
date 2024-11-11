# κ_SRME: heat-conductivity benchmark test for foundational machine-learning potentials based on the Wigner Transport Equation 

κ_SRME employs foundation Machine Learning Interatomic Potentials and phono3py to determine the wigner thermal conductivity in crystals and compare them to DFT reference data.

# Install 
Clone repository:
```
git clone https://github.com/MPA2suite/k-srme.git
```
Then install in editable mode:
```
pip install -e .
```


phonopy and phono3py dependencies are optional to allow model output analysis.
 These pre-requisites need to be installed seperately or added to PYTHONPATH.
See https://phonopy.github.io/phono3py/install.html for installation instructions of phono3py.



# Usage
The example scripts showcase a sample workflow for testing a MACE potential and comparing the thermal conductivity with DFT calculations for a collection of different materials. The scripts may be modified easily to use any foundation Machine Learning Interatomic Potentials. 

Example scripts are found in the `scripts` folder. Model results and scripts are found in the `models` folder. 

To obtain conductivity results, you need to run a CPU job, as phono3py does not support GPUs. The `1_test_srme.py` script calculates the displaced force sets and the thermal conductivity for each material. We recommend setting OMP_NUM_THREADS to 4 to 8, to get speedup in both the forceand conductivity calculations. The script also supports job arrays outputting one file per array task, which are collected in the evaluation script. For the 103 materials, the wurtzite structures require the longest runtime. Therefore to minimize the runtime, we recommend a maximum of 33 array tasks.

The `2_evaluate.py` script evaluates the predictions, collecting the array task files and printing the results both to the terminal and to a file. The `k_srme.json.gz` output file contain additional information about the model run, which can be read as a pandas DataFrame for further analysis.




# How to cite

```
@misc{póta2024thermalconductivitypredictionsfoundation,
      title={Thermal Conductivity Predictions with Foundation Atomistic Models}, 
      author={Balázs Póta and Paramvir Ahlawat and Gábor Csányi and Michele Simoncelli},
      year={2024},
      eprint={2408.00755},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2408.00755}, 
}
```
