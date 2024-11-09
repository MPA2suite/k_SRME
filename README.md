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

 Pre-requisites (need to be installed seperately or added to PYTHONPATH)
- phono3py (see https://phonopy.github.io/phono3py/install.html for installation instructions)


Installed automatically during pip install:
- phonopy
- ase
- numpy
- matplotlib
- spglib
- tqdm
- h5py
- pandas




# Usage
The example scripts showcase a sample workflow for testing a MACE potential and comparing the thermal conductivity with DFT calculations for a collection of different materials. The scripts may be modified easily to use any foundation Machine Learning Interatomic Potentials. See k-srme/MLPS.py for calculator setup utilities.


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
