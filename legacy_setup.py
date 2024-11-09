from setuptools import setup
import setuptools
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")
    

setup(
    name='k-srme',
    version=get_version("k-srme/__init__.py"),    
    description='Thermal conductivity test for foundation interatomic potentials',
    url='https://github.com/MPA2suite/k-srme',
    author='Balázs Póta, Paramvir Ahlawat, Gábor Csányi, and Michele Simoncelli',
    author_email='ms2855@cam.ac.uk or michele.simoncelli@gmail.com',
    license='GPL-3.0-only',
    packages=setuptools.find_packages(),
    install_requires=['numpy',  
                      'tqdm',
                      'ase>=3.23.0',   
                      'h5py',
                      'phonopy>=2.26.6',
                      'pandas>=2.2.3'                
                      ],
    #phono3py has to be installed separately

    classifiers=[
        'Development Status :: in development',
        'Intended Audience :: Science/Research',
        'License :: MIT',  
        'Operating System :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
    include_package_data=True,
    package_data={'k_srme': ['*']},
)