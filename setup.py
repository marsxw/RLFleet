from setuptools import setup, find_packages

setup(
    name='RLFleet',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'seaborn==0.8.1',
        'pandas',
        'numpy',
        'mpi4py',
        'joblib',
    ],
)
# --color orange  green  purple  aqua
