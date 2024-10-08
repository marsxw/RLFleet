from setuptools import setup, find_packages

setup(
    name='RLFleet',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'seaborn',
        'pandas',
        'numpy',
        'mpi4py',
        'joblib',
    ],
)
# --color orange  green  purple  aqua
# python utils/plot.py  ./data/fleet  -x TotalEnvInteracts -y AverageEpRet --save-name test.png