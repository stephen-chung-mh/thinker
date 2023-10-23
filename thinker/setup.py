from setuptools import setup
from Cython.Build import cythonize


setup(ext_modules=cythonize("thinker/cenv.pyx", include_path=[]))
