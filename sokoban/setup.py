#!/usr/bin/python3

import os, sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import zipfile, numpy

class my_build_ext(build_ext):
    # Override the build_ext command to set inplace=1 by default
    def initialize_options(self):
        super().initialize_options()
        self.inplace = 1

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                ext = ".cpp"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

CYTHONIZE = cythonize is not None
#CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None
#CYTHONIZE = False

extensions = [
    Extension(
        "gym_sokoban.envs.csokoban", 
        ["gym_sokoban/envs/csokoban.pyx"],
    ),
]

if CYTHONIZE:
    print("Compiling Cython sources")
    compiler_directives = {"language_level": 3}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    print("Not compiling Cython sources")
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    name="gym_sokoban",
    version="1.0.0",
    packages=["gym_sokoban"],
    ext_modules=extensions,
    install_requires=install_requires,
    include_dirs=[".", numpy.get_include()],
    cmdclass={'build_ext': my_build_ext}
)

def decompress_data():
    data_zip = os.path.join(os.path.dirname(__file__), 'resources.zip')
    extract_folder = os.path.join(os.path.dirname(__file__), 'gym_sokoban', 'envs')
    if os.path.exists(os.path.join(extract_folder, 'boxoban-levels')): return        
    
    with zipfile.ZipFile(data_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

try:
    decompress_data()
except Exception as e:
    print(f"Failed to decompress data: {e}", file=sys.stderr)