#!/usr/bin/python3

import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Override the build_ext command to set inplace=1 by default
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
#CYTHONIZE = True

extensions = [Extension("thinker.cenv", ["thinker/cenv.pyx"])]
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
    name="thinker",
    version="1.1.0",
    packages=["thinker"],
    ext_modules=extensions,
    install_requires=install_requires,
    cmdclass={'build_ext': my_build_ext}
)
