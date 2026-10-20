"""Build optional native kernels; the Python implementation remains available."""

import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

flags = ["/O2", "/fp:strict"] if sys.platform == "win32" else ["-O3", "-ffp-contract=off", "-fno-fast-math"]
extension = Extension("ddc._native", ["src/ddc/_native.pyx"], optional=True, extra_compile_args=flags)
setup(ext_modules=cythonize([extension], build_dir="build/cython", compiler_directives={
    "language_level": 3, "boundscheck": False, "wraparound": False,
    "initializedcheck": False, "cdivision": True,
}))
