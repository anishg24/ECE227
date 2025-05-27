# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fast_ic",
        sources=["src/independent_cascade.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="fast_ic",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    author="You",
    description="Blazing-fast Independent Cascade via C++/pybind11",
)
