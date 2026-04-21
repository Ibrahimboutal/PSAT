import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class get_pybind_include:
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "psat_cpp_core",
        ["psat/cpp_core.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language="c++",
    ),
]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc", "/O2", "/std:c++14", "/openmp"],
        "unix": ["-O3", "-std=c++14", "-Wall", "-fopenmp"],
    }

    if sys.platform == "darwin":
        c_opts["unix"] += [
            "-stdlib=libc++",
            "-mmacosx-version-min=10.14",
            "-Xpreprocessor",
            "-fopenmp",
        ]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()


setup(
    name="psat",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
