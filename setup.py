from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup


setup(
    ext_modules=cythonize(
        [
            Extension(
                "mini_yt.lib",
                sources=["src/mini_yt/lib.pyx"],
                include_dirs=[numpy.get_include()],
                define_macros=[
                    ("NPY_TARGET_VERSION", "NPY_1_21_API_VERSION"),
                    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                ],
            )
        ],
        compiler_directives={"language_level": 3},
    ),
)
