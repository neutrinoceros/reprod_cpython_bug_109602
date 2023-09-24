from distutils.extension import Extension

from Cython.Build import cythonize
from setuptools import setup


setup(
    ext_modules=cythonize(
        [
            Extension(
                "mini_yt.lib",
                sources=["src/mini_yt/lib.pyx"],
            )
        ],
        compiler_directives={"language_level": 3},
    ),
)
