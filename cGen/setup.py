from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from setuptools.command.build_ext import build_ext

ext_modules = [Extension(name="cGen",
                        sources=["cGen.pyx"],
                        language='c',
                        extra_compile_args=['-O3'
                                        # , '-fopenmp'
                                           ],
                        # extra_link_args=['-fopenmp'],
                        include_dirs=[numpy.get_include()],
                       )]
setup(
    name="cGen",
    ext_modules=cythonize(ext_modules),
)

# setup(
#     ext_modules=cythonize("SPBM.pyx",
#         aliases={'extra_compile_args': ['-O3']}),
#     include_dirs=[numpy.get_include()],
# )


# setup(
    # ext_modules=cythonize("SPBM.pyx",annotate=True),
#     ext_modules=cythonize("SPBM.pyx"),
#     include_dirs=[numpy.get_include()],
# )
