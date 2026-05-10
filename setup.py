
from setuptools import setup, Extension
from Cython.Build import cythonize

compile_args = ["-std=c++11", "-O3", "-march=native"]

ext_modules = [
    Extension(
        "luna.screening.engine.disruptor",
        ["luna/screening/engine/disruptor.pyx"],
        language="c++",
        extra_compile_args=compile_args,
    ),
    Extension(
        "luna.screening.engine.memory",
        ["luna/screening/engine/memory.pyx"],
        language="c++",
        extra_compile_args=compile_args,
    ),
    Extension(
        "luna.screening.engine.transformer",
        ["luna/screening/engine/transformer.pyx"],
        language="c++",
        extra_compile_args=compile_args,
    )
]

setup(
    name="luna-engine",
    ext_modules=cythonize(
        ext_modules, 
        compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False}
    )
)