
from setuptools import setup, Extension, find_packages

ext_modules = []
try:
    from Cython.Build import cythonize
    compile_args = ["-std=c++11", "-O3"]
    extensions = [
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
        ),
    ]
    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False},
    )
except Exception:
    ext_modules = []

setup(
    ext_modules=ext_modules,
)