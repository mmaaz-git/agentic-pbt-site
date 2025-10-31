from Cython.Distutils.build_ext import build_ext
from Cython.Distutils import Extension
from distutils.dist import Distribution

dist = Distribution()
build_ext_instance = build_ext(dist)
build_ext_instance.initialize_options()
build_ext_instance.finalize_options()

build_ext_instance.cython_compile_time_env = {}

ext = Extension(
    "test_module",
    ["test.pyx"],
    cython_compile_time_env={"DEBUG": True}
)

result = build_ext_instance.get_extension_attr(ext, 'cython_compile_time_env', default=None)

print(f"Expected: {{}}, Actual: {result}")