from Cython.Distutils.build_ext import build_ext
from distutils.dist import Distribution


class MockExtension:
    cython_cplus = 1


dist = Distribution()
build_ext_instance = build_ext(dist)
build_ext_instance.initialize_options()
build_ext_instance.finalize_options()

build_ext_instance.cython_cplus = 0

ext = MockExtension()

result = build_ext_instance.get_extension_attr(ext, 'cython_cplus')

print(f"Expected: 0, Actual: {result}")