from Cython.Distutils.build_ext import build_ext
from distutils.dist import Distribution


class MockExtension:
    cython_cplus = 1


dist = Distribution()
build_ext_instance = build_ext(dist)
build_ext_instance.initialize_options()
build_ext_instance.finalize_options()

# Set command-line option to 0 (disable C++)
build_ext_instance.cython_cplus = 0

ext = MockExtension()

# Should return 0 (command-line value), but returns 1 (extension value)
result = build_ext_instance.get_extension_attr(ext, 'cython_cplus')

print(f"Command-line setting (build_ext_instance.cython_cplus): {build_ext_instance.cython_cplus}")
print(f"Extension setting (ext.cython_cplus): {ext.cython_cplus}")
print(f"Result from get_extension_attr: {result}")
print(f"Expected: 0 (command-line should override)")
print(f"Actual: {result}")
print(f"Bug confirmed: {result != 0}")