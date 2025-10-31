from distutils.dist import Distribution
from Cython.Distutils import build_ext, Extension

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()
cmd.finalize_options()

cmd.cython_gdb = False

ext = Extension("test_module", ["test.pyx"])
ext.cython_gdb = True

result = cmd.get_extension_attr(ext, 'cython_gdb')

print(f"Command line: cython_gdb = {cmd.cython_gdb}")
print(f"Extension: cython_gdb = {ext.cython_gdb}")
print(f"Result: {result}")

assert result == False, f"Expected False (from command), got {result}"