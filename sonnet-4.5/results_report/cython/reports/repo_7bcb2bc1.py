from distutils.dist import Distribution
from Cython.Distutils import build_ext, Extension

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()
cmd.finalize_options()

# Set command line option to False (explicitly disabling)
cmd.cython_gdb = False

ext = Extension("test_module", ["test.pyx"])
# Extension has this enabled
ext.cython_gdb = True

result = cmd.get_extension_attr(ext, 'cython_gdb')

print(f"Command line: cython_gdb = {cmd.cython_gdb}")
print(f"Extension: cython_gdb = {ext.cython_gdb}")
print(f"Result: {result}")
print(f"Expected: {cmd.cython_gdb} (from command line)")
print()

if result != cmd.cython_gdb:
    print(f"ERROR: Expected {cmd.cython_gdb} (from command), got {result}")
    print("This is a bug - command line value should take precedence!")
else:
    print("SUCCESS: Command line value correctly takes precedence")