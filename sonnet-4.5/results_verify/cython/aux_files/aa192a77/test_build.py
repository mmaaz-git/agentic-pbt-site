import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from distutils.extension import Extension
from Cython.Distutils import build_ext

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

# Simulate command-line option
cmd.cython_directives = "boundscheck=True"
cmd.finalize_options()

# Create a simple extension
ext = Extension('test_module', ['test.pyx'])

# Try to build it (this will fail at line 107)
try:
    cmd.build_extension(ext)
except ValueError as e:
    print(f"Error in build_extension: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")