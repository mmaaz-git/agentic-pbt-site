import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

cmd.cython_directives = "boundscheck=True"
cmd.finalize_options()

print(f"Type of cython_directives after finalize_options: {type(cmd.cython_directives)}")
print(f"Value: {cmd.cython_directives}")

# This is where it would fail in build_extension
try:
    directives = dict(cmd.cython_directives)
    print("Successfully converted to dict:", directives)
except ValueError as e:
    print(f"ValueError when converting to dict: {e}")