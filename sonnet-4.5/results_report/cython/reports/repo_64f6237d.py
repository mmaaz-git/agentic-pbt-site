import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

# Create a Distribution and build_ext instance
dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

# Set cython_directives as it would be from command line
cmd.cython_directives = "boundscheck=True"
print(f"Before finalize_options: cython_directives = {repr(cmd.cython_directives)}")
print(f"Type: {type(cmd.cython_directives)}")

# Call finalize_options
cmd.finalize_options()
print(f"\nAfter finalize_options: cython_directives = {repr(cmd.cython_directives)}")
print(f"Type: {type(cmd.cython_directives)}")

# Try to convert to dict as build_extension does at line 107
print("\nAttempting dict(cmd.cython_directives) as done in build_extension line 107:")
try:
    directives = dict(cmd.cython_directives)
    print(f"Success: {directives}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")