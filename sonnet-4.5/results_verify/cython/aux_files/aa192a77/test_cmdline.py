"""Test how command-line options work"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

# Simulate how it would work from command line
dist = Distribution()

# Test if Distribution can parse command-line options
# Normally, setup.py would be called like:
# python setup.py build_ext --cython-directives="boundscheck=True,wraparound=False"

# Let's see how the option is defined:
print("User options defined:")
for option in build_ext.user_options:
    if 'cython-directives' in str(option):
        print(f"  {option}")

# Check what type the option expects
cmd = build_ext(dist)
cmd.initialize_options()
print(f"\nInitial value of cython_directives: {cmd.cython_directives}")
print(f"Type: {type(cmd.cython_directives)}")

# Set as string (as would come from command line)
cmd.cython_directives = "boundscheck=True,wraparound=False"
print(f"\nAfter setting string value: {cmd.cython_directives}")
print(f"Type: {type(cmd.cython_directives)}")

# Call finalize_options
cmd.finalize_options()
print(f"\nAfter finalize_options: {cmd.cython_directives}")
print(f"Type: {type(cmd.cython_directives)}")