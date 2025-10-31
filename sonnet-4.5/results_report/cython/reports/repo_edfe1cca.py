#!/usr/bin/env python3
"""Minimal reproduction of Cython.Distutils.build_ext crash with string directives"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

# Create a distribution and build_ext command
dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

# Set cython_directives as a string (as would happen from command line)
cmd.cython_directives = "boundscheck=True,wraparound=False"
print(f"Before finalize_options: cython_directives = {repr(cmd.cython_directives)}")
print(f"Type: {type(cmd.cython_directives)}")

# Call finalize_options (should parse string to dict, but doesn't)
cmd.finalize_options()
print(f"\nAfter finalize_options: cython_directives = {repr(cmd.cython_directives)}")
print(f"Type: {type(cmd.cython_directives)}")

# This is what build_extension does on line 107, which will crash
print("\nAttempting dict(cmd.cython_directives) as done in build_extension()...")
try:
    directives = dict(cmd.cython_directives)
    print(f"Success: directives = {directives}")
except Exception as e:
    print(f"CRASH: {type(e).__name__}: {e}")