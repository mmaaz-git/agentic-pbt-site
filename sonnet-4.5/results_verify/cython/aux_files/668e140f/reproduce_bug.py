#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

print("Creating build_ext instance...")
dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

print("Setting cython_directives as string...")
cmd.cython_directives = "boundscheck=True,wraparound=False"
print(f"cython_directives type after setting: {type(cmd.cython_directives)}")

print("Calling finalize_options...")
cmd.finalize_options()
print(f"cython_directives type after finalize_options: {type(cmd.cython_directives)}")
print(f"cython_directives value: {cmd.cython_directives}")

print("\nAttempting to convert to dict (as build_extension does)...")
try:
    directives = dict(cmd.cython_directives)
    print(f"Success! directives = {directives}")
except Exception as e:
    print(f"Failed with error: {type(e).__name__}: {e}")