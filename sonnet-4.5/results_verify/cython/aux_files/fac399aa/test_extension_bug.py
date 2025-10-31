#!/usr/bin/env python3
"""Test script to reproduce the Cython.Distutils.Extension parameter loss bug"""

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Distutils import Extension

print("=" * 60)
print("Testing Cython.Distutils.Extension parameter handling")
print("=" * 60)

# Test 1: Basic reproduction from bug report
print("\nTest 1: Basic reproduction - mixing pyrex_include_dirs and cython_gdb")
ext1 = Extension(
    "mymodule",
    ["mymodule.pyx"],
    pyrex_include_dirs=["/legacy/path"],
    cython_gdb=True
)
print(f"Expected cython_gdb: True")
print(f"Actual cython_gdb: {ext1.cython_gdb}")
assert ext1.cython_gdb == True, f"FAILURE: cython_gdb should be True but got {ext1.cython_gdb}"

# Test 2: pyrex parameter without cython parameter
print("\nTest 2: Only pyrex parameter, no cython parameter")
ext2 = Extension(
    "module2",
    ["module2.pyx"],
    pyrex_include_dirs=["/some/path"]
)
print(f"pyrex_include_dirs converted to cython_include_dirs: {ext2.cython_include_dirs}")
print(f"cython_gdb (default): {ext2.cython_gdb}")

# Test 3: Only cython parameter, no pyrex parameter
print("\nTest 3: Only cython parameter, no pyrex parameter")
ext3 = Extension(
    "module3",
    ["module3.pyx"],
    cython_gdb=True
)
print(f"cython_gdb: {ext3.cython_gdb}")
assert ext3.cython_gdb == True, "FAILURE: cython_gdb should be True"

# Test 4: Multiple cython parameters with one pyrex parameter
print("\nTest 4: Multiple cython params with one pyrex param")
ext4 = Extension(
    "module4",
    ["module4.pyx"],
    pyrex_directives={'language_level': 3},
    cython_gdb=True,
    cython_create_listing=True,
    cython_line_directives=True
)
print(f"cython_directives: {ext4.cython_directives}")
print(f"cython_gdb: {ext4.cython_gdb}")
print(f"cython_create_listing: {ext4.cython_create_listing}")
print(f"cython_line_directives: {ext4.cython_line_directives}")

# Test 5: Check all cython_* parameters with pyrex present
print("\nTest 5: All cython_* params with pyrex_include_dirs present")
ext5 = Extension(
    "module5",
    ["module5.pyx"],
    pyrex_include_dirs=["/path"],
    cython_include_dirs=["/other/path"],
    cython_directives={'embedsignature': True},
    cython_create_listing=True,
    cython_line_directives=True,
    cython_cplus=True,
    cython_c_in_temp=True,
    cython_gen_pxi=True,
    cython_gdb=True,
    cython_compile_time_env={'TEST': 1}
)
print(f"cython_include_dirs: {ext5.cython_include_dirs}")
print(f"cython_directives: {ext5.cython_directives}")
print(f"cython_create_listing: {ext5.cython_create_listing}")
print(f"cython_line_directives: {ext5.cython_line_directives}")
print(f"cython_cplus: {ext5.cython_cplus}")
print(f"cython_c_in_temp: {ext5.cython_c_in_temp}")
print(f"cython_gen_pxi: {ext5.cython_gen_pxi}")
print(f"cython_gdb: {ext5.cython_gdb}")
print(f"cython_compile_time_env: {ext5.cython_compile_time_env}")

print("\n" + "=" * 60)
print("Test completed")