#!/usr/bin/env python3
"""Test to understand the context where removespaces is used"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')
import numpy.f2py.crackfortran as cf

# Test what markinnerspaces does
test_inputs = [
    "a 'b c' d",
    "integer :: x = 5",
    "real*8 :: array(10)",
    "character*80 :: str",
    "\n  integer :: x\n",
    "\t\treal :: y\t\t"
]

print("Testing markinnerspaces:")
for inp in test_inputs:
    marked = cf.markinnerspaces(inp)
    print(f"  Input:  {repr(inp)}")
    print(f"  Marked: {repr(marked)}")
    removed = cf.removespaces(marked)
    print(f"  Removed: {repr(removed)}")
    print()

# Now test actual Fortran-like declarations
fortran_examples = [
    "integer :: x",
    "real*8 :: array ( 10 )",
    "character * 80 :: str",
    "integer, dimension(10) :: arr",
    "real, parameter :: pi = 3.14159"
]

print("\nTesting Fortran-like declarations:")
for decl in fortran_examples:
    result = cf.removespaces(cf.markinnerspaces(decl))
    print(f"  {decl:<35} -> {result}")