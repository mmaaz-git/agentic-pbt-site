#!/usr/bin/env python3
"""Test the reported numpy.matrixlib bug"""

import numpy as np
from numpy import matrix

print("Testing the basic reproduction case...")
m = matrix(";")
print(f"Shape: {m.shape}")
print(f"Size: {m.size}")
print(f"Matrix object: {m}")
print(f"Matrix type: {type(m)}")

assert m.shape == (2, 0)
assert m.size == 0
print("Basic test passed - confirms bug exists\n")

print("Testing other mentioned cases...")

try:
    m_empty = matrix("")
    print(f"Empty string result - Shape: {m_empty.shape}, Size: {m_empty.size}")
except Exception as e:
    print(f"Empty string raised: {type(e).__name__}: {e}")

try:
    m_space = matrix(" ")
    print(f"Single space result - Shape: {m_space.shape}, Size: {m_space.size}")
except Exception as e:
    print(f"Single space raised: {type(e).__name__}: {e}")

try:
    m_double = matrix(";;")
    print(f"Double semicolon result - Shape: {m_double.shape}, Size: {m_double.size}")
except Exception as e:
    print(f"Double semicolon raised: {type(e).__name__}: {e}")