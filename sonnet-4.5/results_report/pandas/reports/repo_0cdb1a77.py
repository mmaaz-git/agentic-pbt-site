#!/usr/bin/env python3
"""Minimal reproduction of InfinityType comparison inconsistency bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas.util.version as v

print("Testing InfinityType:")
print("-" * 40)
inf = v.Infinity
print(f"Infinity == Infinity: {inf == inf}")
print(f"Infinity <= Infinity: {inf <= inf}")
print(f"Infinity >= Infinity: {inf >= inf}")

print("\nTesting NegativeInfinityType:")
print("-" * 40)
neginf = v.NegativeInfinity
print(f"NegativeInfinity == NegativeInfinity: {neginf == neginf}")
print(f"NegativeInfinity <= NegativeInfinity: {neginf <= neginf}")
print(f"NegativeInfinity >= NegativeInfinity: {neginf >= neginf}")

print("\nMathematical Property Violation:")
print("-" * 40)
print("Expected: If x == x is True, then x <= x and x >= x must both be True")
print(f"Infinity violates this: {inf == inf} but {inf <= inf}")
print(f"NegativeInfinity violates this: {neginf == neginf} but {neginf >= neginf}")

print("\nComparison with Python's built-in infinity:")
print("-" * 40)
py_inf = float('inf')
print(f"float('inf') == float('inf'): {py_inf == py_inf}")
print(f"float('inf') <= float('inf'): {py_inf <= py_inf}")
print(f"float('inf') >= float('inf'): {py_inf >= py_inf}")