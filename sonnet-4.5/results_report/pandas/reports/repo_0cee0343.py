#!/usr/bin/env python3
"""Minimal reproduction of pandas.cut duplicates='drop' bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd

# Test case from the bug report
x = [0.0, 0.0, 0.0, 0.0, 5e-324]

print("=" * 60)
print("Test Case 1: pd.cut with bins=2")
print("Input: x =", x)
print()

# First try without duplicates='drop'
print("Attempting pd.cut(x, bins=2):")
try:
    result = pd.cut(x, bins=2)
    print("Success! Result:", result)
except ValueError as e:
    print(f"Error: {e}")
    print()

    # Following the error message's advice
    print("Following error message advice - using duplicates='drop':")
    print("Attempting pd.cut(x, bins=2, duplicates='drop'):")
    try:
        result = pd.cut(x, bins=2, duplicates='drop')
        print("Success! Result:", result)
    except ValueError as e:
        print(f"Error: {e}")

print()
print("=" * 60)
print("Test Case 2: pd.qcut with q=2")
x2 = [0.0]*9 + [2.225073858507e-311]
print("Input: x =", x2)
print()

print("Attempting pd.qcut(x, q=2, duplicates='drop'):")
try:
    result = pd.qcut(x2, q=2, duplicates='drop')
    print("Success! Result:", result)
except ValueError as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("Test Case 3: Works with larger values")
x3 = [0.0, 0.0, 0.0, 0.0, 0.1]
print("Input: x =", x3)
print()

print("Attempting pd.cut(x, bins=2, duplicates='drop'):")
try:
    result = pd.cut(x3, bins=2, duplicates='drop')
    print("Success! Result:", result.tolist())
except ValueError as e:
    print(f"Error: {e}")