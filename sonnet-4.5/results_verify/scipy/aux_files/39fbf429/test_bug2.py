#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

import numpy as np

# Test the actual division behavior
print("Testing division by zero behavior in Python:")
print("-" * 50)

try:
    result = 1.0 / 0
    print(f"1.0 / 0 = {result}")
except ZeroDivisionError as e:
    print(f"1.0 / 0 raises ZeroDivisionError: {e}")

try:
    result = 1.0 / (1 - 1)
    print(f"1.0 / (1 - 1) = {result}")
except ZeroDivisionError as e:
    print(f"1.0 / (1 - 1) raises ZeroDivisionError: {e}")

# Test with numpy operations
print("\nTesting numpy operations:")
data = np.array([5.0])
print(f"data.size = {data.size}")
print(f"data.size - 1 = {data.size - 1}")

try:
    nbfac = data.size * 1. / (data.size - 1)
    print(f"nbfac = {nbfac}")
except ZeroDivisionError as e:
    print(f"nbfac calculation raises ZeroDivisionError: {e}")

# Test what np.std returns for single element
print(f"\nnp.std([5.0]) = {np.std(np.array([5.0]))}")
print(f"np.nanmin([5.0]) = {np.nanmin(np.array([5.0]))}")
print(f"np.nanmax([5.0]) = {np.nanmax(np.array([5.0]))}")
print(f"np.mean([5.0]) = {np.mean(np.array([5.0]))}")

# Test the actual method
from scipy.io.arff._arffread import NumericAttribute

print("\nTrying to call _basic_stats on single element:")
attr = NumericAttribute("test")
data = np.array([5.0])

try:
    result = attr._basic_stats(data)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError raised: {e}")
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")