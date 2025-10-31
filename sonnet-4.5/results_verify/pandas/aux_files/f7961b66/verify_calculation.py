#!/usr/bin/env python3
"""Manually verify the calculation to understand what's happening"""

import numpy as np
from pandas.core.indexers import length_of_indexer

def manual_calculation(target_len, start, stop, step):
    """Replicate the logic from length_of_indexer for slices"""
    if start is None:
        start = 0
    elif start < 0:
        start += target_len

    if stop is None or stop > target_len:
        stop = target_len
    elif stop < 0:
        stop += target_len

    if step is None:
        step = 1
    elif step < 0:
        start, stop = stop + 1, start + 1
        step = -step

    # This is the formula from the source code
    result = (stop - start + step - 1) // step
    return result

# Test the failing case
print("Testing slice(1, None, None) on np.arange(0):")
print()

target_len = 0
start = 1
stop = None
step = None

print(f"Input: target_len={target_len}, slice({start}, {stop}, {step})")
print()

# Manual calculation step by step
print("Step-by-step calculation:")
print(f"1. start = {start} (unchanged since start >= 0)")

if stop is None:
    stop_calc = target_len
    print(f"2. stop = None -> {stop_calc} (set to target_len)")
else:
    stop_calc = stop
    print(f"2. stop = {stop}")

if step is None:
    step_calc = 1
    print(f"3. step = None -> {step_calc}")
else:
    step_calc = step
    print(f"3. step = {step}")

print(f"4. Formula: (stop - start + step - 1) // step")
print(f"   = ({stop_calc} - {start} + {step_calc} - 1) // {step_calc}")
print(f"   = ({stop_calc - start + step_calc - 1}) // {step_calc}")
print(f"   = {(stop_calc - start + step_calc - 1) // step_calc}")

manual_result = manual_calculation(target_len, start, stop, step)
computed_result = length_of_indexer(slice(start, stop, step), np.arange(target_len))
actual_result = len(np.arange(target_len)[slice(start, stop, step)])

print()
print("Results:")
print(f"Manual calculation: {manual_result}")
print(f"length_of_indexer: {computed_result}")
print(f"Actual length: {actual_result}")
print()

print("=" * 60)
print("Analysis:")
print("=" * 60)
print("The formula (stop - start + step - 1) // step produces negative values")
print("when start > stop, which happens when:")
print("  1. The slice starts beyond the array bounds")
print("  2. The slice has start > stop (backwards slice with positive step)")
print()
print("In Python/NumPy, such slices produce empty results (length 0),")
print("but the formula returns negative values instead.")
print()
print("The mathematical concept of 'length' is always non-negative.")
print("A slice that selects no elements has length 0, not negative.")