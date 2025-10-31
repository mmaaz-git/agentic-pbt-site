#!/usr/bin/env python3
"""Test script to reproduce the reported bug in pandas.io.formats.format.format_percentiles"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

# First, let's try the direct reproduction cases
print("=" * 60)
print("Direct reproduction of reported failing cases:")
print("=" * 60)

try:
    from pandas.io.formats.format import format_percentiles

    # Test case 1: [0.0, 5e-324]
    print("\nTest case 1: [0.0, 5e-324]")
    result1 = format_percentiles([0.0, 5e-324])
    print(f"Result: {result1}")
    print(f"Expected from report: ['nan%', 'nan%']")

    # Test case 2: [0.0, 1.401298464324817e-45]
    print("\nTest case 2: [0.0, 1.401298464324817e-45]")
    result2 = format_percentiles([0.0, 1.401298464324817e-45])
    print(f"Result: {result2}")
    print(f"Expected from report: ['0%', '0%']")

    # Test case 3: [1e-10]
    print("\nTest case 3: [1e-10]")
    result3 = format_percentiles([1e-10])
    print(f"Result: {result3}")
    print(f"Expected from report: ['0%']")

except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternate import path...")
    from pandas.core.methods.describe import format_percentiles

    # Test case 1: [0.0, 5e-324]
    print("\nTest case 1: [0.0, 5e-324]")
    result1 = format_percentiles([0.0, 5e-324])
    print(f"Result: {result1}")
    print(f"Expected from report: ['nan%', 'nan%']")

    # Test case 2: [0.0, 1.401298464324817e-45]
    print("\nTest case 2: [0.0, 1.401298464324817e-45]")
    result2 = format_percentiles([0.0, 1.401298464324817e-45])
    print(f"Result: {result2}")
    print(f"Expected from report: ['0%', '0%']")

    # Test case 3: [1e-10]
    print("\nTest case 3: [1e-10]")
    result3 = format_percentiles([1e-10])
    print(f"Result: {result3}")
    print(f"Expected from report: ['0%']")

# Additional test cases to understand the behavior
print("\n" + "=" * 60)
print("Additional test cases:")
print("=" * 60)

# Test with normal percentile values
print("\nNormal values: [0.25, 0.5, 0.75]")
result_normal = format_percentiles([0.25, 0.5, 0.75])
print(f"Result: {result_normal}")

# Test with exact 0 and 1
print("\nExact boundaries: [0.0, 1.0]")
result_bounds = format_percentiles([0.0, 1.0])
print(f"Result: {result_bounds}")

# Test with very close values
print("\nVery close values: [0.5, 0.50001]")
result_close = format_percentiles([0.5, 0.50001])
print(f"Result: {result_close}")