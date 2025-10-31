#!/usr/bin/env python3
"""Test script to reproduce the reported bug in parse_datetime_format_str"""

import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.core.interchange.from_dataframe import parse_datetime_format_str

# Test with the specific failing input from the bug report
days = 4_735_838_584_154_958_556

format_str = "tdD"
data = np.array([days], dtype=np.int64)

print("=" * 60)
print("REPRODUCING BUG WITH SPECIFIC VALUE")
print("=" * 60)
print(f"Input days: {days}")
print(f"Input days in scientific notation: {days:.2e}")

result = parse_datetime_format_str(format_str, data)
print(f"Result: {result}")
print(f"Result dtype: {result.dtype}")

result_as_int = result.view('int64')[0]
print(f"Result as int64: {result_as_int}")
print(f"Is negative? {result_as_int < 0}")

# Calculate expected seconds
expected_seconds_uint64 = np.uint64(days) * np.uint64(24 * 60 * 60)
print(f"\nExpected seconds (as uint64): {expected_seconds_uint64}")
print(f"Max int64: {2**63 - 1}")
print(f"Overflow? {expected_seconds_uint64 > 2**63 - 1}")

# Test with boundary value
print("\n" + "=" * 60)
print("TESTING BOUNDARY VALUE")
print("=" * 60)
max_days_no_overflow = (2**63 - 1) // (24 * 60 * 60)
print(f"Max days without overflow: {max_days_no_overflow}")
print(f"Max days in years: {max_days_no_overflow / 365.25:.0f}")

# Test just below boundary
test_days = max_days_no_overflow
data_below = np.array([test_days], dtype=np.int64)
result_below = parse_datetime_format_str("tdD", data_below)
print(f"\nTesting with max safe days ({test_days}):")
print(f"Result: {result_below}")
print(f"Result as int64: {result_below.view('int64')[0]}")

# Test just above boundary
test_days_over = max_days_no_overflow + 1
data_above = np.array([test_days_over], dtype=np.int64)
result_above = parse_datetime_format_str("tdD", data_above)
print(f"\nTesting with max safe days + 1 ({test_days_over}):")
print(f"Result: {result_above}")
print(f"Result as int64: {result_above.view('int64')[0]}")
print(f"Is negative? {result_above.view('int64')[0] < 0}")

# Test with small positive value to confirm normal operation
print("\n" + "=" * 60)
print("TESTING NORMAL OPERATION")
print("=" * 60)
normal_days = 365
data_normal = np.array([normal_days], dtype=np.int64)
result_normal = parse_datetime_format_str("tdD", data_normal)
print(f"Testing with normal days ({normal_days}):")
print(f"Result: {result_normal}")
print(f"Result as int64: {result_normal.view('int64')[0]}")

# Verify the calculation logic
print("\n" + "=" * 60)
print("VERIFYING OVERFLOW CALCULATION")
print("=" * 60)
print(f"Days from bug report: {4_735_838_584_154_958_556}")
uint_days = np.uint64(4_735_838_584_154_958_556)
seconds_per_day = np.uint64(24 * 60 * 60)
product = uint_days * seconds_per_day
print(f"As uint64 multiplication result: {product}")
print(f"When cast to int64: {np.int64(product)}")
print(f"This explains the negative value!")