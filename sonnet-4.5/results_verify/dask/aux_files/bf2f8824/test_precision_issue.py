#!/usr/bin/env python3
"""Test to understand the floating point precision issue better"""

from decimal import Decimal
from dask.diagnostics.profile_visualize import fix_bounds

# The specific failing case from the bug report
start = 6442450945.0
end = 0.0
min_span = 2147483647.9201343

print("=== Precision Analysis ===")
print(f"start = {start} (repr: {repr(start)})")
print(f"end = {end} (repr: {repr(end)})")
print(f"min_span = {min_span} (repr: {repr(min_span)})")
print()

# Use Decimal for exact arithmetic
d_start = Decimal(repr(start))
d_min_span = Decimal(repr(min_span))
d_exact_sum = d_start + d_min_span
print(f"Exact sum (Decimal): {d_exact_sum}")

# What Python actually computes
float_sum = start + min_span
print(f"Float sum: {float_sum} (repr: {repr(float_sum)})")
print()

# The actual function result
new_start, new_end = fix_bounds(start, end, min_span)
print(f"fix_bounds result: new_start={new_start}, new_end={new_end}")
actual_span = new_end - new_start
print(f"Actual span: {actual_span} (repr: {repr(actual_span)})")
print(f"min_span: {min_span} (repr: {repr(min_span)})")
print()

# The difference
print(f"Difference (min_span - actual_span): {min_span - actual_span}")
print(f"Is actual_span < min_span? {actual_span < min_span}")

# Let's check the floating point representation details
import struct
def float_to_hex(f):
    return struct.unpack('>Q', struct.pack('>d', f))[0]

print()
print("=== Binary representation (as hex) ===")
print(f"min_span hex: 0x{float_to_hex(min_span):016x}")
print(f"actual_span hex: 0x{float_to_hex(actual_span):016x}")
print(f"Bits differ: {float_to_hex(min_span) != float_to_hex(actual_span)}")

# Check if this is just a rounding issue at the ULP level
import math
ulp_min_span = math.ulp(min_span)
ulp_actual_span = math.ulp(actual_span)
print()
print(f"ULP of min_span: {ulp_min_span}")
print(f"ULP of actual_span: {ulp_actual_span}")
print(f"Difference in ULPs: {(min_span - actual_span) / ulp_min_span}")