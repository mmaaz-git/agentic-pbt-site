#!/usr/bin/env python3
"""Minimal reproduction of ujson_dumps precision bug."""

from pandas.io.json import ujson_dumps, ujson_loads
import json
import math

# The problematic value - very close to max float
value = 1.7976931345e+308

print("Testing JSON round-trip with value:", value)
print("Max float value:", 1.7976931348623157e+308)
print()

# Test with standard library json
print("=== Standard Library json ===")
stdlib_serialized = json.dumps(value)
stdlib_result = json.loads(stdlib_serialized)
print(f"Original:    {value}")
print(f"Serialized:  {stdlib_serialized}")
print(f"Deserialized: {stdlib_result}")
print(f"Is finite:    {math.isfinite(stdlib_result)}")
print(f"Matches original: {stdlib_result == value}")
print()

# Test with ujson default precision (10)
print("=== ujson with default precision (10) ===")
ujson_serialized = ujson_dumps(value)
ujson_result = ujson_loads(ujson_serialized)
print(f"Original:    {value}")
print(f"Serialized:  {ujson_serialized}")
print(f"Deserialized: {ujson_result}")
print(f"Is finite:    {math.isfinite(ujson_result)}")
print(f"Matches original: {ujson_result == value}")
print()

# Test with ujson precision 15
print("=== ujson with precision 15 ===")
ujson15_serialized = ujson_dumps(value, double_precision=15)
ujson15_result = ujson_loads(ujson15_serialized)
print(f"Original:    {value}")
print(f"Serialized:  {ujson15_serialized}")
print(f"Deserialized: {ujson15_result}")
print(f"Is finite:    {math.isfinite(ujson15_result)}")
print(f"Matches original: {ujson15_result == value}")
print()

# Demonstrate the bug
print("=== BUG DEMONSTRATION ===")
assert math.isfinite(value), "Original value is finite"
assert math.isfinite(stdlib_result), "stdlib preserves finiteness"
assert not math.isfinite(ujson_result), "ujson default turns finite to infinity!"
print("BUG CONFIRMED: ujson with default precision turns finite value into infinity")