#!/usr/bin/env python3
"""Test how JSON handles special float values."""

import json
import math

print("Testing JSON serialization of special float values:")
print("=" * 60)

# Test standard JSON encoder
values = {
    "positive_infinity": float('inf'),
    "negative_infinity": float('-inf'),
    "nan": float('nan'),
    "regular_float": 1.5,
}

print("\n1. Standard JSON encoder (default):")
for name, value in values.items():
    try:
        result = json.dumps(value)
        print(f"  json.dumps({name}) = {result}")
    except Exception as e:
        print(f"  json.dumps({name}) raised {type(e).__name__}: {e}")

print("\n2. Standard JSON encoder with allow_nan=False:")
for name, value in values.items():
    try:
        result = json.dumps(value, allow_nan=False)
        print(f"  json.dumps({name}, allow_nan=False) = {result}")
    except Exception as e:
        print(f"  json.dumps({name}, allow_nan=False) raised {type(e).__name__}: {e}")

print("\n3. Standard JSON encoder with allow_nan=True (default):")
for name, value in values.items():
    try:
        result = json.dumps(value, allow_nan=True)
        print(f"  json.dumps({name}, allow_nan=True) = {result}")
    except Exception as e:
        print(f"  json.dumps({name}, allow_nan=True) raised {type(e).__name__}: {e}")

print("\n4. Checking JSON spec compliance:")
print("  According to JSON spec (RFC 7159), NaN and Infinity are NOT valid JSON values.")
print("  Python's json module allows them by default but it's non-standard.")
print("  JavaScript JSON.stringify() outputs: null for NaN, null for Infinity")