#!/usr/bin/env python3
"""Test reproduction for normalise_float_repr bug - additional examples"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.Utils import normalise_float_repr

# Test more edge cases
test_cases = [
    -1.670758163823954e-133,  # Very small negative
    1.114036198514633e-05,    # Small positive
    -1.0,                      # Simple negative
    -0.5,                      # Negative fraction
    -10.5,                     # Negative with integer part
    -1e-10,                    # Small negative in scientific
    -1e10,                     # Large negative
    0.0,                       # Zero
    1.0,                       # One
    123.456,                   # Simple positive
]

print("=== Testing various float values ===")
for f in test_cases:
    float_str = str(f)
    result = normalise_float_repr(float_str)
    print(f"\nInput: {float_str}")
    print(f"Output: {result}")
    try:
        converted = float(result)
        print(f"Converted back: {converted}")
        if abs(f) > 1e-300:  # Avoid division by zero for very small numbers
            error_pct = abs(converted - f) / abs(f) * 100
            if error_pct > 0.001:
                print(f"ERROR: Value changed by {error_pct:.2f}%")
    except ValueError as e:
        print(f"ERROR: Cannot convert back to float: {e}")