#!/usr/bin/env python3
"""Test the simple reproduction case from the bug report"""

from scipy import integrate

print("Testing: integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)")
try:
    result = integrate.tanhsinh(lambda x: 0.0, 0.0, 1.0)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()