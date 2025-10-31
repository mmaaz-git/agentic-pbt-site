#!/usr/bin/env python3
"""Test the reported bug in scipy.signal.deconvolve"""

import numpy as np
import scipy.signal
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

print("Testing scipy.signal.deconvolve with zero leading coefficient")
print("=" * 60)

# Test case from bug report
signal = np.array([1.0, 2.0])
divisor = np.array([0.0, 1.0])

print(f"Signal: {signal}")
print(f"Divisor: {divisor}")
print()

try:
    quotient, remainder = scipy.signal.deconvolve(signal, divisor)
    print(f"Success! Quotient: {quotient}, Remainder: {remainder}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing with the failing input from hypothesis test:")
signal2 = np.array([0.0, 0.0])
divisor2 = np.array([0.0, 1.0])

print(f"Signal: {signal2}")
print(f"Divisor: {divisor2}")
print()

try:
    quotient, remainder = scipy.signal.deconvolve(signal2, divisor2)
    print(f"Success! Quotient: {quotient}, Remainder: {remainder}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing numpy.polydiv with same inputs for comparison:")
print()

# Test numpy.polydiv with the same inputs
print("numpy.polydiv([1.0, 2.0], [0.0, 1.0]):")
try:
    q, r = np.polydiv([1.0, 2.0], [0.0, 1.0])
    print(f"Quotient: {q}, Remainder: {r}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nnumpy.polydiv([0.0, 0.0], [0.0, 1.0]):")
try:
    q, r = np.polydiv([0.0, 0.0], [0.0, 1.0])
    print(f"Quotient: {q}, Remainder: {r}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")