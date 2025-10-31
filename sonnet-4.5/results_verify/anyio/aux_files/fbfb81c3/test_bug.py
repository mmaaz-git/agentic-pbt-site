#!/usr/bin/env python3
"""Test to reproduce the CapacityLimiter bug"""

import sys
import math
import traceback

# Add the anyio env to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

from hypothesis import given, settings, strategies as st
from anyio._core._synchronization import CapacityLimiter

# First test: The property-based test from the bug report
print("Running property-based test...")
@given(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=10)  # Reduced for quick testing
def test_capacity_limiter_accepts_float_tokens(tokens):
    limiter = CapacityLimiter(tokens)
    assert limiter.total_tokens == tokens

try:
    test_capacity_limiter_accepts_float_tokens()
    print("Property-based test passed (unexpected!)")
except Exception as e:
    print(f"Property-based test failed as expected: {e}")
    print()

# Second test: Direct reproduction from the bug report
print("Testing direct instantiation with float 1.5...")
try:
    limiter = CapacityLimiter(1.5)
    print(f"Created limiter with 1.5 tokens successfully. total_tokens = {limiter.total_tokens}")
except Exception as e:
    print(f"Failed to create limiter with 1.5: {type(e).__name__}: {e}")
    print()

print("Testing direct instantiation with int 1...")
try:
    limiter = CapacityLimiter(1)
    print(f"Created limiter with 1 token successfully. total_tokens = {limiter.total_tokens}")

    print("Now trying to set total_tokens to 2.5...")
    try:
        limiter.total_tokens = 2.5
        print(f"Successfully set total_tokens to 2.5")
    except Exception as e:
        print(f"Failed to set total_tokens to 2.5: {type(e).__name__}: {e}")
except Exception as e:
    print(f"Failed to create limiter with 1: {type(e).__name__}: {e}")

print()
print("Testing with math.inf...")
try:
    limiter = CapacityLimiter(math.inf)
    print(f"Created limiter with math.inf successfully. total_tokens = {limiter.total_tokens}")

    print("Now trying to set total_tokens to math.inf again...")
    limiter.total_tokens = math.inf
    print(f"Successfully set total_tokens to math.inf")
except Exception as e:
    print(f"Failed with math.inf: {type(e).__name__}: {e}")