#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm
import math
from hypothesis import given, settings, strategies as st

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)))
@settings(max_examples=1000)
def test_encode_decode_round_trip(values):
    encoded = llm.encode(values)
    decoded = llm.decode(encoded)
    assert len(decoded) == len(values)
    for original, recovered in zip(values, decoded):
        assert math.isclose(original, recovered, rel_tol=1e-6)

# Run the test
print("Running Hypothesis test...")
try:
    test_encode_decode_round_trip()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed with assertion error")
except Exception as e:
    print(f"Test failed: {e}")

# Test with the specific failing input
print("\nTesting with specific failing input from bug report:")
values = [4.484782386619779e-144]
encoded = llm.encode(values)
decoded = llm.decode(encoded)
print(f"Original: {values[0]}")
print(f"Decoded:  {decoded[0]}")
try:
    assert math.isclose(values[0], decoded[0], rel_tol=1e-6)
    print("Assertion passed")
except AssertionError:
    print("Assertion failed: values don't match within tolerance")