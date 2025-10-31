#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, settings
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1)
)
@settings(max_examples=100)
def test_cosine_similarity_handles_zero_vectors(a, b):
    assume(len(a) == len(b))
    try:
        result = llm.cosine_similarity(a, b)
        # Check result is valid (between -1 and 1) or NaN
        if result == result:  # Not NaN
            assert -1.0 <= result <= 1.0, f"Result {result} out of range [-1, 1]"
        print(f"✓ Passed: a={a[:3]}{'...' if len(a) > 3 else ''}, b={b[:3]}{'...' if len(b) > 3 else ''}, result={result}")
    except ZeroDivisionError:
        # Check if zero vector was involved
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        if magnitude_a == 0 or magnitude_b == 0:
            print(f"✗ Failed with ZeroDivisionError on zero vector: a_mag={magnitude_a:.3f}, b_mag={magnitude_b:.3f}")
            raise
        else:
            print(f"✗ Unexpected ZeroDivisionError: a={a}, b={b}")
            raise

print("Running property-based tests...")
try:
    test_cosine_similarity_handles_zero_vectors()
    print("\nAll tests passed!")
except Exception as e:
    print(f"\nTest failed: {e}")