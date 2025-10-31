#!/usr/bin/env python3

import sys
import os

# Add the virtual env path to sys.path
venv_path = "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages"
sys.path.insert(0, venv_path)

# First test the hypothesis test
from hypothesis import given, strategies as st, assume

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    import llm
    assume(len(a) == len(b))
    result = llm.cosine_similarity(a, b)
    assert isinstance(result, float)

# Run hypothesis test
print("Running Hypothesis property-based test...")
try:
    test_cosine_similarity_no_crash()
    print("Hypothesis test passed - no crash found")
except Exception as e:
    print(f"Hypothesis test failed with: {e}")

# Now test the specific failing example
print("\n" + "="*50)
print("Testing specific failing example from bug report...")
print("a = [0.0, 0.0, 0.0]")
print("b = [1.0, 2.0, 3.0]")

import llm

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Test other zero vector cases
print("\n" + "="*50)
print("Testing other zero vector cases...")

test_cases = [
    ([0.0, 0.0], [0.0, 0.0]),  # Both zero
    ([1.0, 2.0], [0.0, 0.0]),  # Second zero
    ([0.0], [1.0]),            # Single element zero
]

for i, (vec_a, vec_b) in enumerate(test_cases, 1):
    print(f"\nTest case {i}: a={vec_a}, b={vec_b}")
    try:
        result = llm.cosine_similarity(vec_a, vec_b)
        print(f"  Result: {result}")
    except ZeroDivisionError as e:
        print(f"  ZeroDivisionError: {e}")
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")