#!/usr/bin/env python3
"""Test script to reproduce the reported bug in dask.sizeof"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

print("Testing dask.sizeof dict non-determinism bug")
print("=" * 50)

# First, let's run the hypothesis test
from hypothesis import given, settings, strategies as st
from dask.sizeof import sizeof

@given(st.dictionaries(st.text(), st.integers()))
@settings(max_examples=500)
def test_sizeof_dict_formula(d):
    expected = (
        sys.getsizeof(d)
        + sizeof(list(d.keys()))
        + sizeof(list(d.values()))
        - 2 * sizeof(list())
    )
    assert sizeof(d) == expected

print("\n1. Running hypothesis test...")
try:
    test_sizeof_dict_formula()
    print("Hypothesis test passed (no failures found)")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Now let's reproduce the specific failing input
print("\n2. Testing specific failing input from bug report...")
d_failing = {'': 0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '00': 0, '000': 0, '0000': 0, '00000': 0, '000000': 0}
print(f"Dictionary size: {len(d_failing)} items")

expected = (
    sys.getsizeof(d_failing)
    + sizeof(list(d_failing.keys()))
    + sizeof(list(d_failing.values()))
    - 2 * sizeof(list())
)
actual = sizeof(d_failing)
print(f"Expected: {expected}")
print(f"Actual: {actual}")
print(f"Match: {expected == actual}")

# Now test the non-determinism
print("\n3. Testing non-determinism with dict of 11 items...")
d = {str(i): i for i in range(11)}
print(f"Dictionary: {d}")

results = [sizeof(d) for _ in range(20)]

print(f"\nResults from 20 calls: {results}")
print(f"Unique values: {sorted(set(results))}")
print(f"Range: {min(results)} to {max(results)}")
print(f"Non-deterministic: {len(set(results)) > 1}")

# Test with smaller dict (10 items - should be deterministic)
print("\n4. Testing with dict of 10 items (should be deterministic)...")
d_small = {str(i): i for i in range(10)}
results_small = [sizeof(d_small) for _ in range(20)]
print(f"Results from 20 calls: {results_small}")
print(f"Unique values: {sorted(set(results_small))}")
print(f"Deterministic: {len(set(results_small)) == 1}")

# Test with larger dict (20 items)
print("\n5. Testing with dict of 20 items...")
d_large = {str(i): i for i in range(20)}
results_large = [sizeof(d_large) for _ in range(20)]
print(f"Results from 20 calls (first 10): {results_large[:10]}")
print(f"Unique values: {sorted(set(results_large))}")
print(f"Range: {min(results_large)} to {max(results_large)}")
print(f"Non-deterministic: {len(set(results_large)) > 1}")