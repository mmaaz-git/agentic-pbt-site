#!/usr/bin/env python3
"""Test to reproduce the dedup_names bug"""

from hypothesis import given, strategies as st, assume
import pandas.io.common as pd_common

# First let's run the direct reproduction
print("Testing direct reproduction with ['0', '0']...")
try:
    names = ['0', '0']
    result = pd_common.dedup_names(names, is_potential_multiindex=True)
    print(f"Result: {result}")
except AssertionError as e:
    print(f"AssertionError raised as expected")
except Exception as e:
    print(f"Unexpected exception: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Now test with the hypothesis test
@given(
    names=st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=20)
)
def test_dedup_names_multiindex_with_non_tuples_and_duplicates(names):
    assume(len(names) != len(set(names)))

    result = pd_common.dedup_names(names, is_potential_multiindex=True)
    result_list = list(result)

    assert len(result_list) == len(names)
    assert len(result_list) == len(set(result_list))

print("Running hypothesis test...")
try:
    test_dedup_names_multiindex_with_non_tuples_and_duplicates()
    print("Hypothesis test passed unexpectedly")
except Exception as e:
    print(f"Hypothesis test failed: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# Test with is_potential_multiindex=False
print("\n" + "="*50 + "\n")
print("Testing with is_potential_multiindex=False...")
names = ['0', '0']
result = pd_common.dedup_names(names, is_potential_multiindex=False)
print(f"Result with is_potential_multiindex=False: {result}")

# Test with tuples when is_potential_multiindex=True
print("\n" + "="*50 + "\n")
print("Testing with tuples when is_potential_multiindex=True...")
names = [('a',), ('a',)]
result = pd_common.dedup_names(names, is_potential_multiindex=True)
print(f"Result with tuples: {result}")

# Test with non-duplicates and strings when is_potential_multiindex=True
print("\n" + "="*50 + "\n")
print("Testing with non-duplicates and strings when is_potential_multiindex=True...")
names = ['0', '1', '2']
result = pd_common.dedup_names(names, is_potential_multiindex=True)
print(f"Result with non-duplicates: {result}")