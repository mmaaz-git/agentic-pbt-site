import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.indexes.api import union_indexes
from pandas import Index


@st.composite
def index_strategy(draw):
    values = draw(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=50))
    return Index(values)


@given(index_strategy())
@settings(max_examples=500)
def test_union_indexes_idempotent(idx):
    result = union_indexes([idx, idx])
    assert result.equals(idx.unique())

# Test with the specific failing input mentioned
idx_failing = Index([0, 0])
print(f"Testing with Index([0, 0])")
result = union_indexes([idx_failing, idx_failing])
print(f"Result: {list(result)}")
print(f"Expected (unique values): {list(idx_failing.unique())}")
print(f"Test passes: {result.equals(idx_failing.unique())}")

# Run the hypothesis test
print("\nRunning hypothesis test...")
try:
    test_union_indexes_idempotent()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")