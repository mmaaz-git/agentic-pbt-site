#!/usr/bin/env python3
import numpy.rec as rec
from hypothesis import given, strategies as st, example
import pytest

@given(st.lists(st.tuples(st.integers(), st.floats(allow_nan=False)), max_size=10))
@example([])  # Explicitly test the empty list case
def test_array_handles_variable_length_lists(records):
    print(f"Testing with records: {records}")
    try:
        result = rec.array(records, names='a,b')
        print(f"Success: len(result)={len(result)}, len(records)={len(records)}")
        assert len(result) == len(records)
    except IndexError as e:
        print(f"IndexError with records={records}: {e}")
        if len(records) == 0:
            print("Bug confirmed: Empty list causes IndexError")
            raise
        else:
            raise

if __name__ == "__main__":
    test_array_handles_variable_length_lists()