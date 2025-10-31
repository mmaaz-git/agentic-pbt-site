from hypothesis import given, strategies as st
import numpy.rec as rec
import pytest

@given(st.lists(st.lists(st.integers(), min_size=2, max_size=2), min_size=0, max_size=10))
def test_array_handles_empty_input(records):
    records_tuples = [tuple(r) for r in records]
    r = rec.array(records_tuples, formats=['i4', 'i4'], names='x,y')
    assert len(r) == len(records)

if __name__ == "__main__":
    test_array_handles_empty_input()