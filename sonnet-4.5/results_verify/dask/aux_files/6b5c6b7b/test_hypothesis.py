#!/usr/bin/env python3
from hypothesis import given
import hypothesis.strategies as st
from dask.utils import key_split


@given(st.binary(min_size=1, max_size=50))
def test_key_split_with_bytes(b):
    result = key_split(b)
    assert isinstance(result, str)


if __name__ == "__main__":
    # Run the test
    test_key_split_with_bytes()