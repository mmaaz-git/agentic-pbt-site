#!/usr/bin/env python3
"""Property-based test that discovers the numpy.ma clump_masked bug."""

import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_arrays_1d(draw, dtype=np.int64, max_size=50):
    size = draw(st.integers(min_value=0, max_value=max_size))
    data = draw(npst.arrays(dtype=dtype, shape=(size,),
                           elements=st.integers(min_value=-1000, max_value=1000)))
    mask = draw(npst.arrays(dtype=bool, shape=(size,)))
    return ma.array(data, mask=mask)

@given(masked_arrays_1d())
@settings(max_examples=500)
def test_clump_masked_partition(arr):
    clumps = ma.clump_masked(arr)
    mask = ma.getmaskarray(arr)
    covered_indices = set()
    for clump in clumps:
        for i in range(clump.start, clump.stop):
            assert mask[i]
            covered_indices.add(i)
    for i in range(len(arr)):
        if mask[i]:
            assert i in covered_indices

if __name__ == "__main__":
    test_clump_masked_partition()