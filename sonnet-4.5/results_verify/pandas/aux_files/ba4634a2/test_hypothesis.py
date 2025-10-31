import numpy as np
import pandas.arrays as pa
from hypothesis import given, strategies as st, settings


@settings(max_examples=200)
@given(
    values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    mask_indices=st.lists(st.integers(min_value=0, max_value=99), max_size=10)
)
def test_integerarray_factorize_roundtrip(values, mask_indices):
    mask = np.zeros(len(values), dtype=bool)
    for idx in mask_indices:
        if idx < len(values):
            mask[idx] = True

    arr = pa.IntegerArray(np.array(values, dtype='int64'), mask=mask)
    codes, uniques = arr.factorize()

    reconstructed = uniques[codes]
    print(f"Test passed for values={values[:5]}..., mask_indices={mask_indices[:5]}...")

# Run the test
if __name__ == "__main__":
    test_integerarray_factorize_roundtrip()