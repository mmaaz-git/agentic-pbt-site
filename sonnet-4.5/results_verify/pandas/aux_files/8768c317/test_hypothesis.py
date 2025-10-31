import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst


@given(
    values=npst.arrays(dtype=np.int64, shape=st.integers(5, 20),
                      elements=st.integers(-100, 100))
)
def test_cumsum_should_not_mutate_input(values):
    """cumsum should not modify the input array."""
    original = values.copy()
    mask = np.zeros(len(values), dtype=bool)
    mask[len(values) // 2] = True  # Set at least one mask value to True

    result_values, result_mask = cumsum(values, mask, skipna=True)

    # This assertion fails!
    np.testing.assert_array_equal(
        values, original,
        err_msg="cumsum modified the input array!"
    )

if __name__ == "__main__":
    # Run the test
    test_cumsum_should_not_mutate_input()