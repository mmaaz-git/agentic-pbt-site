import numpy as np
from hypothesis import given, example
from hypothesis.extra import numpy as hnp


@given(hnp.arrays(dtype=np.float64, shape=(4, 4)))
@example(np.array([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]]))
def test_dead_code_bug_noncontiguous_copy(arr):
    non_contiguous = arr[::2, ::2]

    if not non_contiguous.flags.contiguous:
        m = np.matrix(non_contiguous, copy=False)

        if not m.flags.c_contiguous and not m.flags.f_contiguous:
            assert False, "Matrix from non-contiguous array is non-contiguous"

if __name__ == "__main__":
    test_dead_code_bug_noncontiguous_copy()