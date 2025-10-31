import numpy as np
import numpy.rec
from hypothesis import given, strategies as st

@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_fromarrays_dtype_preserves_data(data):
    arr1 = np.array(data)
    arr2 = np.array(data)
    dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
    rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)
    assert np.array_equal(rec.a, arr1)

if __name__ == "__main__":
    test_fromarrays_dtype_preserves_data()