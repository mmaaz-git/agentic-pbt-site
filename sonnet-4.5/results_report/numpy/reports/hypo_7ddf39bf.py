import numpy as np
import numpy.rec
from hypothesis import given, strategies as st, settings


@given(st.lists(st.sampled_from(['i4', 'f8', 'U10']), min_size=0, max_size=5))
@settings(max_examples=100)
def test_pprint_handles_any_number_of_fields(formats):
    if len(formats) == 0:
        dtype = np.dtype([])
    else:
        names = [f'f{i}' for i in range(len(formats))]
        dtype = np.dtype(list(zip(names, formats)))

    arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
    rec = arr[0]

    result = rec.pprint()
    assert isinstance(result, str)

if __name__ == "__main__":
    test_pprint_handles_any_number_of_fields()