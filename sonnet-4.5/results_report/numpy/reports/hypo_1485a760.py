import numpy.rec
from hypothesis import given, strategies as st


@given(st.lists(st.integers(), min_size=0, max_size=30))
def test_array_handles_all_list_sizes(lst):
    result = numpy.rec.array(lst)
    assert isinstance(result, numpy.rec.recarray)

if __name__ == "__main__":
    test_array_handles_all_list_sizes()