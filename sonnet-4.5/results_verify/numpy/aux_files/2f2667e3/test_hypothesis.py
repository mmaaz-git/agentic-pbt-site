from hypothesis import given, strategies as st, settings
import numpy.rec


@settings(max_examples=500)
@given(st.lists(st.integers(), min_size=0, max_size=20))
def test_array_constructor_preserves_length(int_list):
    try:
        rec_arr = numpy.rec.array(int_list, dtype=[('value', 'i4')])
        assert len(rec_arr) == len(int_list)
    except (ValueError, TypeError):
        pass

if __name__ == "__main__":
    test_array_constructor_preserves_length()