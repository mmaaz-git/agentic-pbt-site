from hypothesis import given, strategies as st, settings
import numpy.rec
import pytest


@settings(max_examples=500)
@given(
    st.integers(1, 10).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(st.integers(-100, 100), min_size=n, max_size=n),
            st.integers(0, n-1)
        )
    )
)
def test_recarray_field_by_invalid_index(args):
    n, arr, idx = args

    rec_arr = numpy.rec.fromarrays([arr], names='x')

    field = rec_arr.field(idx)
    assert list(field) == arr

    with pytest.raises((IndexError, KeyError)):
        rec_arr.field(n)

if __name__ == "__main__":
    # Test with the specific failing case mentioned
    test_args = (2, [0, 0], 1)
    print(f"Testing with args={test_args}")

    # Actually test the issue - this has 2 elements but only 1 field
    n, arr, idx = test_args
    rec_arr = numpy.rec.fromarrays([arr], names='x')
    print(f"Created recarray with {len(rec_arr.dtype.names)} field(s)")
    print(f"Fields: {rec_arr.dtype.names}")

    # This should work (field index 0)
    field = rec_arr.field(0)
    print(f"rec_arr.field(0) works: {list(field)}")

    # This should raise an error (field index 1 doesn't exist)
    try:
        rec_arr.field(1)
        print("rec_arr.field(1) did not raise an error!")
    except Exception as e:
        print(f"rec_arr.field(1) raised: {type(e).__name__}: {e}")