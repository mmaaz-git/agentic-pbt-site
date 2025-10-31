from hypothesis import given, strategies as st
import numpy.rec


@given(st.lists(st.tuples(), min_size=1, max_size=10))
def test_fromrecords_empty_tuples(records):
    rec_arr = numpy.rec.fromrecords(records)
    assert len(rec_arr) == len(records)

if __name__ == "__main__":
    test_fromrecords_empty_tuples()