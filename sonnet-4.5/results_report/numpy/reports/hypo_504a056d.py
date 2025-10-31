from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.tuples(st.integers(), st.floats(allow_nan=False, allow_infinity=False)), min_size=0, max_size=10))
def test_fromrecords_length_invariant(records):
    """
    Test that np.rec.fromrecords correctly creates a record array
    with the same length as the input list.
    """
    rec = np.rec.fromrecords(records, names='x,y')
    assert len(rec) == len(records)

if __name__ == "__main__":
    # Run the test
    test_fromrecords_length_invariant()
    print("All tests passed!")