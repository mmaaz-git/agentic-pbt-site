import numpy.rec
from hypothesis import given, strategies as st, settings


@given(st.lists(st.tuples(st.integers(), st.text(max_size=10)), min_size=1, max_size=20))
@settings(max_examples=500)
def test_fromrecords_dtype_inference(records):
    r = numpy.rec.fromrecords(records, names='num,text')

    assert len(r) == len(records)
    assert r.dtype.names == ('num', 'text')

    for i, (expected_num, expected_text) in enumerate(records):
        assert r.num[i] == expected_num
        assert r.text[i] == expected_text

if __name__ == "__main__":
    try:
        test_fromrecords_dtype_inference()
        print("All tests passed")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()