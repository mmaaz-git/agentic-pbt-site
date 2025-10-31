from pandas.io.json import ujson_dumps, ujson_loads
from hypothesis import given, settings, strategies as st


@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
    st.booleans(),
    st.none()
))
@settings(max_examples=500)
def test_ujson_roundtrip(obj):
    json_str = ujson_dumps(obj)
    recovered = ujson_loads(json_str)
    assert obj == recovered

if __name__ == "__main__":
    test_ujson_roundtrip()