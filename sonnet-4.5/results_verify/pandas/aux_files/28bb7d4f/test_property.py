from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads

json_values = st.recursive(
    st.none() | st.booleans() | st.floats(allow_nan=False, allow_infinity=False) | st.integers() | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=20
)

@settings(max_examples=1000)
@given(json_values)
def test_ujson_roundtrip(obj):
    serialized = ujson_dumps(obj)
    deserialized = ujson_loads(serialized)
    assert deserialized == obj

if __name__ == "__main__":
    test_ujson_roundtrip()