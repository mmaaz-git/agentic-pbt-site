from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads

@settings(max_examples=500)
@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_ujson_dict_roundtrip(d):
    serialized = ujson_dumps(d)
    deserialized = ujson_loads(serialized)
    assert deserialized == d

if __name__ == "__main__":
    test_ujson_dict_roundtrip()