from hypothesis import given, strategies as st
from pandas.io.json._normalize import nested_to_record

@given(st.dictionaries(
    keys=st.integers(),
    values=st.one_of(
        st.text(),
        st.dictionaries(keys=st.integers(), values=st.text(), max_size=3)
    ),
    max_size=5
))
def test_nested_to_record_handles_non_string_keys(d):
    result = nested_to_record(d)
    assert isinstance(result, dict)

# Run the test
if __name__ == "__main__":
    test_nested_to_record_handles_non_string_keys()