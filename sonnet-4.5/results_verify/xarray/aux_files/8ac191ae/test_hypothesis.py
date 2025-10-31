from hypothesis import given, strategies as st, settings
from xarray.plot.facetgrid import _nicetitle

@given(
    st.text(min_size=0, max_size=50),
    st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(min_size=0, max_size=100), st.none()),
    st.integers(min_value=1, max_value=1000),
    st.text(min_size=0, max_size=100)
)
@settings(max_examples=1000)
def test_nicetitle_length_property(coord, value, maxchar, template_base):
    template = template_base + "{coord}={value}"
    result = _nicetitle(coord, value, maxchar, template)
    print(f"Testing with coord='{coord}', value={value}, maxchar={maxchar}")
    print(f"Result: '{result}', Length: {len(result)}")
    assert len(result) <= maxchar, f"Result '{result}' has length {len(result)} but maxchar is {maxchar}"

# Run the test
if __name__ == "__main__":
    test_nicetitle_length_property()
    print("Test completed without finding any issues (if no assertion error)")