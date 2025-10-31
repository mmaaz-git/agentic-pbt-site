from hypothesis import given, strategies as st, settings
from starlette.datastructures import URL


@given(port=st.integers(min_value=1, max_value=65535))
@settings(max_examples=100)
def test_url_replace_port_with_empty_hostname(port):
    url = URL("http://@/path")
    result = url.replace(port=port)
    assert isinstance(result, URL)

# Run the test
test_url_replace_port_with_empty_hostname()