from hypothesis import given, strategies as st, settings
from Cython.Utils import build_hex_version
import re

@settings(max_examples=500)
@given(st.from_regex(r'^[0-9]+\.[0-9]+(\.[0-9]+)?([ab]|rc)?[0-9]*$', fullmatch=True))
def test_build_hex_version_format(version_string):
    print(f"Testing: '{version_string}'")
    result = build_hex_version(version_string)
    assert re.match(r'^0x[0-9A-F]{8}$', result)

# Run the test
test_build_hex_version_format()