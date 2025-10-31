from hypothesis import given, strategies as st, assume
import dask.utils
import pytest

@given(st.text())
def test_parse_bytes_rejects_whitespace_only(s):
    assume(s.strip() == '')
    assume(s != '')

    with pytest.raises(ValueError):
        dask.utils.parse_bytes(s)

# Test with specific examples
whitespace_examples = ['\r', '\n', '\t', ' ', '  ', '\r\n', '\t\t', '   ']

for example in whitespace_examples:
    print(f"Testing with {repr(example)}...")
    try:
        result = dask.utils.parse_bytes(example)
        print(f"  Result: {result} (expected ValueError)")
    except ValueError as e:
        print(f"  Raised ValueError: {e}")

# Run the hypothesis test
print("\nRunning Hypothesis test...")
test_parse_bytes_rejects_whitespace_only()