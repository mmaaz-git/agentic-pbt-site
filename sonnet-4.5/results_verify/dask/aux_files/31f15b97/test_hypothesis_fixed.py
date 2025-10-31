from hypothesis import given, strategies as st, assume, settings, HealthCheck
import dask.utils
import pytest

@given(st.sampled_from(['\r', '\n', '\t', ' ', '  ', '\r\n', '\t\t', '   ', '\v', '\f']))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_parse_bytes_rejects_whitespace_only(s):
    """Test that parse_bytes should reject whitespace-only strings"""
    print(f"Testing with {repr(s)}")
    with pytest.raises(ValueError):
        result = dask.utils.parse_bytes(s)
        print(f"  Unexpectedly got result: {result}")

# Run the test
print("Running Hypothesis test with whitespace strings...")
try:
    test_parse_bytes_rejects_whitespace_only()
    print("Test passed - all whitespace strings raised ValueError")
except AssertionError as e:
    print(f"Test failed - whitespace strings did not raise ValueError")
except Exception as e:
    print(f"Test error: {e}")