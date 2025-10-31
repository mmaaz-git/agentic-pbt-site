import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_encode_decode_roundtrip(s):
    encoded = char.encode(s, encoding='utf-8')
    decoded = char.decode(encoded, encoding='utf-8')
    decoded_str = str(decoded) if hasattr(decoded, 'item') else decoded
    assert decoded_str == s, f"Expected {repr(s)}, got {repr(decoded_str)}"

# Run the test
if __name__ == "__main__":
    # Test with null character specifically
    def test_null_char():
        s = '\x00'
        encoded = char.encode(s, encoding='utf-8')
        decoded = char.decode(encoded, encoding='utf-8')
        decoded_str = str(decoded) if hasattr(decoded, 'item') else decoded
        assert decoded_str == s, f"Expected {repr(s)}, got {repr(decoded_str)}"

    try:
        test_null_char()
        print("Test passed with '\\x00'")
    except AssertionError as e:
        print(f"Test failed with '\\x00': {e}")

    # Run the full hypothesis test
    import pytest
    pytest.main([__file__, "-v"])