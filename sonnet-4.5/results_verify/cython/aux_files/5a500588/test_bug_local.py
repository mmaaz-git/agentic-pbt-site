from hypothesis import given, strategies as st
from truncated_string_io import TruncatedStringIO, StringTruncated

# Property-based test from bug report
@given(st.text(min_size=1))
def test_maxlen_zero_should_reject_all_data(data):
    sio = TruncatedStringIO(maxlen=0)
    try:
        sio.write(data)
    except StringTruncated:
        pass
    assert len(sio.getvalue()) == 0, f"maxlen=0 should not store any data, but got {repr(sio.getvalue())}"

# Simple reproduction case
def test_simple_reproduction():
    print("Testing TruncatedStringIO with maxlen=0")
    sio = TruncatedStringIO(maxlen=0)
    sio.write("hello")

    print(f"Value: {repr(sio.getvalue())}")
    print(f"Length: {len(sio.getvalue())}")
    print()

# Test with maxlen=1 for comparison
def test_maxlen_one():
    print("Testing TruncatedStringIO with maxlen=1")
    sio = TruncatedStringIO(maxlen=1)
    try:
        sio.write("hello")
        print("No exception raised")
    except StringTruncated:
        print("StringTruncated exception raised")

    print(f"Value: {repr(sio.getvalue())}")
    print(f"Length: {len(sio.getvalue())}")
    print()

# Test with maxlen=None
def test_maxlen_none():
    print("Testing TruncatedStringIO with maxlen=None")
    sio = TruncatedStringIO(maxlen=None)
    sio.write("hello")

    print(f"Value: {repr(sio.getvalue())}")
    print(f"Length: {len(sio.getvalue())}")
    print()

# Test with empty string and maxlen=0
def test_empty_string_maxlen_zero():
    print("Testing empty string with maxlen=0")
    sio = TruncatedStringIO(maxlen=0)
    sio.write("")

    print(f"Value: {repr(sio.getvalue())}")
    print(f"Length: {len(sio.getvalue())}")
    print()

if __name__ == "__main__":
    test_simple_reproduction()
    test_maxlen_one()
    test_maxlen_none()
    test_empty_string_maxlen_zero()

    print("Running hypothesis test with 'a' as input:")
    try:
        test_maxlen_zero_should_reject_all_data("a")
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")