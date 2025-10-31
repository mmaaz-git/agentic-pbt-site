from truncated_string_io import TruncatedStringIO, StringTruncated

# Simple reproduction case
def test_simple_reproduction():
    print("Testing TruncatedStringIO with maxlen=0")
    sio = TruncatedStringIO(maxlen=0)
    sio.write("hello")

    print(f"Value: {repr(sio.getvalue())}")
    print(f"Length: {len(sio.getvalue())}")
    print(f"Expected: Empty string or StringTruncated exception")
    print(f"Actual: Data was stored despite maxlen=0")
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

# Manual hypothesis test
def test_manual_hypothesis():
    print("Manual hypothesis test - writing 'a' with maxlen=0:")
    sio = TruncatedStringIO(maxlen=0)
    try:
        sio.write("a")
    except StringTruncated:
        print("StringTruncated exception was raised")

    result = sio.getvalue()
    print(f"Value after write: {repr(result)}")
    print(f"Length: {len(result)}")

    if len(result) == 0:
        print("PASS: maxlen=0 correctly prevented storing data")
    else:
        print(f"FAIL: maxlen=0 should not store any data, but got {repr(result)}")
    print()

# Test the falsy check issue
def test_falsy_values():
    print("Testing falsy values:")

    # Test with 0
    print("  maxlen=0:")
    sio = TruncatedStringIO(maxlen=0)
    if sio.maxlen:
        print("    0 is truthy (unexpected)")
    else:
        print("    0 is falsy (as expected in Python)")

    # Test with None
    print("  maxlen=None:")
    sio = TruncatedStringIO(maxlen=None)
    if sio.maxlen:
        print("    None is truthy (unexpected)")
    else:
        print("    None is falsy (as expected in Python)")

    # Test with 1
    print("  maxlen=1:")
    sio = TruncatedStringIO(maxlen=1)
    if sio.maxlen:
        print("    1 is truthy (as expected)")
    else:
        print("    1 is falsy (unexpected)")
    print()

if __name__ == "__main__":
    test_simple_reproduction()
    test_maxlen_one()
    test_maxlen_none()
    test_empty_string_maxlen_zero()
    test_manual_hypothesis()
    test_falsy_values()