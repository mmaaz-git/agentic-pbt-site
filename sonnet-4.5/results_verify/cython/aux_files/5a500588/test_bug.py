import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Debugger.libpython import TruncatedStringIO, StringTruncated

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

if __name__ == "__main__":
    test_simple_reproduction()
    test_maxlen_one()
    test_maxlen_none()

    print("Running hypothesis test with 'a' as input:")
    try:
        test_maxlen_zero_should_reject_all_data("a")
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")