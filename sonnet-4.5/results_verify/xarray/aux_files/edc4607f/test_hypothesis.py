import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from xarray.indexes import RangeIndex

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    num=st.integers(min_value=1, max_value=1000),
    endpoint=st.booleans()
)
@settings(max_examples=100)
def test_linspace_no_crash(start, stop, num, endpoint):
    try:
        index = RangeIndex.linspace(start, stop, num, endpoint=endpoint, dim="x")
        assert index.size == num
        print(f"✓ start={start:.2f}, stop={stop:.2f}, num={num}, endpoint={endpoint}")
    except ZeroDivisionError:
        print(f"✗ ZeroDivisionError: start={start:.2f}, stop={stop:.2f}, num={num}, endpoint={endpoint}")
        assert False, "RangeIndex.linspace crashed with valid inputs"

# Run the test
if __name__ == "__main__":
    try:
        test_linspace_no_crash()
        print("\nTest completed successfully if no failures above")
    except AssertionError as e:
        print(f"\nTest failed: {e}")