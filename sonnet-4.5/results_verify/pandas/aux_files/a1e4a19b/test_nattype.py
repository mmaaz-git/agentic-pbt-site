from hypothesis import given, strategies as st, settings
from pandas.api.typing import NaTType

@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=100))
def test_nattype_singleton_multiple_calls(n):
    instances = [NaTType() for _ in range(n)]
    first = instances[0]
    assert all(inst is first for inst in instances), "NaTType() should always return the same singleton instance"

# Run the test
if __name__ == "__main__":
    try:
        test_nattype_singleton_multiple_calls()
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed: {e}")
        # Try with n=2 specifically (the failing input mentioned)
        instances = [NaTType() for _ in range(2)]
        print(f"With n=2: instances[0] is instances[1] = {instances[0] is instances[1]}")