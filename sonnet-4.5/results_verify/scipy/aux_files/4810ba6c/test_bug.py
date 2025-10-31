from hypothesis import given, strategies as st
import pytest
import scipy.datasets

@given(st.text(min_size=1))
def test_clear_cache_rejects_arbitrary_strings(text_input):
    """Property: clear_cache should reject any string input"""
    if callable(text_input):
        return

    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(text_input)

@given(st.integers())
def test_clear_cache_rejects_integers(int_input):
    """Property: clear_cache should reject integer inputs"""
    with pytest.raises((ValueError, TypeError, AssertionError)):
        scipy.datasets.clear_cache(int_input)

if __name__ == "__main__":
    # Test with specific examples
    print("Testing with string '0'...")
    try:
        test_clear_cache_rejects_arbitrary_strings('0')
        print("String test passed!")
    except Exception as e:
        print(f"String test failed: {e}")

    print("\nTesting with integer 42...")
    try:
        test_clear_cache_rejects_integers(42)
        print("Integer test passed!")
    except Exception as e:
        print(f"Integer test failed: {e}")