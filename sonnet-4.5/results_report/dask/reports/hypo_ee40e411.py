from hypothesis import given, strategies as st, settings, example
from dask.utils import key_split

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10),
       st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10))
@example(key1='task', key2='feedback')  # Add the specific failing case
@settings(max_examples=500)
def test_key_split_compound_key(key1, key2):
    """Test that key_split preserves legitimate English words in compound keys"""
    s = f"{key1}-{key2}-1"
    result = key_split(s)

    # If key2 is not 8 chars of only a-f letters, it should be preserved
    if len(key2) != 8 or not all(c in 'abcdef' for c in key2):
        assert result == f"{key1}-{key2}", f"key_split('{s}') returned '{result}', expected '{key1}-{key2}'"

# Run the test
if __name__ == "__main__":
    print("Running property-based test for key_split...")
    print("Testing that legitimate words are preserved in compound keys")
    print()

    try:
        test_key_split_compound_key()
        print("✓ All tests passed!")
    except AssertionError as e:
        print(f"✗ Test failed!")
        print(f"Assertion error: {e}")
        print("\nThis demonstrates that key_split incorrectly strips legitimate English words")
        print("that happen to be 8 characters long and contain only letters a-f.")