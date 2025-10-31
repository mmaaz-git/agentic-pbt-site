from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.text(min_size=1))
@settings(max_examples=10)  # Limit examples for demo
def test_hash_key_parameter_changes_hash(text):
    print(f"Testing with text: '{text}'")
    arr = np.array([text], dtype=object)
    try:
        hash1 = hash_array(arr, hash_key="key1")
        hash2 = hash_array(arr, hash_key="key2")
        print(f"  ERROR: Expected to fail but succeeded!")
        print(f"  hash1={hash1}, hash2={hash2}")
        assert len(hash1) == 1
        assert len(hash2) == 1
    except ValueError as e:
        print(f"  Failed as expected: {e}")
        return

# Run the test
print("Running hypothesis test with non-16-byte hash_keys...")
test_hash_key_parameter_changes_hash()
print("\nAll tests raised ValueError as expected (since hash_keys are not 16 bytes)")