import pickle
import warnings
from hypothesis import given, settings, strategies as st
from pydantic.deprecated.parse import load_str_bytes, Protocol

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# First test the hypothesis test
print("Testing with Hypothesis...")
@settings(max_examples=500)
@given(st.lists(st.integers()))
def test_load_str_bytes_pickle_encoding_parameter(lst):
    pickled_bytes = pickle.dumps(lst)
    pickled_str = pickled_bytes.decode('latin1')

    result = load_str_bytes(pickled_str, proto=Protocol.pickle,
                          encoding='latin1', allow_pickle=True)
    assert result == lst

# Run the test
try:
    test_load_str_bytes_pickle_encoding_parameter()
    print("Hypothesis test passed - no bug found")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Now test the specific failing case
print("\nTesting specific failing case...")
data = []
pickled_bytes = pickle.dumps(data)
pickled_str = pickled_bytes.decode('latin1')

print(f"Pickled bytes: {pickled_bytes}")
print(f"Pickled string (decoded with latin1): {repr(pickled_str)}")

try:
    result = load_str_bytes(pickled_str, proto=Protocol.pickle,
                           encoding='latin1', allow_pickle=True)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")