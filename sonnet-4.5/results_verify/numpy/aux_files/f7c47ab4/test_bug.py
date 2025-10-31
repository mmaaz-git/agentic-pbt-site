from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.rec


@settings(max_examples=500)
@given(
    st.lists(st.lists(st.integers(), min_size=1, max_size=5), min_size=0, max_size=3),
    st.lists(st.text(alphabet='abc', min_size=1, max_size=3), min_size=1, max_size=5)
)
def test_fromarrays_round_trip(int_arrays, names_list):
    if len(int_arrays) != len(names_list):
        return

    arrays = [np.array(arr) for arr in int_arrays]
    names = ','.join(names_list)

    try:
        rec_arr = numpy.rec.fromarrays(arrays, names=names)
        assert len(rec_arr) == (len(arrays[0]) if arrays else 0)
    except (ValueError, TypeError):
        pass

# Test the specific failing case directly
print("Testing empty array list with name 'a'...")
try:
    rec_arr = numpy.rec.fromarrays([], names='a')
    print(f"Test passed! Result: {rec_arr}")
except IndexError as e:
    print(f"IndexError occurred: {e}")

# Try with shape specified
print("\nTesting empty array list with explicit shape...")
try:
    result = numpy.rec.fromarrays([], names='a', shape=(0,))
    print(f"Success with shape=(0,): {result}")
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Failed even with shape specified: {e}")

# Test how other NumPy functions handle empty lists
print("\nTesting other NumPy functions with empty lists:")
try:
    arr = np.array([])
    print(f"np.array([]): {arr}, shape: {arr.shape}")
except Exception as e:
    print(f"np.array([]) failed: {e}")

try:
    np.stack([])
except ValueError as e:
    print(f"np.stack([]) raises ValueError: {e}")

try:
    np.concatenate([])
except ValueError as e:
    print(f"np.concatenate([]) raises ValueError: {e}")