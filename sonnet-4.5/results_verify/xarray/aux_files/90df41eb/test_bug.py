import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

print("Testing is_uniform_spaced with single-element array...")
from xarray.core.utils import is_uniform_spaced

# Test 1: Single element array
try:
    result = is_uniform_spaced([5.0])
    print(f"Single element [5.0]: result = {result}")
except Exception as e:
    print(f"Single element [5.0]: ERROR - {type(e).__name__}: {e}")

# Test 2: Empty array
print("\nTesting with empty array...")
try:
    result = is_uniform_spaced([])
    print(f"Empty array []: result = {result}")
except Exception as e:
    print(f"Empty array []: ERROR - {type(e).__name__}: {e}")

# Test 3: Two element array (should always be uniform)
print("\nTesting with two-element array...")
try:
    result = is_uniform_spaced([1.0, 5.0])
    print(f"Two elements [1.0, 5.0]: result = {result}")
except Exception as e:
    print(f"Two elements [1.0, 5.0]: ERROR - {type(e).__name__}: {e}")

# Test 4: Normal uniform array
print("\nTesting with normal uniform array...")
try:
    result = is_uniform_spaced([1.0, 2.0, 3.0, 4.0])
    print(f"Uniform [1.0, 2.0, 3.0, 4.0]: result = {result}")
except Exception as e:
    print(f"Uniform [1.0, 2.0, 3.0, 4.0]: ERROR - {type(e).__name__}: {e}")

# Test 5: Non-uniform array
print("\nTesting with non-uniform array...")
try:
    result = is_uniform_spaced([-4, 0, 100])
    print(f"Non-uniform [-4, 0, 100]: result = {result}")
except Exception as e:
    print(f"Non-uniform [-4, 0, 100]: ERROR - {type(e).__name__}: {e}")

# Test the hypothesis test
print("\nRunning the hypothesis test from the bug report...")
from hypothesis import given, strategies as st, settings

@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=10)
def test_single_element_uniform(x):
    try:
        result = is_uniform_spaced([x])
        print(f"Hypothesis test with [{x}]: result = {result}")
        assert result, f"Single element array should be uniformly spaced: [{x}]"
    except Exception as e:
        print(f"Hypothesis test with [{x}]: ERROR - {type(e).__name__}: {e}")
        raise

test_single_element_uniform()