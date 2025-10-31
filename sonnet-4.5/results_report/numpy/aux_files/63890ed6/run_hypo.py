import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings, assume
import sys
import traceback

# Test the specific case we found
print('Testing ndim=2.0 (a float that equals integer 2):')
try:
    ptr = npc.ndpointer(ndim=2.0)
    arr = np.zeros((2, 3), dtype=np.int32)
    result = ptr.from_param(arr)
    print('ERROR: ndim=2.0 was accepted when it should have been rejected!')
    print(f'  Created pointer with _ndim_={ptr._ndim_}')
    print(f'  from_param returned: {result}')
except Exception as e:
    print(f'  Good: Raised {type(e).__name__}: {e}')
    traceback.print_exc()

# Now test the hypothesis test
print('\n' + '='*60)
print('Running hypothesis test:')
@given(ndim_value=st.one_of(st.floats(), st.text(), st.lists(st.integers())))
@settings(max_examples=200, database=None)
def test_ndpointer_ndim_type_validation(ndim_value):
    assume(not isinstance(ndim_value, (int, type(None))))

    try:
        ptr = npc.ndpointer(ndim=ndim_value)
        arr = np.zeros((2, 3), dtype=np.int32)
        result = ptr.from_param(arr)

        # Check if this float is exactly equal to an integer
        if isinstance(ndim_value, float) and ndim_value == int(ndim_value) and not np.isnan(ndim_value) and not np.isinf(ndim_value):
            print(f'\nFailing input: `ndim={repr(ndim_value)}`')
            assert False, f'Should reject non-integer ndim: {ndim_value}'
    except (TypeError, ValueError, OverflowError) as e:
        pass  # Expected - these exceptions are correct

try:
    test_ndpointer_ndim_type_validation()
    print('\nAll tests passed')
except AssertionError as e:
    print(f'\nAssertion Error: {e}')
    print('\nThis confirms the bug: float values that happen to equal integers (like 2.0) are incorrectly accepted.')