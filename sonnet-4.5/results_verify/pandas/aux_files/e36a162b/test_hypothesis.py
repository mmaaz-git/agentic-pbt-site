import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from pandas.core.ops.mask_ops import kleene_and

@st.composite
def bool_arrays_with_masks(draw):
    size = draw(st.integers(min_value=1, max_value=100))
    values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    mask_presence = draw(st.booleans())
    if mask_presence:
        mask_values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
        mask = np.array(mask_values, dtype=bool)
    else:
        mask = None
    return np.array(values, dtype=bool), mask

@settings(max_examples=500)
@given(bool_arrays_with_masks(), bool_arrays_with_masks())
def test_kleene_and_commutativity_arrays(left_data, right_data):
    left, left_mask = left_data
    right, right_mask = right_data

    if len(left) != len(right):
        return

    try:
        result1, mask1 = kleene_and(left, right, left_mask, right_mask)
        result2, mask2 = kleene_and(right, left, right_mask, left_mask)

        assert np.array_equal(result1, result2)
        assert np.array_equal(mask1, mask2)
        print(".", end="", flush=True)
    except RecursionError:
        print(f"\nRecursionError with left_mask={left_mask is None}, right_mask={right_mask is None}")
        raise
    except Exception as e:
        print(f"\nOther error: {e}")
        raise

print("Running hypothesis test...")
try:
    test_kleene_and_commutativity_arrays()
    print("\nAll tests passed!")
except Exception as e:
    print(f"\nTest failed: {e}")