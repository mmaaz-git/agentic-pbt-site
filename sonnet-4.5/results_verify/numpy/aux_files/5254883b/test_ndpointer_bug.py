import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings

@given(shape_str=st.text(min_size=1, max_size=20))
@settings(max_examples=300)
def test_ndpointer_rejects_string_shape(shape_str):
    try:
        ptr = npc.ndpointer(shape=shape_str)
        assert False, f"ndpointer should reject string shape, but got _shape_={ptr._shape_}"
    except (TypeError, ValueError) as e:
        pass

# Test with the specific failing input
print("Testing with shape='0':")
try:
    ptr = npc.ndpointer(shape='0')
    print(f"FAIL: ndpointer accepted string shape '0', got _shape_={ptr._shape_}")
except (TypeError, ValueError) as e:
    print(f"PASS: Raised {type(e).__name__}: {e}")

# Run the hypothesis test
print("\nRunning hypothesis test:")
test_ndpointer_rejects_string_shape()