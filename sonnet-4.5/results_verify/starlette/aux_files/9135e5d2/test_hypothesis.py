import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, strategies as st, settings

@given(
    flag=st.sampled_from(['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']),
    num_duplicates=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=200)
def test_ndpointer_duplicate_flags_invariant(flag, num_duplicates):
    flags_single = [flag]
    flags_dup = [flag] * num_duplicates

    ptr_single = npc.ndpointer(flags=flags_single)
    ptr_dup = npc.ndpointer(flags=flags_dup)

    assert ptr_single._flags_ == ptr_dup._flags_, \
        f"Duplicate flags should have same effect: {ptr_single._flags_} vs {ptr_dup._flags_}"

if __name__ == "__main__":
    test_ndpointer_duplicate_flags_invariant()
    print("Test completed")