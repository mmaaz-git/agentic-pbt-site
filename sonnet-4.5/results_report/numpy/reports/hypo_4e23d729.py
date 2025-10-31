import numpy.ma as ma
from hypothesis import given, settings, strategies as st, assume


@settings(max_examples=1000)
@given(st.lists(st.booleans(), min_size=1, max_size=20))
def test_mask_or_accepts_lists(mask1):
    assume(len(mask1) > 0)
    mask2 = [not m for m in mask1]
    result = ma.mask_or(mask1, mask2)

# Run the test
if __name__ == "__main__":
    test_mask_or_accepts_lists()