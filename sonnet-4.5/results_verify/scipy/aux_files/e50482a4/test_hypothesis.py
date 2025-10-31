import numpy as np
import scipy.stats as stats
from hypothesis import given, strategies as st, settings, example

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.sampled_from(['rank', 'weak', 'strict', 'mean'])
)
@settings(max_examples=100)
@example(data=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], kind='rank')
def test_percentileofscore_all_kinds_bounded(data, kind):
    score = data[0]
    result = stats.percentileofscore(data, score, kind=kind)
    if not np.isnan(result):
        if result < 0 or result > 100:
            print(f"FAILURE: data={data}, kind='{kind}'")
            print(f"  Result: {result}, repr: {repr(result)}")
            print(f"  Outside [0, 100]: result < 0 = {result < 0}, result > 100 = {result > 100}")
            assert False, f"Result {result} outside [0, 100]"

# Run the test
test_percentileofscore_all_kinds_bounded()
print("Test completed - checking for failures...")