import numpy as np
import numpy.strings
from hypothesis import given, strategies as st, settings, example


@given(st.lists(st.text(), min_size=1), st.text(min_size=1))
@example([''], '\x00')  # Force the failing case
@settings(max_examples=10)
def test_endswith_rfind_consistency(strings, suffix):
    arr = np.array(strings)
    endswith_result = numpy.strings.endswith(arr, suffix)
    rfind_result = numpy.strings.rfind(arr, suffix)
    str_lens = numpy.strings.str_len(arr)

    for ew, rfind_idx, s_len in zip(endswith_result, rfind_result, str_lens):
        if ew:
            expected_idx = s_len - len(suffix)
            assert rfind_idx == expected_idx, f"endswith=True but rfind={rfind_idx} != expected {expected_idx} (str_len={s_len}, suffix_len={len(suffix)})"

# Run the test
if __name__ == "__main__":
    test_endswith_rfind_consistency()