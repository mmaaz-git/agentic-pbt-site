import numpy as np
import numpy.strings
from hypothesis import given, strategies as st, settings


@given(st.lists(st.text(), min_size=1), st.text(min_size=1))
@settings(max_examples=1000)
def test_endswith_rfind_consistency(strings, suffix):
    arr = np.array(strings)
    endswith_result = numpy.strings.endswith(arr, suffix)
    rfind_result = numpy.strings.rfind(arr, suffix)
    str_lens = numpy.strings.str_len(arr)

    for ew, rfind_idx, s_len in zip(endswith_result, rfind_result, str_lens):
        if ew:
            expected_idx = s_len - len(suffix)
            assert rfind_idx == expected_idx, f"endswith={ew}, rfind={rfind_idx}, len={s_len}, suffix_len={len(suffix)}"


# Test the specific failing case manually
print("Testing specific case: strings=[''], suffix='\\x00'")
arr = np.array([''])
endswith_result = numpy.strings.endswith(arr, '\x00')
rfind_result = numpy.strings.rfind(arr, '\x00')
str_lens = numpy.strings.str_len(arr)

for ew, rfind_idx, s_len in zip(endswith_result, rfind_result, str_lens):
    if ew:
        expected_idx = s_len - len('\x00')
        try:
            assert rfind_idx == expected_idx, f"endswith={ew}, rfind={rfind_idx}, len={s_len}, suffix_len={len('\x00')}, expected_idx={expected_idx}"
            print("Test passed (no assertion error)")
        except AssertionError as e:
            print(f"Test failed with: {e}")