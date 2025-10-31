from hypothesis import given, settings, strategies as st
import scipy.fftpack

@given(st.integers(min_value=-1000, max_value=0))
@settings(max_examples=100)
def test_next_fast_len_rejects_non_positive(target):
    try:
        result = scipy.fftpack.next_fast_len(target)
        if target <= 0:
            assert False, f"Should raise ValueError but returned {result}"
    except (ValueError, RuntimeError):
        pass

if __name__ == "__main__":
    test_next_fast_len_rejects_non_positive()