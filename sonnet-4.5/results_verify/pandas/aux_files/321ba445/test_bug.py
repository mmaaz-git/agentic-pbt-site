import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings

VALID_FREQS = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]

@given(st.sampled_from(VALID_FREQS), st.sampled_from(VALID_FREQS))
@settings(max_examples=200)
def test_subperiod_superperiod_inverse(source, target):
    """
    Property: is_subperiod and is_superperiod should be inverse relations.
    If is_subperiod(A, B) is True, then is_superperiod(B, A) should also be True.
    """
    sub_result = freq.is_subperiod(source, target)
    super_result = freq.is_superperiod(target, source)

    assert sub_result == super_result, \
        f"Inverse property violated: is_subperiod({source}, {target})={sub_result} but is_superperiod({target}, {source})={super_result}"

# Run the test
if __name__ == "__main__":
    test_subperiod_superperiod_inverse()
    print("Test passed!")