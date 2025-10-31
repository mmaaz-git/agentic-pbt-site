from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st

VALID_FREQS = ["D", "B", "C", "M", "h", "min", "s", "ms", "us", "ns", "W", "Y", "Q"]

@given(
    source=st.sampled_from(VALID_FREQS),
    target=st.sampled_from(VALID_FREQS)
)
def test_subperiod_superperiod_inverse(source, target):
    result_sub = is_subperiod(source, target)
    result_super = is_superperiod(target, source)

    assert result_sub == result_super, (
        f"is_subperiod({source}, {target}) = {result_sub} but "
        f"is_superperiod({target}, {source}) = {result_super}"
    )

if __name__ == "__main__":
    # Run the test
    test_subperiod_superperiod_inverse()