from pandas.tseries.frequencies import is_subperiod, is_superperiod
from hypothesis import given, strategies as st

FREQ_STRINGS = ['D', 'B', 'C', 'W', 'M', 'Q', 'Y', 'h', 'min', 's', 'ms', 'us', 'ns']

@given(
    source=st.sampled_from(FREQ_STRINGS),
    target=st.sampled_from(FREQ_STRINGS)
)
def test_subperiod_superperiod_inverse(source, target):
    sub_result = is_subperiod(source, target)
    super_result = is_superperiod(target, source)

    assert sub_result == super_result, (
        f"Inverse relationship violated: "
        f"is_subperiod({source}, {target}) = {sub_result}, "
        f"but is_superperiod({target}, {source}) = {super_result}"
    )

if __name__ == "__main__":
    # Run the test
    test_subperiod_superperiod_inverse()