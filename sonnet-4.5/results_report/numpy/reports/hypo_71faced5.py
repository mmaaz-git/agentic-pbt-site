import pandas.tseries.frequencies
from hypothesis import given, strategies as st, settings

FREQ_STRINGS = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN",
    "M", "MS", "ME", "BM", "BMS",
    "Q", "QS", "Q-JAN", "Q-FEB", "Q-MAR", "Q-APR", "Q-MAY", "Q-JUN",
    "Q-JUL", "Q-AUG", "Q-SEP", "Q-OCT", "Q-NOV", "Q-DEC",
    "Y", "YS", "Y-JAN", "Y-FEB", "Y-MAR", "Y-APR", "Y-MAY", "Y-JUN",
    "Y-JUL", "Y-AUG", "Y-SEP", "Y-OCT", "Y-NOV", "Y-DEC",
]

@given(
    source=st.sampled_from(FREQ_STRINGS),
    target=st.sampled_from(FREQ_STRINGS)
)
@settings(max_examples=1000)
def test_subperiod_superperiod_inverse(source, target):
    is_sub = pandas.tseries.frequencies.is_subperiod(source, target)
    is_super_reverse = pandas.tseries.frequencies.is_superperiod(target, source)

    assert is_sub == is_super_reverse, (
        f"is_subperiod('{source}', '{target}') = {is_sub} "
        f"but is_superperiod('{target}', '{source}') = {is_super_reverse}"
    )

if __name__ == "__main__":
    test_subperiod_superperiod_inverse()