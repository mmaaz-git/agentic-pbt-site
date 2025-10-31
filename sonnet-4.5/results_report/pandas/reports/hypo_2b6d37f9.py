from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

VALID_FREQUENCIES = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT",
    "M", "MS", "BM", "BMS",
    "Q", "QS", "BQ", "BQS",
    "Q-JAN", "Q-FEB", "Q-MAR", "Q-APR", "Q-MAY", "Q-JUN",
    "Q-JUL", "Q-AUG", "Q-SEP", "Q-OCT", "Q-NOV", "Q-DEC",
    "Y", "YS", "BY", "BYS",
    "Y-JAN", "Y-FEB", "Y-MAR", "Y-APR", "Y-MAY", "Y-JUN",
    "Y-JUL", "Y-AUG", "Y-SEP", "Y-OCT", "Y-NOV", "Y-DEC",
]

freq_strategy = st.sampled_from(VALID_FREQUENCIES)

@given(source=freq_strategy, target=freq_strategy)
@settings(max_examples=1000)
def test_inverse_relationship_superperiod_subperiod(source, target):
    """
    Property: If is_superperiod(source, target) is True,
    then is_subperiod(target, source) should also be True.
    """
    super_result = is_superperiod(source, target)
    sub_result = is_subperiod(target, source)

    if super_result:
        assert sub_result, (
            f"is_superperiod({source!r}, {target!r}) = True, "
            f"but is_subperiod({target!r}, {source!r}) = {sub_result}"
        )

if __name__ == "__main__":
    test_inverse_relationship_superperiod_subperiod()