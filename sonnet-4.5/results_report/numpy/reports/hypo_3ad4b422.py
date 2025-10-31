import numpy as np
from hypothesis import given, settings, strategies as st


datetime_strategy = st.integers(min_value=0, max_value=20000).map(
    lambda days: np.datetime64('2000-01-01') + np.timedelta64(days, 'D')
)


@given(datetime_strategy, datetime_strategy)
@settings(max_examples=1000)
def test_busday_count_antisymmetric(date1, date2):
    count_forward = np.busday_count(date1, date2)
    count_backward = np.busday_count(date2, date1)
    assert count_forward == -count_backward, f"Antisymmetry violated: busday_count({date1}, {date2})={count_forward}, busday_count({date2}, {date1})={count_backward}"


if __name__ == "__main__":
    test_busday_count_antisymmetric()