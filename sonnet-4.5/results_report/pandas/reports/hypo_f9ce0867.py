from hypothesis import given, strategies as st, example
import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime, _convert_datetimes


@given(
    values=st.lists(
        st.floats(min_value=-1e8, max_value=1e8, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    )
)
@example(values=[0.5])  # The failing example from the bug report
def test_convert_datetimes_consistency_with_parse_datetime(values):
    series = pd.Series(values)
    vectorized_result = _convert_datetimes(series, "d")

    for i, value in enumerate(values):
        scalar_result = _parse_datetime(value, "d")
        vectorized_value = vectorized_result.iloc[i]

        scalar_ts = pd.Timestamp(scalar_result)
        vectorized_ts = pd.Timestamp(vectorized_value)

        time_diff_ms = abs((scalar_ts - vectorized_ts).total_seconds() * 1000)

        assert time_diff_ms < 1, (
            f"Inconsistency at index {i} for value {value} with unit d: "
            f"_parse_datetime returned {scalar_result}, "
            f"_convert_datetimes returned {vectorized_value}, "
            f"difference: {time_diff_ms}ms"
        )

# Run the test
if __name__ == "__main__":
    test_convert_datetimes_consistency_with_parse_datetime()