from hypothesis import given, strategies as st, assume
from scipy.io.arff._arffread import DateAttribute
import pytest


@given(st.text(alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=10))
def test_invalid_date_format_should_raise_valueerror(invalid_pattern):
    assume('yyyy' not in invalid_pattern)
    assume('yy' not in invalid_pattern)
    assume('MM' not in invalid_pattern)
    assume('dd' not in invalid_pattern)
    assume('HH' not in invalid_pattern)
    assume('mm' not in invalid_pattern)
    assume('ss' not in invalid_pattern)

    date_string = f'date "{invalid_pattern}"'

    with pytest.raises(ValueError):
        DateAttribute._get_date_format(date_string)

if __name__ == "__main__":
    test_invalid_date_format_should_raise_valueerror()