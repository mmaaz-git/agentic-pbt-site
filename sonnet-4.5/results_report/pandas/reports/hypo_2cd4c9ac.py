from hypothesis import given, strategies as st, assume, settings
from pandas.core.dtypes.common import is_dtype_equal


@given(st.text())
@settings(max_examples=200)
def test_is_dtype_equal_invalid_string(invalid_str):
    assume(invalid_str not in ['int8', 'int16', 'int32', 'int64',
                                'uint8', 'uint16', 'uint32', 'uint64',
                                'float16', 'float32', 'float64',
                                'bool', 'object', 'string',
                                'datetime64', 'timedelta64', 'int', 'float'])
    result = is_dtype_equal(invalid_str, 'int64')
    if result:
        result_rev = is_dtype_equal('int64', invalid_str)
        assert result == result_rev

if __name__ == "__main__":
    test_is_dtype_equal_invalid_string()