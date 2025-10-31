from hypothesis import given, strategies as st, settings
from scipy.odr._odrpack import _report_error

@given(st.integers(min_value=0, max_value=99999))
@settings(max_examples=1000)
def test_report_error_always_returns_list(info):
    result = _report_error(info)
    assert len(result) > 0, f"_report_error({info}) returned empty list"

if __name__ == "__main__":
    test_report_error_always_returns_list()