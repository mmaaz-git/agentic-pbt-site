from hypothesis import given, strategies as st
from xarray.backends.netcdf3 import is_valid_nc3_name

@given(st.text())
def test_is_valid_nc3_name_doesnt_crash(s):
    result = is_valid_nc3_name(s)
    assert isinstance(result, bool)

# Run the test
test_is_valid_nc3_name_doesnt_crash()