from hypothesis import given, strategies as st, settings
from xarray.backends.netcdf3 import is_valid_nc3_name

@given(st.text())
@settings(max_examples=1000)
def test_is_valid_nc3_name_does_not_crash(name):
    result = is_valid_nc3_name(name)
    assert isinstance(result, bool)

# Run the test
if __name__ == "__main__":
    test_is_valid_nc3_name_does_not_crash()