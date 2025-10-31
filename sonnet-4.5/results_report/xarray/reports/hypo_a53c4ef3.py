from hypothesis import given, strategies as st
from xarray.util.deprecation_helpers import deprecate_dims


@given(
    dims_value=st.text(),
    dim_value=st.text()
)
def test_deprecate_dims_precedence(dims_value, dim_value):
    @deprecate_dims
    def func(*, dim=None):
        return dim

    result = func(dims=dims_value, dim=dim_value)

    assert result == dim_value, \
        f"Expected dim={dim_value!r} to take precedence, but got {result!r}"


if __name__ == "__main__":
    test_deprecate_dims_precedence()
