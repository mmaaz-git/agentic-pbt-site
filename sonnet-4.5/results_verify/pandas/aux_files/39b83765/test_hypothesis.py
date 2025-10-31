from hypothesis import given, strategies as st
from pandas.plotting._misc import _Options


@given(st.booleans())
def test_options_get_handles_aliases(value):
    opts = _Options()
    opts["xaxis.compat"] = value

    result_canonical = opts.get("xaxis.compat", "default")
    result_alias = opts.get("x_compat", "default")

    assert result_canonical == value
    assert result_alias == value

if __name__ == "__main__":
    test_options_get_handles_aliases()