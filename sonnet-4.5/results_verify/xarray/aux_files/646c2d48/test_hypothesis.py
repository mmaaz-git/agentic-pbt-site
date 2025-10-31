import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import xarray.core.formatting_html as fmt_html

@given(st.text(min_size=1))
def test_wrap_default_matches_documented_behavior(html_input):
    """
    Test that the default behavior of _wrap_datatree_repr matches its documentation.

    The docstring says: "Default is True."
    Therefore, calling the function without the end parameter should behave
    the same as calling it with end=True.
    """
    result_default = fmt_html._wrap_datatree_repr(html_input)
    result_true = fmt_html._wrap_datatree_repr(html_input, end=True)

    # According to the docstring, these should be the same
    assert result_default == result_true, \
        "Default behavior should match end=True as documented"

# Run the test
test_wrap_default_matches_documented_behavior()