import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

@given(
    st.text(min_size=1),
    st.text(),
    st.text(),
    st.integers(min_value=0, max_value=10),
    st.booleans(),
    st.booleans()
)
def test_tooltip_should_match_enabled_state(name, inline_details, details, n_items, enabled_param, collapsed):
    """Tooltip should appear on enabled checkboxes, not disabled ones"""
    result = collapsible_section(name, inline_details, details, n_items, enabled_param, collapsed)

    has_items = n_items is not None and n_items
    is_enabled = enabled_param and has_items
    has_tooltip = "title='Expand/collapse section'" in result

    if is_enabled:
        assert has_tooltip, f"Enabled checkboxes should have tooltip. enabled={enabled_param}, n_items={n_items}, result snippet: {result[:200]}"
    else:
        assert not has_tooltip, f"Disabled checkboxes should not have tooltip. enabled={enabled_param}, n_items={n_items}, result snippet: {result[:200]}"

# Run the test
test_tooltip_should_match_enabled_state()
print("Test completed")