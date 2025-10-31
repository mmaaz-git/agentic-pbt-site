from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

@given(
    st.text(min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=-1)
)
def test_collapsible_section_negative_n_items(name, n_items):
    """Property: collapsible_section should treat negative n_items as invalid"""
    result = collapsible_section(name, n_items=n_items)

    assert "disabled" in result, f"Negative n_items should result in disabled section. Got: {result[:200]}"

# Run the test
if __name__ == "__main__":
    test_collapsible_section_negative_n_items()