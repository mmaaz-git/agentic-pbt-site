from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

@given(st.text())
def test_collapsible_section_escapes_html_in_name(user_input):
    html = collapsible_section(user_input)
    if '<script>' in user_input:
        assert '<script>' not in html or '&lt;script&gt;' in html

# Run the test
if __name__ == "__main__":
    # This will find the failing case
    try:
        test_collapsible_section_escapes_html_in_name()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed!")
        import traceback
        traceback.print_exc()