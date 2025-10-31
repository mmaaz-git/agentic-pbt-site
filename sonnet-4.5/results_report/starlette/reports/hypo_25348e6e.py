from hypothesis import given, strategies as st
from starlette.middleware.errors import ServerErrorMiddleware
import html


@given(st.text(min_size=1, max_size=100).filter(lambda x: ' ' in x))
def test_html_entity_completeness(line_with_spaces):
    middleware = ServerErrorMiddleware(None, debug=True)

    escaped = html.escape(line_with_spaces).replace(" ", "&nbsp")

    if "&nbsp" in escaped and "&nbsp;" not in escaped:
        raise AssertionError(
            f"Incomplete HTML entity found. "
            f"Missing semicolon in &nbsp entity. "
            f"Input: {repr(line_with_spaces)}, "
            f"Output: {escaped}"
        )


if __name__ == "__main__":
    # Run the property test
    test_html_entity_completeness()