"""Test the HTML entity bug in starlette.middleware.errors"""
from hypothesis import given, strategies as st, settings
from starlette.middleware.errors import ServerErrorMiddleware


# First let's test the specific reproduction
def test_basic_reproduction():
    """Basic test showing the issue"""
    middleware = ServerErrorMiddleware(app=lambda: None)
    line = "hello world"
    formatted = middleware.format_line(
        index=0,
        line=line,
        frame_lineno=10,
        frame_index=0
    )
    print("Formatted output:", formatted)
    # Check if it contains the incomplete entity
    if "&nbsp" in formatted and "&nbsp;" not in formatted:
        print("BUG CONFIRMED: Contains '&nbsp' without semicolon")
        return True
    else:
        print("No bug found - either contains '&nbsp;' or no nbsp at all")
        return False


# Property-based test from the bug report
@given(line=st.text(min_size=1, max_size=100).filter(lambda x: " " in x))
@settings(max_examples=100)
def test_format_line_proper_html_entities(line):
    middleware = ServerErrorMiddleware(app=lambda: None)

    formatted = middleware.format_line(
        index=0,
        line=line,
        frame_lineno=10,
        frame_index=0
    )

    # Check the assertion from the bug report
    assert "&nbsp;" in formatted or "&nbsp" not in formatted, \
        f"HTML entity &nbsp should end with semicolon. Line: {line!r}, Formatted: {formatted!r}"


if __name__ == "__main__":
    print("Running basic reproduction test...")
    bug_found = test_basic_reproduction()

    if bug_found:
        print("\nRunning property-based tests...")
        try:
            test_format_line_proper_html_entities()
            print("Property tests passed - no assertion errors")
        except AssertionError as e:
            print(f"Property test failed: {e}")