from hypothesis import given, strategies as st
from starlette.middleware.errors import ServerErrorMiddleware


@given(st.text(min_size=1, max_size=100))
def test_nbsp_has_semicolon(line_text):
    middleware = ServerErrorMiddleware(app=lambda scope, receive, send: None)
    formatted = middleware.format_line(0, line_text, 1, 0)

    if " " in line_text:
        assert "&nbsp;" in formatted, \
            "HTML entity for non-breaking space must end with semicolon"


if __name__ == "__main__":
    test_nbsp_has_semicolon()