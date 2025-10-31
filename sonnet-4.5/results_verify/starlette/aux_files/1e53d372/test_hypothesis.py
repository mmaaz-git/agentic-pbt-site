from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.middleware.errors import ServerErrorMiddleware


def dummy_app(scope, receive, send):
    pass


@given(
    st.text(min_size=1, max_size=100),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=0, max_value=20)
)
@settings(max_examples=300)
def test_format_line_html_entity_correctness(line, frame_lineno, frame_index):
    middleware = ServerErrorMiddleware(dummy_app, debug=True)
    formatted = middleware.format_line(frame_index, line, frame_lineno, frame_index)

    if " " in line and "&nbsp" in formatted:
        assert "&nbsp;" in formatted

# Run the test
if __name__ == "__main__":
    test_format_line_html_entity_correctness()
    print("Test completed")