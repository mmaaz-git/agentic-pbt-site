from hypothesis import given, strategies as st, settings, HealthCheck
import html


@given(st.text(min_size=1, max_size=100).map(lambda x: " " + x if " " not in x else x))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_nbsp_entity_correctness(line):
    # Only test if line contains spaces
    if " " not in line:
        return

    # This is what Starlette does (line 191 in errors.py)
    result = html.escape(line).replace(" ", "&nbsp")

    # Check if the HTML entity is correct (should have semicolon)
    assert "&nbsp;" in result, (
        f"HTML entity for non-breaking space should be '&nbsp;' with semicolon. "
        f"Got '{result}' which contains '&nbsp' without semicolon"
    )


if __name__ == "__main__":
    # Run the property-based test
    test_nbsp_entity_correctness()