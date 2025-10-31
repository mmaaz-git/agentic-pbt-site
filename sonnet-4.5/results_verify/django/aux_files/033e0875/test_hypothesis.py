import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, assume, settings
from django.template.backends.jinja2 import get_exception_info


class MockException:
    def __init__(self, filename, lineno, source):
        self.filename = filename
        self.lineno = lineno
        self.source = source
        self.message = "Test error"


@settings(max_examples=500)
@given(
    lineno=st.integers(min_value=1, max_value=100),
    num_lines=st.integers(min_value=1, max_value=100),
)
def test_get_exception_info_line_indexing(lineno, num_lines):
    assume(lineno <= num_lines)

    lines_list = [f"line {i}" for i in range(1, num_lines + 1)]
    source = "\n".join(lines_list)

    exc = MockException(
        filename="test.html",
        lineno=lineno,
        source=source
    )

    info = get_exception_info(exc)

    assert info["line"] == lineno
    expected_during = f"line {lineno}"
    assert info["during"] == expected_during, (
        f"Line {lineno} should contain '{expected_during}', "
        f"but got '{info['during']}'"
    )

if __name__ == "__main__":
    # Run the hypothesis test
    test_get_exception_info_line_indexing()