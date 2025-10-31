import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, assume, settings, example
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
    num_leading_newlines=st.integers(min_value=0, max_value=10),
)
def test_get_exception_info_line_indexing(lineno, num_lines, num_leading_newlines):
    assume(lineno <= num_lines + num_leading_newlines)
    assume(lineno > num_leading_newlines)  # The error must be in actual content, not empty lines

    # Create source with leading newlines
    leading = "\n" * num_leading_newlines
    lines_list = [f"line {i}" for i in range(1, num_lines + 1)]
    source = leading + "\n".join(lines_list)

    exc = MockException(
        filename="test.html",
        lineno=lineno,
        source=source
    )

    try:
        info = get_exception_info(exc)

        assert info["line"] == lineno
        # The expected line content considering the leading newlines
        if lineno <= num_leading_newlines:
            expected_during = ""  # Empty line
        else:
            expected_during = f"line {lineno - num_leading_newlines}"

        assert info["during"] == expected_during, (
            f"Line {lineno} should contain '{expected_during}', "
            f"but got '{info['during']}'"
        )
    except IndexError as e:
        # This is also a bug - it shouldn't raise IndexError
        print(f"IndexError for lineno={lineno}, num_lines={num_lines}, num_leading_newlines={num_leading_newlines}")
        print(f"Source has {len(source.split('\n'))} lines originally")
        print(f"After strip(), source has {len(source.strip().split('\n'))} lines")
        raise


if __name__ == "__main__":
    test_get_exception_info_line_indexing()
