from hypothesis import given, strategies as st, settings, example
from Cython.Build.Inline import strip_common_indent


@given(st.integers(min_value=1, max_value=20), st.text(alphabet='abcdef', min_size=1, max_size=10))
@settings(max_examples=1000)
@example(indent=8, comment_text='comment')
@example(indent=5, comment_text='x')
def test_comment_preservation_with_code(indent, comment_text):
    code = ' ' * indent + 'code1\n#' + comment_text + '\n' + ' ' * indent + 'code2'

    result = strip_common_indent(code)
    result_lines = result.splitlines()

    assert len(result_lines) == 3, f"Expected 3 lines, got {len(result_lines)}"
    comment_line = result_lines[1]
    assert comment_line.startswith('#'), f"Comment line should start with #, got: {repr(comment_line)}"
    assert comment_text in comment_line, f"Comment text '{comment_text}' not found in {repr(comment_line)}"

if __name__ == "__main__":
    test_comment_preservation_with_code()