from hypothesis import given, strategies as st, settings, assume
from Cython.Build.Inline import strip_common_indent


@st.composite
def code_with_comment_at_lower_indent(draw):
    code_indent = draw(st.integers(min_value=2, max_value=10))
    comment_indent = draw(st.integers(min_value=0, max_value=code_indent-1))
    assume(comment_indent < code_indent)

    return (
        ' ' * code_indent + 'x = 1\n' +
        ' ' * comment_indent + '# comment\n' +
        ' ' * code_indent + 'y = 2'
    )


@given(code_with_comment_at_lower_indent())
@settings(max_examples=500)
def test_strip_common_indent_preserves_comment_hash(code):
    result = strip_common_indent(code)
    for inp_line, out_line in zip(code.split('\n'), result.split('\n')):
        if inp_line.lstrip().startswith('#'):
            assert out_line.lstrip().startswith('#'), f"Comment line lost '#' character!\nInput: {repr(code)}\nOutput: {repr(result)}"


if __name__ == "__main__":
    test_strip_common_indent_preserves_comment_hash()