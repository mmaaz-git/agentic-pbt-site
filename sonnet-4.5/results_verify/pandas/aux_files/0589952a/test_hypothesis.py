from hypothesis import given, strategies as st, settings, assume
from Cython.Build.Inline import strip_common_indent


@st.composite
def code_with_comments_at_different_indents(draw):
    code_indent = draw(st.integers(min_value=2, max_value=10))
    comment_indent = draw(st.integers(min_value=0, max_value=10))
    assume(comment_indent < code_indent)

    return (
        ' ' * code_indent + 'x = 1\n' +
        ' ' * comment_indent + '# comment\n' +
        ' ' * code_indent + 'y = 2'
    )


@given(code_with_comments_at_different_indents())
@settings(max_examples=500)
def test_strip_common_indent_preserves_comment_markers(code):
    """
    Property: Comment lines should retain their '#' marker.
    Evidence: Comments are syntactically significant in Python
    """
    result = strip_common_indent(code)

    for line in result.split('\n'):
        stripped_line = line.strip()
        if stripped_line and 'comment' in stripped_line:
            assert stripped_line.startswith('#'), \
                f"Comment line lost '#' marker: {line!r}"

if __name__ == "__main__":
    test_strip_common_indent_preserves_comment_markers()
    print("Test passed!")