from hypothesis import given, strategies as st, example, settings
from Cython.Build.Inline import strip_common_indent

# Test with various combinations of comments and blanks
@given(st.lists(st.sampled_from(['#comment', '  #comment', '', '  '])))
@example(['#comment'])
@example(['  #comment'])
@example([''])
@settings(max_examples=10, deadline=None)
def test_strip_common_indent_only_comments_and_blanks(lines):
    code = '\n'.join(lines)
    try:
        result = strip_common_indent(code)
        print(f"Input: {repr(code)}")
        print(f"Output: {repr(result)}")
    except Exception as e:
        print(f"Exception for input {repr(code)}: {e}")
        raise

# Test the specific failing case from the bug report
def test_specific_failing_case():
    print("\n=== Testing specific failing case ===")
    code = """  x = 1
    y = 2
 #comment
  z = 3"""

    print(f"Input code:\n{repr(code)}\n")
    result = strip_common_indent(code)
    print(f"Result:\n{repr(result)}\n")

    result_lines = result.splitlines()
    comment_line = result_lines[2]

    print(f"Comment line: {repr(comment_line)}")
    print(f"Expected: {repr(' #comment')}")

    try:
        assert comment_line == ' #comment'
        print("✓ Test passed")
    except AssertionError:
        print(f"✗ Test FAILED: Comment line was incorrectly modified")
        print(f"  Got: {repr(comment_line)}")
        print(f"  Expected: ' #comment' (with leading space preserved)")
        raise

if __name__ == "__main__":
    print("Running property-based tests...")
    try:
        test_strip_common_indent_only_comments_and_blanks()
        print("\nAll property-based tests passed.\n")
    except Exception as e:
        print(f"Property-based test failed: {e}\n")

    try:
        test_specific_failing_case()
    except AssertionError:
        print("\nThe specific test case demonstrates the bug.")