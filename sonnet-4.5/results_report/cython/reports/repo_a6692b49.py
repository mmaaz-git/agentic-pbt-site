from Cython.Build.Inline import strip_common_indent

# Test case that demonstrates the bug
code = """  x = 1
    y = 2
 #comment
  z = 3"""

print("Input code:")
print(repr(code))
print()

result = strip_common_indent(code)

print("Result:")
print(repr(result))
print()

result_lines = result.splitlines()
print("Result lines:")
for i, line in enumerate(result_lines):
    print(f"Line {i}: {repr(line)}")
print()

# Check the comment line
comment_line = result_lines[2]
print(f"Comment line (index 2): {repr(comment_line)}")
print(f"Expected: {repr(' #comment')}")

# This should fail - the comment's leading space is incorrectly stripped
try:
    assert comment_line == ' #comment', f"Expected ' #comment' but got {repr(comment_line)}"
    print("✓ Assertion passed (unexpected!)")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")