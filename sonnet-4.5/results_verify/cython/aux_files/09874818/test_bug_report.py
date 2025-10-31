#!/usr/bin/env python3
"""Test the reported bug in strip_common_indent function."""

from Cython.Build.Inline import strip_common_indent

# First test case from the bug report
code1 = """        code1
#comment
        code2"""

result1 = strip_common_indent(code1)

print("=== Test 1: Basic reproduction ===")
print("Input:")
print(repr(code1))
print("\nResult:")
print(repr(result1))
print("\nResult lines:")
for i, line in enumerate(result1.splitlines()):
    print(f"  {i}: {repr(line)}")

expected_lines = ['code1', '#comment', 'code2']
actual_lines = result1.splitlines()

print(f"\nExpected line 1: {repr('#comment')}")
print(f"Actual line 1: {repr(actual_lines[1] if len(actual_lines) > 1 else 'NO LINE')}")

try:
    assert actual_lines[1] == '#comment'
    print("PASS: Comment preserved correctly")
except AssertionError:
    print(f"FAIL: Comment was not preserved, got: {repr(actual_lines[1])}")


# Test using the property test values
print("\n=== Test 2: Property test example ===")
indent = 8
comment_text = 'comment'
code2 = ' ' * indent + 'code1\n#' + comment_text + '\n' + ' ' * indent + 'code2'

print(f"Input code:")
print(repr(code2))

result2 = strip_common_indent(code2)
result_lines = result2.splitlines()

print(f"\nResult:")
print(repr(result2))
print(f"\nResult lines:")
for i, line in enumerate(result_lines):
    print(f"  {i}: {repr(line)}")

try:
    assert len(result_lines) == 3
    comment_line = result_lines[1]
    assert comment_line.startswith('#'), f"Comment line should start with #, got: {repr(comment_line)}"
    assert comment_text in comment_line
    print("PASS: Property test passed")
except AssertionError as e:
    print(f"FAIL: Property test failed - {e}")

# Let's trace through the function logic to understand the bug
print("\n=== Debug trace of the function ===")
code3 = """        code1
#comment
        code2"""

lines = code3.splitlines()
print(f"Lines: {lines}")

# First loop logic
min_indent = None
_find_non_space = __import__('re').compile(r'\S').search

for line in lines:
    match = _find_non_space(line)
    if not match:
        continue  # blank
    indent = match.start()
    if line[indent] == '#':
        print(f"Skipping comment line: {repr(line)}, indent would be {indent}")
        continue  # comment
    if min_indent is None or min_indent > indent:
        min_indent = indent
        print(f"Updated min_indent to {min_indent} from line: {repr(line)}")

print(f"After first loop: min_indent={min_indent}, indent={indent}")

# Second loop logic
for ix, line in enumerate(lines):
    match = _find_non_space(line)
    print(f"Line {ix}: {repr(line)}")
    print(f"  match: {match}")
    if match:
        print(f"  match.start(): {match.start()}")
    print(f"  Using indent value: {indent}")
    print(f"  line[indent:indent+1]: {repr(line[indent:indent+1]) if len(line) > indent else 'OUT OF BOUNDS'}")

    # This is the bug - using stale 'indent' variable
    if not match or not line or line[indent:indent+1] == '#':
        print(f"  -> Skipping this line")
        continue
    print(f"  -> Would process: line[{min_indent}:] = {repr(line[min_indent:])}")