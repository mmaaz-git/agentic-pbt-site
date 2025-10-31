#!/usr/bin/env python3
"""Test the fixed version of strip_common_indent."""

import re

_find_non_space = re.compile(r'\S').search

def strip_common_indent_fixed(code):
    min_indent = None
    lines = code.splitlines()
    for line in lines:
        match = _find_non_space(line)
        if not match:
            continue  # blank
        indent = match.start()
        if line[indent] == '#':
            continue  # comment
        if min_indent is None or min_indent > indent:
            min_indent = indent
    for ix, line in enumerate(lines):
        match = _find_non_space(line)
        # Fixed version: check if line starts with # using match.start()
        if not match or not line or (match and line[match.start()] == '#'):
            continue
        lines[ix] = line[min_indent:]
    return '\n'.join(lines)

# Test the fixed version
code1 = """        code1
#comment
        code2"""

result1 = strip_common_indent_fixed(code1)

print("=== Test 1: Basic reproduction with fixed version ===")
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
print("\n=== Test 2: Property test example with fixed version ===")
indent = 8
comment_text = 'comment'
code2 = ' ' * indent + 'code1\n#' + comment_text + '\n' + ' ' * indent + 'code2'

print(f"Input code:")
print(repr(code2))

result2 = strip_common_indent_fixed(code2)
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


# Test with different indentations
print("\n=== Test 3: Various indentation levels ===")
test_cases = [
    ("    code\n#comment\n    code2", ['code', '#comment', 'code2']),
    ("  x\n#y\n  z", ['x', '#y', 'z']),
    ("no_indent\n#comment\nno_indent2", ['no_indent', '#comment', 'no_indent2']),
]

for code, expected in test_cases:
    result = strip_common_indent_fixed(code)
    result_lines = result.splitlines()
    if result_lines == expected:
        print(f"PASS: {repr(code[:20])}...")
    else:
        print(f"FAIL: {repr(code[:20])}...")
        print(f"  Expected: {expected}")
        print(f"  Got: {result_lines}")