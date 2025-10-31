from Cython.Build.Inline import strip_common_indent

code = """        code1
#comment
        code2"""

result = strip_common_indent(code)

print("Input:")
print(code)
print("\nResult:")
print(result)
print("\nResult lines:")
for i, line in enumerate(result.splitlines()):
    print(f"  {i}: {repr(line)}")

expected_lines = ['code1', '#comment', 'code2']
actual_lines = result.splitlines()

print(f"\nExpected line 1: {repr(expected_lines[1])}")
print(f"Actual line 1: {repr(actual_lines[1])}")

assert actual_lines[1] == '#comment', f"Expected '#comment' but got {repr(actual_lines[1])}"