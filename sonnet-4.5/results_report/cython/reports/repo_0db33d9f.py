from Cython.Build.Inline import strip_common_indent

test_input = """  x = 1
# comment
  y = 2"""

print("Input:")
print(repr(test_input))
print("\nFormatted Input:")
print(test_input)

result = strip_common_indent(test_input)

print("\nOutput:")
print(repr(result))
print("\nFormatted Output:")
print(result)

print("\n--- Analysis ---")
for i, (inp_line, out_line) in enumerate(zip(test_input.split('\n'), result.split('\n')), 1):
    print(f"Line {i}:")
    print(f"  Input:  {repr(inp_line)}")
    print(f"  Output: {repr(out_line)}")
    if inp_line.lstrip().startswith('#'):
        if not out_line.lstrip().startswith('#'):
            print(f"  ERROR: Comment line lost '#' character!")