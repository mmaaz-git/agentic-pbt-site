from Cython.Build.Inline import strip_common_indent

test_input = """    x = 1
  # comment
    y = 2"""

result = strip_common_indent(test_input)

print("Input:")
print(repr(test_input))
print("\nOutput:")
print(repr(result))
print("\n--- Formatted Input ---")
print(test_input)
print("\n--- Formatted Output ---")
print(result)