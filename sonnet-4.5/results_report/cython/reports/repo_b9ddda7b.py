import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Code import _indent_chunk

# Test single character without newline
result = _indent_chunk('0', 0)
print(f"_indent_chunk('0', 0) = {repr(result)}")
print(f"Expected: '0', Got: {repr(result)}")
print(f"Content lost: {result != '0'}")
print()

# Test other single characters
print("Other single character tests:")
for char in ['a', 'x', '9', '1', 'b', 'z']:
    result = _indent_chunk(char, 0)
    print(f"_indent_chunk('{char}', 0) = {repr(result)} (expected: '{char}')")

print()

# Test with different indentation levels
print("Single character with various indentation:")
for indent in [0, 1, 4, 8]:
    result = _indent_chunk('0', indent)
    expected = ' ' * indent + '0'
    print(f"_indent_chunk('0', {indent}) = {repr(result)} (expected: {repr(expected)})")

print()

# Show that multi-character strings work correctly
print("Multi-character strings (working correctly):")
for s in ['ab', 'abc', '12', '123']:
    result = _indent_chunk(s, 0)
    print(f"_indent_chunk('{s}', 0) = {repr(result)} (expected: '{s}')")