"""Test how parse_list is used in the real context"""

# Simulate what happens when parsing a directive line
source = """
# distutils: libraries = foo bar # comment about libraries
# distutils: include_dirs = /path/to/include # another comment
"""

# This simulates what DistutilsSettings does
from Cython.Build.Dependencies import parse_list
from io import StringIO

values = {}
source_lines = StringIO(source)

for line in source_lines:
    line = line.lstrip()
    if not line:
        continue
    if line[0] != '#':
        break
    line = line[1:].lstrip()  # Remove the initial #

    if line.startswith("distutils:"):
        key, _, value = [s.strip() for s in line[len("distutils:"):].partition('=')]
        print(f"Parsing directive: {key} = '{value}'")

        # This is what happens in the actual code
        parsed = parse_list(value)
        print(f"  Result: {parsed}")
        print()

# Show what we'd expect vs what we get
print("Expected behavior:")
print("  libraries = ['foo', 'bar']")
print("  include_dirs = ['/path/to/include']")
print()
print("But with comments in the value, we get placeholder tokens like '#__Pyx_L1_'")