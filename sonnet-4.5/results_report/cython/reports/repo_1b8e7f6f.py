from distutils.dist import Distribution
from unittest.mock import Mock
from Cython.Distutils import build_ext

# Test case: Builder value of 0 should take precedence over extension value of 1
dist = Distribution()
builder = build_ext(dist)
builder.cython_cplus = 0  # Explicitly set to 0 (falsy but valid)

ext = Mock()
ext.cython_cplus = 1  # Extension has value of 1

result = builder.get_extension_attr(ext, 'cython_cplus')
print(f"Test 1: builder=0, extension=1")
print(f"  Expected: 0 (builder value should take precedence)")
print(f"  Actual: {result}")
print(f"  Bug present: {result != 0}")
print()

# Test case 2: Builder value of False should take precedence
builder.cython_gdb = False  # Explicitly set to False
ext.cython_gdb = True
result = builder.get_extension_attr(ext, 'cython_gdb')
print(f"Test 2: builder=False, extension=True")
print(f"  Expected: False (builder value should take precedence)")
print(f"  Actual: {result}")
print(f"  Bug present: {result != False}")
print()

# Test case 3: Empty list should take precedence
builder.cython_directives = []  # Empty list (falsy but valid)
ext.cython_directives = {'language_level': 3}
result = builder.get_extension_attr(ext, 'cython_directives')
print(f"Test 3: builder=[], extension={{'language_level': 3}}")
print(f"  Expected: [] (builder value should take precedence)")
print(f"  Actual: {result}")
print(f"  Bug present: {result != []}")