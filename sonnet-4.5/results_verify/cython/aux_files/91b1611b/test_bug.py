from distutils.dist import Distribution
from unittest.mock import Mock
from Cython.Distutils import build_ext

# Test Case 1: Basic falsy value test
dist = Distribution()
builder = build_ext(dist)
builder.cython_cplus = 0  # Explicitly setting to 0

ext = Mock()
ext.cython_cplus = 1  # Extension has 1

result = builder.get_extension_attr(ext, 'cython_cplus')
print(f"Test 1 - Builder=0, Extension=1: Result: {result}")
print(f"Expected: 0, Got: {result}, Match: {result == 0}")

# Test Case 2: With False
builder2 = build_ext(dist)
builder2.cython_cplus = False  # Explicitly setting to False

ext2 = Mock()
ext2.cython_cplus = True  # Extension has True

result2 = builder2.get_extension_attr(ext2, 'cython_cplus')
print(f"\nTest 2 - Builder=False, Extension=True: Result: {result2}")
print(f"Expected: False, Got: {result2}, Match: {result2 == False}")

# Test Case 3: With empty list
builder3 = build_ext(dist)
builder3.cython_directives = []  # Explicitly setting to empty list

ext3 = Mock()
ext3.cython_directives = {'language_level': 3}  # Extension has directives

result3 = builder3.get_extension_attr(ext3, 'cython_directives')
print(f"\nTest 3 - Builder=[], Extension=dict: Result: {result3}")
print(f"Expected: [], Got: {result3}, Match: {result3 == []}")

# Test Case 4: With None (should use extension's value)
builder4 = build_ext(dist)
builder4.test_attr = None  # Setting to None

ext4 = Mock()
ext4.test_attr = "extension_value"

result4 = builder4.get_extension_attr(ext4, 'test_attr')
print(f"\nTest 4 - Builder=None, Extension='extension_value': Result: {result4}")
print(f"Expected behavior: should return 'extension_value' since None is falsy")

# Test Case 5: With truthy builder value (should work correctly)
builder5 = build_ext(dist)
builder5.cython_cplus = 1  # Truthy value

ext5 = Mock()
ext5.cython_cplus = 0  # Extension has 0

result5 = builder5.get_extension_attr(ext5, 'cython_cplus')
print(f"\nTest 5 - Builder=1, Extension=0: Result: {result5}")
print(f"Expected: 1, Got: {result5}, Match: {result5 == 1}")