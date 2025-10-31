"""Test to reproduce the reported bug about option precedence in old_build_ext"""

# First test case from the bug report (property-based test simplified)
from Cython.Distutils import Extension
from Cython.Distutils.old_build_ext import old_build_ext
from distutils.dist import Distribution

print("=" * 60)
print("Test 1: Testing option precedence with 'or' operator")
print("=" * 60)

# Test each affected option
test_options = ['cython_cplus', 'cython_gen_pxi']

for option_name in test_options:
    print(f"\nTesting option: {option_name}")

    dist = Distribution()
    builder = old_build_ext(dist)
    builder.initialize_options()
    builder.finalize_options()

    # Set command-level to 1
    setattr(builder, option_name, 1)
    ext = Extension('test', ['test.pyx'])
    # Set extension-level to 0
    setattr(ext, option_name, 0)

    # This is how the code actually works (using 'or')
    option_value = getattr(builder, option_name) or getattr(ext, option_name, 0)

    print(f"  Command-level (builder): {getattr(builder, option_name)}")
    print(f"  Extension-level: {getattr(ext, option_name)}")
    print(f"  Computed value (using 'or'): {option_value}")
    print(f"  Expected if command-level has precedence: {getattr(builder, option_name)}")

    # The bug report expects command-level to always win
    # But with 'or', command-level = 1 will win (1 or 0 = 1)
    assert option_value == 1, f"Expected 1, got {option_value}"
    print(f"  Result: PASSED (command-level wins when it's 1)")

print("\n" + "=" * 60)
print("Test 2: Reproducing the actual bug (command=0, extension=1)")
print("=" * 60)

dist = Distribution()
builder = old_build_ext(dist)
builder.initialize_options()

# This is the key test case: command-level = 0, extension-level = 1
builder.cython_gen_pxi = 0
ext = Extension('test', ['test.pyx'], cython_gen_pxi=1)

# This is how it's computed in the actual code (line 231)
computed = builder.cython_gen_pxi or getattr(ext, 'cython_gen_pxi', 0)

print(f"\nCommand-level (builder): {builder.cython_gen_pxi}")
print(f"Extension-level: {ext.cython_gen_pxi}")
print(f"Computed value: {computed}")
print(f"Expected if command-level has precedence: 0")
print(f"Actual: {computed}")

if computed == 0:
    print("Result: Command-level (0) correctly overrides extension-level (1)")
else:
    print("Result: Extension-level (1) incorrectly overrides command-level (0)")
    print("This demonstrates the bug!")

print("\n" + "=" * 60)
print("Test 3: Examining actual code from old_build_ext.py lines 223-234")
print("=" * 60)

# Let's look at how it's actually used in the cython_sources method
import os

dist = Distribution()
builder = old_build_ext(dist)
builder.initialize_options()
builder.finalize_options()

# Set various command-level options to 0
builder.cython_create_listing = 0
builder.cython_line_directives = 0
builder.no_c_in_traceback = 0
builder.cython_cplus = 0
builder.cython_gen_pxi = 0
builder.cython_gdb = False

# Create extension with all options set to 1 or True
ext = Extension('test', ['test.pyx'])
ext.cython_create_listing = 1
ext.cython_line_directives = 1
ext.no_c_in_traceback = 1
ext.cython_cplus = 1
ext.cython_gen_pxi = 1
ext.cython_gdb = True
ext.language = None  # For cplus test

# Now test how each option is computed (following the actual code pattern)
print("\nActual code patterns from old_build_ext.py:")

# Line 223-224
create_listing = builder.cython_create_listing or getattr(ext, 'cython_create_listing', 0)
print(f"create_listing: command={builder.cython_create_listing}, ext={ext.cython_create_listing}, computed={create_listing}")

# Line 225-226
line_directives = builder.cython_line_directives or getattr(ext, 'cython_line_directives', 0)
print(f"line_directives: command={builder.cython_line_directives}, ext={ext.cython_line_directives}, computed={line_directives}")

# Line 227-228
no_c_in_traceback = builder.no_c_in_traceback or getattr(ext, 'no_c_in_traceback', 0)
print(f"no_c_in_traceback: command={builder.no_c_in_traceback}, ext={ext.no_c_in_traceback}, computed={no_c_in_traceback}")

# Line 229-230
cplus = builder.cython_cplus or getattr(ext, 'cython_cplus', 0) or (ext.language and ext.language.lower() == 'c++')
print(f"cplus: command={builder.cython_cplus}, ext={ext.cython_cplus}, computed={cplus}")

# Line 231
cython_gen_pxi = builder.cython_gen_pxi or getattr(ext, 'cython_gen_pxi', 0)
print(f"cython_gen_pxi: command={builder.cython_gen_pxi}, ext={ext.cython_gen_pxi}, computed={cython_gen_pxi}")

# Line 232
cython_gdb = builder.cython_gdb or getattr(ext, 'cython_gdb', False)
print(f"cython_gdb: command={builder.cython_gdb}, ext={ext.cython_gdb}, computed={cython_gdb}")

print("\nConclusion: In all cases, extension-level=1 wins over command-level=0")
print("This is because of the 'or' operator: (0 or 1) = 1")

print("\n" + "=" * 60)
print("Test 4: Checking modern build_ext.get_extension_attr")
print("=" * 60)

from Cython.Distutils.build_ext import build_ext

dist2 = Distribution()
modern_builder = build_ext(dist2)
modern_builder.initialize_options()
modern_builder.finalize_options()

# Set command-level to 0
modern_builder.cython_gen_pxi = 0

# Create extension with value 1
ext2 = Extension('test', ['test.pyx'])
ext2.cython_gen_pxi = 1

# Check how modern build_ext handles it (line 81 of build_ext.py)
# get_extension_attr: return getattr(self, option_name) or getattr(extension, option_name, default)
result = modern_builder.get_extension_attr(ext2, 'cython_gen_pxi', default=False)

print(f"Modern build_ext.get_extension_attr:")
print(f"  Command-level: {modern_builder.cython_gen_pxi}")
print(f"  Extension-level: {ext2.cython_gen_pxi}")
print(f"  Result: {result}")
print(f"  Same issue? {result == 1} (extension wins over command)")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("The bug report is correct that command-level=0 cannot override extension-level=1")
print("This is due to the use of 'or' operator in both old_build_ext and modern build_ext")
print("However, the module is deprecated as noted on line 6 of old_build_ext.py")