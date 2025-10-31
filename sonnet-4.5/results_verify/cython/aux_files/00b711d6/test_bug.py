#!/usr/bin/env python3
"""Test the reported bug about pyrex_ options overriding explicit cython_ parameters."""

from Cython.Distutils import Extension

# Test 1: Simple case with pyrex_gdb=True and explicit cython_gdb=False
print("Test 1: pyrex_gdb=True, cython_gdb=False")
ext = Extension(
    "test",
    ["test.pyx"],
    pyrex_gdb=True,
    cython_gdb=False
)

print(f"Expected: cython_gdb=False (explicit parameter)")
print(f"Actual: cython_gdb={ext.cython_gdb}")

if ext.cython_gdb == False:
    print("PASS: Explicit cython_gdb was preserved")
else:
    print("FAIL: Explicit cython_gdb was overridden by pyrex_gdb")
print()

# Test 2: Opposite case
print("Test 2: pyrex_gdb=False, cython_gdb=True")
ext2 = Extension(
    "test2",
    ["test.pyx"],
    pyrex_gdb=False,
    cython_gdb=True
)

print(f"Expected: cython_gdb=True (explicit parameter)")
print(f"Actual: cython_gdb={ext2.cython_gdb}")

if ext2.cython_gdb == True:
    print("PASS: Explicit cython_gdb was preserved")
else:
    print("FAIL: Explicit cython_gdb was overridden by pyrex_gdb")
print()

# Test 3: Test with include_dirs
print("Test 3: pyrex_include_dirs=['old'], cython_include_dirs=['new']")
ext3 = Extension(
    "test3",
    ["test.pyx"],
    pyrex_include_dirs=['old'],
    cython_include_dirs=['new']
)

print(f"Expected: cython_include_dirs=['new'] (explicit parameter)")
print(f"Actual: cython_include_dirs={ext3.cython_include_dirs}")

if ext3.cython_include_dirs == ['new']:
    print("PASS: Explicit cython_include_dirs was preserved")
else:
    print("FAIL: Explicit cython_include_dirs was overridden by pyrex_include_dirs")
print()

# Test 4: Only pyrex option (should be translated)
print("Test 4: Only pyrex_gdb=True")
ext4 = Extension(
    "test4",
    ["test.pyx"],
    pyrex_gdb=True
)

print(f"Expected: cython_gdb=True (translated from pyrex_gdb)")
print(f"Actual: cython_gdb={ext4.cython_gdb}")

if ext4.cython_gdb == True:
    print("PASS: pyrex_gdb was correctly translated to cython_gdb")
else:
    print("FAIL: pyrex_gdb was not translated correctly")
print()

# Test 5: Only cython option (should work normally)
print("Test 5: Only cython_gdb=True")
ext5 = Extension(
    "test5",
    ["test.pyx"],
    cython_gdb=True
)

print(f"Expected: cython_gdb=True")
print(f"Actual: cython_gdb={ext5.cython_gdb}")

if ext5.cython_gdb == True:
    print("PASS: cython_gdb works normally without pyrex options")
else:
    print("FAIL: cython_gdb doesn't work without pyrex options")