#!/usr/bin/env python3
"""Analyze the exact behavior of the Extension class."""

from Cython.Distutils import Extension

# Let's trace what happens when both pyrex_ and cython_ options are provided
print("Analyzing the source code behavior:")
print("=" * 50)

print("\nWhen Extension is called with pyrex_gdb=True and cython_gdb=False:")
print("1. The __init__ method receives:")
print("   - cython_gdb=False as an explicit parameter")
print("   - pyrex_gdb=True in **kw")
print()
print("2. Lines 42-45 convert pyrex_gdb to cython_gdb in kw:")
print("   - kw['cython_gdb'] = kw.pop('pyrex_gdb')  # kw['cython_gdb'] = True")
print()
print("3. Because had_pyrex_options is True, the code calls Extension.__init__ recursively (lines 47-63)")
print("4. The recursive call passes:")
print("   - All the standard parameters like include_dirs, define_macros, etc.")
print("   - **kw which now contains cython_gdb=True (from the converted pyrex_gdb)")
print("   - BUT it does NOT pass the explicit cython_gdb parameter!")
print()
print("5. The recursive call then returns early (line 64)")
print("6. Lines 83-92 (setting self.cython_gdb = cython_gdb) are never reached!")
print()
print("Result: The explicit cython_gdb=False is lost, replaced by the converted pyrex_gdb=True")
print()
print("This is clearly a bug in the implementation!")

# Let's verify our understanding with a test
print("\n" + "=" * 50)
print("Verification test:")

ext = Extension(
    "test",
    ["test.pyx"],
    pyrex_gdb=True,
    cython_gdb=False
)

print(f"With pyrex_gdb=True and cython_gdb=False:")
print(f"  ext.cython_gdb = {ext.cython_gdb}")
print(f"  Expected: False (the explicit cython_gdb value)")
print(f"  Actual: {ext.cython_gdb}")
print(f"  Bug confirmed: {ext.cython_gdb != False}")