from Cython.Distutils import Extension

# Test case: Both pyrex_gdb and cython_gdb provided
# Expected: cython_gdb=False (explicit parameter should take precedence)
# Actual: cython_gdb will be True (pyrex_gdb value overrides)

ext = Extension(
    "test",
    ["test.pyx"],
    pyrex_gdb=True,
    cython_gdb=False
)

print(f"Expected: cython_gdb=False (explicit parameter)")
print(f"Actual: cython_gdb={ext.cython_gdb}")

# This assertion will fail, demonstrating the bug
try:
    assert ext.cython_gdb == False, f"Bug: explicit cython_gdb=False was overridden by pyrex_gdb=True"
    print("✓ Test passed: Explicit cython_gdb parameter was respected")
except AssertionError as e:
    print(f"✗ Test failed: {e}")