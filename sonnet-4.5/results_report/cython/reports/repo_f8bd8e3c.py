import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import _have_importers, PyImportMetaFinder

# Save original state
original_meta_path = sys.meta_path.copy()

try:
    # Clear sys.meta_path and add only PyImportMetaFinder
    sys.meta_path = []
    sys.meta_path.append(PyImportMetaFinder())

    # Call _have_importers() to check detection
    has_py, has_pyx = _have_importers()

    print(f"Result from _have_importers(): has_py={has_py}, has_pyx={has_pyx}")
    print(f"Expected: has_py=True, has_pyx=False")

    # Verify this is actually a bug
    if has_py != True:
        print(f"\n✗ BUG DETECTED: PyImportMetaFinder is in sys.meta_path but has_py={has_py}")
        print(f"  The function failed to detect PyImportMetaFinder")
    else:
        print("\n✓ No bug detected")

finally:
    # Restore original state
    sys.meta_path[:] = original_meta_path