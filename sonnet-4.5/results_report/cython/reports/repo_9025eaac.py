import sys
import pyximport

# First install with pyimport=True
print("Initial sys.meta_path length:", len(sys.meta_path))
py1, pyx1 = pyximport.install(pyximport=False, pyimport=True)

# Count PyImportMetaFinder instances after first install
py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))
print(f"PyImportMetaFinder instances after first install: {py_count}")

# Second install with same parameters
py2, pyx2 = pyximport.install(pyximport=False, pyimport=True)

# Count PyImportMetaFinder instances after second install
py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))
print(f"PyImportMetaFinder instances after second install: {py_count}")

# This demonstrates the bug - we should have at most 1, but we have 2
if py_count > 1:
    print(f"BUG: Found {py_count} PyImportMetaFinder instances, expected <= 1")
else:
    print("No bug detected")

# Cleanup
pyximport.uninstall(py1, pyx1)
pyximport.uninstall(py2, pyx2)