import sys
import pyximport

# Clear sys.meta_path first to ensure clean state
original_meta_path = sys.meta_path.copy()

print("Initial sys.meta_path length:", len(sys.meta_path))

py1, pyx1 = pyximport.install(pyximport=False, pyimport=True)
print("After first install:")
py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))
print(f"  PyImportMetaFinder instances in sys.meta_path: {py_count}")

py2, pyx2 = pyximport.install(pyximport=False, pyimport=True)
print("After second install:")
py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))
print(f"  PyImportMetaFinder instances in sys.meta_path: {py_count}")

# Cleanup
pyximport.uninstall(py1, pyx1)
pyximport.uninstall(py2, pyx2)
sys.meta_path[:] = original_meta_path

print("\nThis demonstrates the bug: calling install() twice with pyimport=True")
print("adds duplicate PyImportMetaFinder instances to sys.meta_path.")