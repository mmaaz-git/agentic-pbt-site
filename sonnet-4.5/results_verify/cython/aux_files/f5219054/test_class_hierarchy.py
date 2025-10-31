import pyximport
import inspect

# Check class hierarchy
print("PyImportMetaFinder base classes:", inspect.getmro(pyximport.PyImportMetaFinder))
print("PyxImportMetaFinder base classes:", inspect.getmro(pyximport.PyxImportMetaFinder))

# Test isinstance relationships
py_finder = pyximport.PyImportMetaFinder()
pyx_finder = pyximport.PyxImportMetaFinder()

print("\npy_finder is instance of PyImportMetaFinder:", isinstance(py_finder, pyximport.PyImportMetaFinder))
print("py_finder is instance of PyxImportMetaFinder:", isinstance(py_finder, pyximport.PyxImportMetaFinder))
print("pyx_finder is instance of PyImportMetaFinder:", isinstance(pyx_finder, pyximport.PyImportMetaFinder))
print("pyx_finder is instance of PyxImportMetaFinder:", isinstance(pyx_finder, pyximport.PyxImportMetaFinder))