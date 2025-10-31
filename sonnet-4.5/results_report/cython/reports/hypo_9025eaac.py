import sys
import copy
from hypothesis import given, settings, strategies as st
import pyximport

@settings(max_examples=50)
@given(
    pyximport_flag=st.booleans(),
    pyimport_flag=st.booleans()
)
def test_install_twice_no_duplicates(pyximport_flag, pyimport_flag):
    original_meta_path = copy.copy(sys.meta_path)

    try:
        py1, pyx1 = pyximport.install(pyximport=pyximport_flag, pyimport=pyimport_flag)
        meta_path_after_first = copy.copy(sys.meta_path)

        py2, pyx2 = pyximport.install(pyximport=pyximport_flag, pyimport=pyimport_flag)
        meta_path_after_second = copy.copy(sys.meta_path)

        pyx_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyxImportMetaFinder))
        py_count = sum(1 for item in sys.meta_path if isinstance(item, pyximport.PyImportMetaFinder))

        if pyximport_flag:
            assert pyx_count <= 1, f"Found {pyx_count} PyxImportMetaFinder instances, expected <= 1"
        if pyimport_flag:
            assert py_count <= 1, f"Found {py_count} PyImportMetaFinder instances, expected <= 1"
    finally:
        pyximport.uninstall(py1, pyx1)
        try:
            pyximport.uninstall(py2, pyx2)
        except:
            pass
        sys.meta_path[:] = original_meta_path

if __name__ == "__main__":
    test_install_twice_no_duplicates()