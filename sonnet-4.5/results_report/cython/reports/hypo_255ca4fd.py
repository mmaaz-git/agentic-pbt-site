from hypothesis import given, strategies as st, settings, assume
import sys
import pyximport

@given(
    pyximport_flag=st.booleans(),
    pyimport_flag=st.booleans(),
    num_calls=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_install_idempotence(pyximport_flag, pyimport_flag, num_calls):
    assume(pyximport_flag or pyimport_flag)

    original_meta_path = sys.meta_path.copy()

    try:
        for _ in range(num_calls):
            pyximport.install(
                pyximport=pyximport_flag,
                pyimport=pyimport_flag
            )

        py_importers = [imp for imp in sys.meta_path
                       if isinstance(imp, pyximport.PyImportMetaFinder)]

        if pyimport_flag:
            assert len(py_importers) == 1, \
                f"Expected 1 py importer, got {len(py_importers)}"
    finally:
        sys.meta_path = original_meta_path

if __name__ == "__main__":
    test_install_idempotence()