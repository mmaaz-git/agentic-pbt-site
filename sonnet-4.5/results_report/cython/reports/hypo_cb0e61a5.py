from hypothesis import given, strategies as st
import sys
import os
import tempfile
import importlib
import pyximport

@given(st.integers(min_value=2, max_value=5))
def test_pyimport_compiles_all_modules(num_modules):
    """Test that pyximport with pyimport=True compiles all .py modules, not just the first one"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create num_modules simple Python modules
        for i in range(num_modules):
            with open(os.path.join(tmpdir, f'mod{i}.py'), 'w') as f:
                f.write(f'VALUE = {i}')

        # Add tmpdir to path
        sys.path.insert(0, tmpdir)
        try:
            # Install pyximport with pyimport=True
            py_imp, pyx_imp = pyximport.install(pyimport=True)

            # Import all modules and check if they were compiled
            compiled_count = 0
            for i in range(num_modules):
                mod = importlib.import_module(f'mod{i}')
                # Check if module was compiled (has .so or .pyd extension)
                if '.so' in mod.__file__ or '.pyd' in mod.__file__:
                    compiled_count += 1

            # All modules should be compiled
            assert compiled_count == num_modules, f"Expected all {num_modules} modules to be compiled, only {compiled_count} were"
        finally:
            # Cleanup
            sys.path.remove(tmpdir)
            pyximport.uninstall(py_imp, pyx_imp)

if __name__ == "__main__":
    # Run the test
    test_pyimport_compiles_all_modules()