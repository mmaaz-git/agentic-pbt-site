from hypothesis import given, strategies as st, settings
import sys
import os
import tempfile
import importlib
import pyximport

@given(st.integers(min_value=2, max_value=5))
@settings(max_examples=3)
def test_pyimport_compiles_all_modules(num_modules):
    print(f"\nTesting with {num_modules} modules:")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_modules):
            with open(os.path.join(tmpdir, f'mod{i}.py'), 'w') as f:
                f.write(f'VALUE = {i}')

        sys.path.insert(0, tmpdir)
        try:
            py_imp, pyx_imp = pyximport.install(pyimport=True)

            compiled_count = 0
            module_details = []
            for i in range(num_modules):
                mod = importlib.import_module(f'mod{i}')
                is_compiled = '.so' in mod.__file__ or '.pyd' in mod.__file__
                if is_compiled:
                    compiled_count += 1
                module_details.append(f"  mod{i}: {'compiled' if is_compiled else 'NOT compiled'} ({mod.__file__})")

            print("\n".join(module_details))
            print(f"Result: {compiled_count}/{num_modules} modules compiled")

            assert compiled_count == num_modules, f"Expected all {num_modules} to be compiled, only {compiled_count} were"
        finally:
            sys.path.remove(tmpdir)
            pyximport.uninstall(py_imp, pyx_imp)
            # Clean up imported modules
            for i in range(num_modules):
                mod_name = f'mod{i}'
                if mod_name in sys.modules:
                    del sys.modules[mod_name]

if __name__ == "__main__":
    print("Running hypothesis test for pyximport bug...")
    try:
        test_pyimport_compiles_all_modules()
        print("\n✓ All tests passed - bug not present")
    except AssertionError as e:
        print(f"\n✗ Test failed - bug confirmed: {e}")