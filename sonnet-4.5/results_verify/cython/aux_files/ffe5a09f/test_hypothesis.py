from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import CythonDebugWriter
import tempfile


@given(st.integers(min_value=0, max_value=100))
def test_invalid_tag_names_should_not_crash(n):
    tag_name = f'.{n}'
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CythonDebugWriter(tmpdir)
        writer.module_name = "test_module"
        writer.start(tag_name)
        writer.end(tag_name)

if __name__ == "__main__":
    # Test with the specific failing input mentioned in the report
    print("Testing with n=0 (tag name '.0')...")
    n = 0
    tag_name = f'.{n}'
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CythonDebugWriter(tmpdir)
            writer.module_name = "test_module"
            writer.start(tag_name)
            writer.end(tag_name)
        print("Test passed without crash")
    except Exception as e:
        print(f"Test failed with error: {type(e).__name__}: {e}")