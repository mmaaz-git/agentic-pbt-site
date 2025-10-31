from hypothesis import given, strategies as st
from Cython.Distutils import build_ext
from distutils.dist import Distribution


@given(st.just(''))
def test_finalize_options_empty_string(empty_str):
    dist = Distribution()
    builder = build_ext(dist)
    builder.initialize_options()
    builder.cython_include_dirs = empty_str
    builder.finalize_options()

    print(f"cython_include_dirs after finalize_options: {builder.cython_include_dirs}")
    print(f"Expected: []")
    print(f"Actual type: {type(builder.cython_include_dirs)}")

    assert builder.cython_include_dirs == [], f"Expected [], got {builder.cython_include_dirs}"

if __name__ == "__main__":
    # Direct test without hypothesis
    dist = Distribution()
    builder = build_ext(dist)
    builder.initialize_options()
    builder.cython_include_dirs = ''
    builder.finalize_options()

    print(f"cython_include_dirs after finalize_options: {builder.cython_include_dirs}")
    print(f"Expected: []")
    print(f"Actual type: {type(builder.cython_include_dirs)}")

    if builder.cython_include_dirs == []:
        print("Test PASSED")
    else:
        print(f"Test FAILED: Expected [], got {builder.cython_include_dirs}")