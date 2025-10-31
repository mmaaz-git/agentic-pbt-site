from hypothesis import given, strategies as st
import os
import tempfile
import zipfile


@given(st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10), min_size=1, max_size=3))
def test_cython_cache_bug_incorrect_zip_extraction(filenames):
    """
    Test that demonstrates the bug in Cython.Build.Cache.load_from_cache.
    The method incorrectly uses zipfile.extract() by passing a file path
    instead of a directory path as the extraction target.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "cache.zip")
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)

        # Create unique filenames with .c extension
        filenames = [f + ".c" for f in filenames]

        # Create a zip file with test artifacts (simulating Cython's cache)
        with zipfile.ZipFile(zip_path, 'w') as z:
            for fname in filenames:
                z.writestr(fname, f"/* {fname} */")

        # Reproduce the bug: passing file path instead of directory to extract()
        with zipfile.ZipFile(zip_path, 'r') as z:
            for fname in filenames:
                # This is what the buggy Cython code does
                wrong_path = os.path.join(extract_dir, fname)
                z.extract(fname, wrong_path)

                # Verify the bug: file should be at wrong_path but isn't
                assert not os.path.isfile(wrong_path), f"Expected {wrong_path} to not be a file (it's a directory due to the bug)"
                assert os.path.isdir(wrong_path), f"Expected {wrong_path} to be a directory (due to the bug)"

                # The file actually ends up nested one level deeper
                actual_path = os.path.join(wrong_path, fname)
                assert os.path.isfile(actual_path), f"File ended up at {actual_path} instead of {wrong_path}"

                # This wrong behavior breaks Cython's caching
                # The compiler would look for files at wrong_path but find directories


if __name__ == "__main__":
    # Run the test with a simple example
    test_cython_cache_bug_incorrect_zip_extraction(['test'])
    print("Test passed - bug confirmed!")