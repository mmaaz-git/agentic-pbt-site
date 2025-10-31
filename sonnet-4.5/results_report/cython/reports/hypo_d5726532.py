from hypothesis import given, strategies as st, settings
from Cython.Build.Cache import Cache
import tempfile
import zipfile
import os

@given(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=20))
@settings(max_examples=100)
def test_cache_load_extracts_to_correct_location(artifact_name):
    artifact_name = artifact_name + '.c'
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir)
        zip_path = os.path.join(tmpdir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w') as z:
            z.writestr(artifact_name, 'content')

        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(output_dir)
        c_file = os.path.join(output_dir, 'test.c')

        cache.load_from_cache(c_file, zip_path)

        expected = os.path.join(output_dir, artifact_name)
        # Check if it's a file, not a directory
        assert os.path.isfile(expected), f'File should be at {expected}, but found directory={os.path.isdir(expected)}, file={os.path.isfile(expected)}'

if __name__ == "__main__":
    test_cache_load_extracts_to_correct_location()