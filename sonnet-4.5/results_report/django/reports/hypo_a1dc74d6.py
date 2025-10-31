import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        USE_TZ=True,
        MEDIA_ROOT='/tmp/test_media',
        MEDIA_URL='/media/',
        FILE_UPLOAD_PERMISSIONS=None,
        FILE_UPLOAD_DIRECTORY_PERMISSIONS=None,
    )
    django.setup()

from hypothesis import given, strategies as st, settings as hypo_settings
from django.core.files.storage import InMemoryStorage, FileSystemStorage
from django.core.files.base import ContentFile
import tempfile
import shutil

@given(
    filename=st.text(min_size=1, max_size=50).filter(lambda x: '/' not in x and '\\' not in x and '..' not in x and x not in ['.', '..']),
    content_bytes=st.binary(min_size=0, max_size=1000)
)
@hypo_settings(max_examples=100)
def test_inmemory_vs_filesystem_equivalence(filename, content_bytes):
    """Test that InMemoryStorage and FileSystemStorage handle filenames consistently."""
    temp_dir = tempfile.mkdtemp()
    try:
        mem_storage = InMemoryStorage()
        fs_storage = FileSystemStorage(location=temp_dir)

        mem_content = ContentFile(content_bytes)
        fs_content = ContentFile(content_bytes)

        # Both should either accept or reject the filename
        mem_error = None
        fs_error = None

        try:
            mem_name = mem_storage.save(filename, mem_content)
            mem_exists = mem_storage.exists(mem_name)
        except Exception as e:
            mem_error = type(e).__name__
            mem_exists = False

        try:
            fs_name = fs_storage.save(filename, fs_content)
            fs_exists = fs_storage.exists(fs_name)
        except Exception as e:
            fs_error = type(e).__name__
            fs_exists = False

        # Check if behavior is consistent
        if mem_error != fs_error:
            print(f"\nInconsistent behavior detected!")
            print(f"Filename: {repr(filename)}")
            print(f"InMemoryStorage: {mem_error or 'Accepted'}")
            print(f"FileSystemStorage: {fs_error or 'Accepted'}")
            assert False, f"Storage backends handle filename {repr(filename)} differently"

        if not mem_error:
            assert mem_exists == fs_exists
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Run the test
if __name__ == "__main__":
    print("Running property-based test to find inconsistencies...")
    print("-" * 60)
    try:
        test_inmemory_vs_filesystem_equivalence()
        print("\nAll tests passed! No inconsistencies found.")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        print("\nThis demonstrates that storage backends behave inconsistently,")