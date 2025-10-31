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

from hypothesis import given, strategies as st, settings as hyp_settings
from django.core.files.storage import InMemoryStorage, FileSystemStorage
from django.core.files.base import ContentFile
import tempfile
import shutil

@given(
    st.text(min_size=1, max_size=50).filter(lambda x: '/' not in x and '\\' not in x and '..' not in x and x not in ['.', '..']),
    st.binary(min_size=0, max_size=1000)
)
@hyp_settings(max_examples=100)
def test_inmemory_vs_filesystem_equivalence(filename, content_bytes):
    temp_dir = tempfile.mkdtemp()
    try:
        mem_storage = InMemoryStorage()
        fs_storage = FileSystemStorage(location=temp_dir)

        mem_content = ContentFile(content_bytes)
        fs_content = ContentFile(content_bytes)

        mem_error = None
        fs_error = None
        mem_name = None
        fs_name = None

        try:
            mem_name = mem_storage.save(filename, mem_content)
        except Exception as e:
            mem_error = type(e).__name__

        try:
            fs_name = fs_storage.save(filename, fs_content)
        except Exception as e:
            fs_error = type(e).__name__

        # If one raises an error, both should raise an error
        if mem_error or fs_error:
            if mem_error != fs_error:
                print(f"MISMATCH: filename={repr(filename)}")
                print(f"  InMemoryStorage: {mem_error or 'Success'}")
                print(f"  FileSystemStorage: {fs_error or 'Success'}")
                assert False, f"Different behavior for {repr(filename)}"
        else:
            # Both succeeded, check existence
            assert mem_storage.exists(mem_name) == fs_storage.exists(fs_name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Test with the specific failing input
    print("Testing with null byte filename...")
    test_inmemory_vs_filesystem_equivalence('\x00', b'')
    print("Test passed!")

    # Run property-based testing
    print("\nRunning property-based tests...")
    test_inmemory_vs_filesystem_equivalence()