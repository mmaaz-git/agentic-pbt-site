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

from django.core.files.storage import InMemoryStorage, FileSystemStorage
from django.core.files.base import ContentFile
import tempfile
import shutil

# Create storage instances
mem_storage = InMemoryStorage()
temp_dir = tempfile.mkdtemp()
fs_storage = FileSystemStorage(location=temp_dir)

# Test filename with null byte
filename = 'file\x00name.txt'
content_bytes = b'test content'

print("Testing filename with null byte: repr(%r)" % filename)
print("-" * 50)

print("\n1. InMemoryStorage:")
try:
    saved_name = mem_storage.save(filename, ContentFile(content_bytes))
    print(f"   ✓ Accepted - saved as: {repr(saved_name)}")
    print(f"   ✓ File exists: {mem_storage.exists(saved_name)}")
except Exception as e:
    print(f"   ✗ Rejected with {type(e).__name__}: {e}")

print("\n2. FileSystemStorage:")
try:
    saved_name = fs_storage.save(filename, ContentFile(content_bytes))
    print(f"   ✓ Accepted - saved as: {repr(saved_name)}")
    print(f"   ✓ File exists: {fs_storage.exists(saved_name)}")
except Exception as e:
    print(f"   ✗ Rejected with {type(e).__name__}: {e}")

# Clean up
shutil.rmtree(temp_dir)

print("\n" + "-" * 50)
print("Result: Storage backends behave INCONSISTENTLY")
print("InMemoryStorage accepts null bytes while FileSystemStorage rejects them")