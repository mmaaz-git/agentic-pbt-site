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

mem_storage = InMemoryStorage()
temp_dir = tempfile.mkdtemp()
fs_storage = FileSystemStorage(location=temp_dir)

filename = 'file\x00name.txt'

print("InMemoryStorage:")
try:
    saved = mem_storage.save(filename, ContentFile(b'test'))
    print(f"  ✓ Accepted - saved as: {repr(saved)}")
except Exception as e:
    print(f"  ✗ Rejected: {e}")

print("FileSystemStorage:")
try:
    saved = fs_storage.save(filename, ContentFile(b'test'))
    print(f"  ✓ Accepted - saved as: {repr(saved)}")
except Exception as e:
    print(f"  ✗ Rejected: {e}")

shutil.rmtree(temp_dir)