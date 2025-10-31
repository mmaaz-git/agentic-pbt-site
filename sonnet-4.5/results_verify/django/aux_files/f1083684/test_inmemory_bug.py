import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
import tempfile

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        USE_TZ=True,
        MEDIA_ROOT=tempfile.mkdtemp(),
        MEDIA_URL='/media/',
    )
    django.setup()

from django.core.files.storage import InMemoryStorage
from django.core.files.base import ContentFile

storage = InMemoryStorage()

content = ContentFile(b"Test content", name="test.txt")
saved_name = storage.save("test.txt", content)

file_node = storage._resolve(saved_name)
print(f"File closed: {file_node.file.closed}")

# Let's also check what type of object file_node.file is
print(f"File type: {type(file_node.file)}")
print(f"File value: {file_node.file.getvalue()}")