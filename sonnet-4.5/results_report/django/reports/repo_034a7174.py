import tempfile
import django
from django.conf import settings

# Configure Django settings
settings.configure(
    USE_TZ=False,
    MEDIA_ROOT='/tmp',
    MEDIA_URL='/media/',
)
django.setup()

from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, InMemoryStorage

# Test case 1: Single CR character
content = '\r'

with tempfile.TemporaryDirectory() as tmpdir:
    fs_storage = FileSystemStorage(location=tmpdir)
    mem_storage = InMemoryStorage()

    fs_file = ContentFile(content, name='test.txt')
    mem_file = ContentFile(content, name='test.txt')

    fs_saved = fs_storage.save('test.txt', fs_file)
    mem_saved = mem_storage.save('test.txt', mem_file)

    with fs_storage.open(fs_saved, 'r') as f:
        fs_result = f.read()

    with mem_storage.open(mem_saved, 'r') as f:
        mem_result = f.read()

    print(f"Test 1: Single CR character")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")
    print()

# Test case 2: CRLF sequence
content = '\r\n'

with tempfile.TemporaryDirectory() as tmpdir:
    fs_storage = FileSystemStorage(location=tmpdir)
    mem_storage = InMemoryStorage()

    fs_file = ContentFile(content, name='test.txt')
    mem_file = ContentFile(content, name='test.txt')

    fs_saved = fs_storage.save('test.txt', fs_file)
    mem_saved = mem_storage.save('test.txt', mem_file)

    with fs_storage.open(fs_saved, 'r') as f:
        fs_result = f.read()

    with mem_storage.open(mem_saved, 'r') as f:
        mem_result = f.read()

    print(f"Test 2: CRLF sequence")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")
    print()

# Test case 3: Text with CR in the middle
content = 'hello\rworld'

with tempfile.TemporaryDirectory() as tmpdir:
    fs_storage = FileSystemStorage(location=tmpdir)
    mem_storage = InMemoryStorage()

    fs_file = ContentFile(content, name='test.txt')
    mem_file = ContentFile(content, name='test.txt')

    fs_saved = fs_storage.save('test.txt', fs_file)
    mem_saved = mem_storage.save('test.txt', mem_file)

    with fs_storage.open(fs_saved, 'r') as f:
        fs_result = f.read()

    with mem_storage.open(mem_saved, 'r') as f:
        mem_result = f.read()

    print(f"Test 3: Text with CR in the middle")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")
    print()

# Test case 4: Windows-style text file
content = 'line1\r\nline2\r\nline3'

with tempfile.TemporaryDirectory() as tmpdir:
    fs_storage = FileSystemStorage(location=tmpdir)
    mem_storage = InMemoryStorage()

    fs_file = ContentFile(content, name='test.txt')
    mem_file = ContentFile(content, name='test.txt')

    fs_saved = fs_storage.save('test.txt', fs_file)
    mem_saved = mem_storage.save('test.txt', mem_file)

    with fs_storage.open(fs_saved, 'r') as f:
        fs_result = f.read()

    with mem_storage.open(mem_saved, 'r') as f:
        mem_result = f.read()

    print(f"Test 4: Windows-style text file")
    print(f"Original:          {repr(content)}")
    print(f"FileSystemStorage: {repr(fs_result)}")
    print(f"InMemoryStorage:   {repr(mem_result)}")
    print(f"Match: {fs_result == mem_result}")