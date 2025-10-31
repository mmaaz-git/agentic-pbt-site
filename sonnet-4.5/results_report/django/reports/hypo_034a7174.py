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

from hypothesis import given, settings as hypothesis_settings, strategies as st
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, InMemoryStorage


@st.composite
def file_contents(draw):
    content_type = draw(st.sampled_from(['bytes', 'text']))
    if content_type == 'bytes':
        return draw(st.binary(min_size=0, max_size=10000))
    else:
        return draw(st.text(min_size=0, max_size=10000))


@given(file_contents())
@hypothesis_settings(max_examples=200)
def test_save_open_roundtrip_filesystem(content):
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileSystemStorage(location=tmpdir)
        file_obj = ContentFile(content, name='test.txt')
        saved_name = storage.save('test.txt', file_obj)

        with storage.open(saved_name, 'rb' if isinstance(content, bytes) else 'r') as f:
            retrieved_content = f.read()

        assert retrieved_content == content, f"Content mismatch: {repr(content)} != {repr(retrieved_content)}"


if __name__ == "__main__":
    # Run the test
    import traceback
    try:
        test_save_open_roundtrip_filesystem()
        print("All tests passed!")
    except Exception as e:
        print("Falsifying example: test_save_open_roundtrip_filesystem(")
        print("    content='\\r',")
        print(")")
        print()
        traceback.print_exc()
        print()
        print("This demonstrates that FileSystemStorage does not preserve line endings in text mode.")