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

from hypothesis import given, strategies as st, settings as hyp_settings
from django.core.files.storage import InMemoryStorage
from django.core.files.base import ContentFile

@given(
    content=st.binary(min_size=1, max_size=1000),
    filename=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20
    ).filter(lambda x: '/' not in x and x not in {'.', '..'})
)
@hyp_settings(max_examples=100)
def test_file_closed_after_save(content, filename):
    storage = InMemoryStorage()

    file_obj = ContentFile(content, name=filename)
    saved_name = storage.save(filename, file_obj)

    file_node = storage._resolve(saved_name)

    assert hasattr(file_node, 'file'), "File node should have a file attribute"

    # The file should ideally be closed after save completes
    # Currently it remains open, which is a resource leak
    is_closed = file_node.file.closed

    # This assertion would fail with current implementation
    print(f"Testing file {filename}: closed={is_closed}")
    # assert is_closed, f"File should be closed after save, but is open"

    # For this test, we'll just verify that the file is NOT closed,
    # confirming the bug exists
    assert not is_closed, "File is unexpectedly closed (bug may have been fixed?)"

# Run the test
test_file_closed_after_save()
print("Test completed successfully - all files remained open (bug confirmed)")