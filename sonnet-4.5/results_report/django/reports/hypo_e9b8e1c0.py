from django.core.files.base import File
from io import BytesIO
from hypothesis import given, strategies as st, example
import tempfile
import os


@given(st.binary(min_size=0, max_size=1000))
@example(b'')  # Test with empty content
@example(b'test')  # Test with simple content
def test_file_reopen_without_mode(content):
    """Test that File.open() works with file-like objects that don't have a mode attribute."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(content)
        temp_path = tf.name

    try:
        # Create a File object with BytesIO (which doesn't have a 'mode' attribute)
        f = File(BytesIO(content), name=temp_path)
        f.close()

        # This should work but raises AttributeError
        f.open()

        # If we get here, the file was reopened successfully
        assert f.closed is False
        f.close()

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    test_file_reopen_without_mode()