#!/usr/bin/env python3
"""Property-based test for FileBased email backend."""

from hypothesis import given, strategies as st, settings
import tempfile
from django.core.mail.backends.filebased import EmailBackend
from django.core.mail.message import EmailMessage


def create_mock_message():
    """Create a simple email message for testing."""
    return EmailMessage(
        subject="Test subject",
        body="Test body",
        from_email="test@example.com",
        to=["recipient@example.com"]
    )


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_filebased_backend_returns_count(num_messages):
    """Test that FileBased backend always returns an integer count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = EmailBackend(file_path=tmpdir, fail_silently=True)
        messages = [create_mock_message() for _ in range(num_messages)]

        result = backend.send_messages(messages)

        assert isinstance(result, int), f"Expected int, got {type(result)} for {num_messages} messages"
        assert result >= 0, f"Expected non-negative count, got {result}"


if __name__ == "__main__":
    # Run the property-based test
    try:
        test_filebased_backend_returns_count()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis test demonstrates that send_messages() should always return an integer,")
        print("but returns None when given an empty list of messages.")