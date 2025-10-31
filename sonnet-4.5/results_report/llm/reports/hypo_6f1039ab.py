#!/usr/bin/env python3
"""Property-based test for Django email backend empty messages bug."""

import tempfile
import django
from django.conf import settings
from hypothesis import given, strategies as st

# Configure Django settings
settings.configure(
    EMAIL_HOST='localhost',
    EMAIL_PORT=25,
    EMAIL_HOST_USER='',
    EMAIL_HOST_PASSWORD='',
    EMAIL_USE_TLS=False,
    EMAIL_USE_SSL=False,
    DEFAULT_CHARSET='utf-8',
)
django.setup()

from django.core.mail.backends.filebased import EmailBackend as FileBasedBackend
from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
from django.core.mail.backends.locmem import EmailBackend as LocmemBackend

@given(st.just([]))
def test_empty_message_consistency_filebased(empty_messages):
    """Test that all email backends return consistent values for empty message lists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filebased = FileBasedBackend(file_path=tmpdir)
        smtp = SMTPBackend()
        locmem = LocmemBackend()

        filebased_result = filebased.send_messages(empty_messages)
        smtp_result = smtp.send_messages(empty_messages)
        locmem_result = locmem.send_messages(empty_messages)

        assert isinstance(filebased_result, int), \
            f"Expected int, got {type(filebased_result).__name__}"
        assert filebased_result == smtp_result == locmem_result, \
            f"Inconsistent return values: filebased={filebased_result}, smtp={smtp_result}, locmem={locmem_result}"

if __name__ == "__main__":
    test_empty_message_consistency_filebased()