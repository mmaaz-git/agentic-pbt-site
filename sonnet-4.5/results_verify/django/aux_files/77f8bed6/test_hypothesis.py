import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-secret-key',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
        EMAIL_SUBJECT_PREFIX='[Django] ',
        SERVER_EMAIL='root@localhost',
        ADMINS=[('Admin', 'admin@example.com')],
        MANAGERS=[('Manager', 'manager@example.com')],
    )

django.setup()

from hypothesis import given, strategies as st
from django.core.mail import send_mail, mail_admins

@given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=200))
def test_email_functions_return_send_count(subject, message):
    """All email sending functions should return the count of messages sent"""

    # send_mail returns count
    result1 = send_mail(subject, message, 'from@example.com', ['to@example.com'])
    assert isinstance(result1, int) and result1 >= 0

    # mail_admins should also return count, but returns None
    settings.ADMINS = [('Admin', 'admin@example.com')]
    result2 = mail_admins(subject, message)
    assert isinstance(result2, int) and result2 >= 0, f"Expected int, got {type(result2)}"

# Run the test
test_email_functions_return_send_count()