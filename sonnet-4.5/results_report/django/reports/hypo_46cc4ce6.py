from hypothesis import given, settings, strategies as st
from django.core.mail import EmailMessage
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEFAULT_CHARSET='utf-8',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
    )

@given(content=st.binary(min_size=0, max_size=1000))
@settings(max_examples=200)
def test_attach_with_none_filename_and_mimetype(content):
    msg = EmailMessage(
        subject="Test",
        body="Test",
        from_email="test@example.com",
        to=["to@example.com"],
    )
    msg.attach(filename=None, content=content, mimetype=None)
    message = msg.message()
    assert message is not None

# Run the test
if __name__ == "__main__":
    test_attach_with_none_filename_and_mimetype()