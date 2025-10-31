from django.core.mail import EmailMessage
from django.conf import settings

# Configure Django settings
if not settings.configured:
    settings.configure(
        DEFAULT_CHARSET='utf-8',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
    )

# Create an EmailMessage instance
msg = EmailMessage(
    subject="Test",
    body="Test",
    from_email="test@example.com",
    to=["to@example.com"],
)

# Try to attach with None filename and None mimetype
# According to docstring: "The filename can be omitted and the mimetype is guessed, if not provided."
try:
    msg.attach(filename=None, content=b"test content", mimetype=None)
    print("Success: Attachment added without error")
    message = msg.message()
    print("Success: Message created")
except TypeError as e:
    print(f"TypeError occurred: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other error occurred: {e}")
    import traceback
    traceback.print_exc()