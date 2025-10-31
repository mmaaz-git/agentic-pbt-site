import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=False)

from django.dispatch import Signal


def receiver(**kwargs):
    return "received"


# Test with use_caching=True and string sender
signal = Signal(use_caching=True)
sender = "my_sender"

signal.connect(receiver, sender=sender, weak=False)

# This should raise TypeError
try:
    responses = signal.send(sender=sender)
    print(f"Success: responses = {responses}")
except TypeError as e:
    print(f"TypeError raised: {e}")