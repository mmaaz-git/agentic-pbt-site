import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(DEBUG=False)

from hypothesis import given, strategies as st
from django.dispatch import Signal


def receiver(**kwargs):
    return "received"


@given(st.text(min_size=1))
def test_signal_with_caching_and_string_sender(sender):
    signal = Signal(use_caching=True)

    signal.connect(receiver, sender=sender, weak=False)
    responses = signal.send(sender=sender)

    assert len(responses) > 0


if __name__ == "__main__":
    test_signal_with_caching_and_string_sender()