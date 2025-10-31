from django.dispatch import Signal
from hypothesis import given, strategies as st

@given(st.booleans())
def test_send_with_none_sender(use_caching):
    signal = Signal(use_caching=use_caching)

    def receiver(sender, **kwargs):
        return "response"

    signal.connect(receiver, weak=False)
    responses = signal.send(sender=None)
    assert len(responses) == 1

# Run the test
if __name__ == "__main__":
    test_send_with_none_sender()