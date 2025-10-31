from hypothesis import given, strategies as st
from django.dispatch import Signal


@given(st.booleans())
def test_connect_disconnect_roundtrip(use_caching):
    signal = Signal(use_caching=use_caching)

    def receiver(sender, **kwargs):
        return "received"

    signal.connect(receiver)
    assert signal.has_listeners()

    result = signal.disconnect(receiver)
    assert result == True
    assert not signal.has_listeners()

if __name__ == "__main__":
    # Run the test
    test_connect_disconnect_roundtrip()