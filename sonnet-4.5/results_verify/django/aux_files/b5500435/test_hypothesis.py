from hypothesis import given, strategies as st
from django.dispatch import Signal


def simple_receiver(**kwargs):
    return "received"


@given(st.booleans())
def test_connect_disconnect_inverse(use_caching):
    signal = Signal(use_caching=use_caching)
    signal.connect(simple_receiver)
    assert signal.has_listeners()
    signal.disconnect(simple_receiver)
    assert not signal.has_listeners()


if __name__ == "__main__":
    # Run the test
    test_connect_disconnect_inverse()