from unittest.mock import Mock
from django.dispatch import Signal

def test_connect_disconnect_inverse(use_caching):
    signal = Signal(use_caching=use_caching)
    recv = Mock(return_value=None, __name__="test_receiver")
    sender = object()

    signal.connect(recv, sender=sender, weak=False)
    assert signal.has_listeners(sender=sender)

    disconnected = signal.disconnect(recv, sender=sender)
    assert disconnected
    assert not signal.has_listeners(sender=sender)

# Run the test
test_connect_disconnect_inverse(use_caching=False)
print("Test passed with use_caching=False")

try:
    test_connect_disconnect_inverse(use_caching=True)
    print("Test passed with use_caching=True")
except Exception as e:
    print(f"Test failed with use_caching=True: {e}")
