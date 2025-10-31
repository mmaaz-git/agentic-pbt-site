import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from django.core.files.base import File
import tempfile
import os

@given(st.sampled_from(['r+', 'w+', 'a', 'a+', 'x', 'x+']))
def test_writable_detects_all_write_modes(mode):
    """Test that FileProxyMixin.writable() correctly detects all writable file modes."""

    # Create a mock file object without a writable() method to force fallback to mode checking
    class MockFileWithMode:
        def __init__(self, mode):
            self.mode = mode
            self.closed = False
            # Provide minimal file-like interface but no writable() method
            self.read = lambda: b''
            self.write = lambda x: None
            self.seek = lambda x: None
            self.tell = lambda: 0

    mock_file = MockFileWithMode(mode)
    django_file = File(mock_file)

    # All these modes should be writable according to Python's file mode specification
    assert django_file.writable() == True, \
        f"File with mode '{mode}' should be writable"

if __name__ == "__main__":
    test_writable_detects_all_write_modes()