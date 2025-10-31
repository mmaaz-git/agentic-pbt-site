import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from io import BytesIO
from django.core.files.base import File

class MockFileWithMode:
    def __init__(self, mode):
        self.mode = mode
        self._io = BytesIO(b'test')
        # Store necessary attributes from BytesIO but not writable()
        self.read = self._io.read
        self.write = self._io.write
        self.seek = self._io.seek
        self.tell = self._io.tell
        self.closed = False

modes_to_test = ['r', 'r+', 'w', 'w+', 'a', 'a+', 'x', 'x+', 'rb', 'r+b', 'wb', 'w+b', 'ab', 'a+b', 'xb', 'x+b']

print("Testing Django's FileProxyMixin.writable() method:")
print("=" * 60)

for mode in modes_to_test:
    mock_file = MockFileWithMode(mode)
    django_file = File(mock_file)

    is_writable = django_file.writable()

    # Determine expected result based on Python's file mode specification
    # A file is writable if it has any of: 'w', 'a', 'x', or '+' in its mode
    should_be_writable = any(c in mode for c in 'wax+')

    if is_writable == should_be_writable:
        status = "✓ PASS"
    else:
        status = "✗ FAIL"

    print(f"{status} Mode '{mode:4s}': writable()={str(is_writable):5s}, expected={str(should_be_writable):5s}")

print("\n" + "=" * 60)
print("Summary of failures:")
print("-" * 60)

failure_count = 0
for mode in modes_to_test:
    mock_file = MockFileWithMode(mode)
    django_file = File(mock_file)
    is_writable = django_file.writable()
    should_be_writable = any(c in mode for c in 'wax+')

    if is_writable != should_be_writable:
        failure_count += 1
        print(f"Mode '{mode}': Expected writable()={should_be_writable}, but got {is_writable}")

if failure_count == 0:
    print("No failures detected!")
else:
    print(f"\nTotal failures: {failure_count}")