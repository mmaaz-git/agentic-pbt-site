import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

# Configure Django settings
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        INSTALLED_APPS=['django.contrib.contenttypes']
    )
    django.setup()

import os
import tempfile
from django.core.mail.backends.filebased import EmailBackend

# Test 1: Valid path
print("Test 1: Valid path")
with tempfile.TemporaryDirectory() as tmpdir:
    try:
        backend = EmailBackend(file_path=tmpdir)
        print(f"Success: backend created with file_path={backend.file_path}")
    except Exception as e:
        print(f"Exception: {e.__class__.__name__}: {e}")

# Test 2: Path exists as a file, not a directory
print("\nTest 2: Path exists as a file, not a directory")
with tempfile.NamedTemporaryFile() as tmpfile:
    try:
        backend = EmailBackend(file_path=tmpfile.name)
        print(f"Success: backend created with file_path={backend.file_path}")
    except Exception as e:
        print(f"Exception: {e.__class__.__name__}: {e}")

# Test 3: Non-writable directory
print("\nTest 3: Non-writable directory")
with tempfile.TemporaryDirectory() as tmpdir:
    os.chmod(tmpdir, 0o444)  # Read-only
    try:
        backend = EmailBackend(file_path=tmpdir)
        print(f"Success: backend created with file_path={backend.file_path}")
    except Exception as e:
        print(f"Exception: {e.__class__.__name__}: {e}")
    finally:
        os.chmod(tmpdir, 0o755)  # Restore permissions for cleanup