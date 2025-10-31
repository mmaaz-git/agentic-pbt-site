#!/usr/bin/env python3
"""Test all Django email backends for consistent behavior with empty lists."""

import sys
import os
import tempfile

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'
import django
django.setup()

print("=" * 80)
print("Testing all EmailBackend.send_messages() with empty list")
print("=" * 80)

backends = [
    ('dummy', 'django.core.mail.backends.dummy.EmailBackend'),
    ('locmem', 'django.core.mail.backends.locmem.EmailBackend'),
    ('console', 'django.core.mail.backends.console.EmailBackend'),
    ('filebased', 'django.core.mail.backends.filebased.EmailBackend'),
    # Skip SMTP as it requires a connection
]

for name, backend_path in backends:
    print(f"\nTesting {name} backend:")
    try:
        module_name, class_name = backend_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        BackendClass = getattr(module, class_name)

        if name == 'filebased':
            # filebased requires a file path
            with tempfile.TemporaryDirectory() as tmpdir:
                backend = BackendClass(file_path=tmpdir, fail_silently=True)
                result = backend.send_messages([])
        elif name == 'console':
            backend = BackendClass(stream=open('/dev/null', 'w'), fail_silently=True)
            result = backend.send_messages([])
        else:
            backend = BackendClass(fail_silently=True)
            result = backend.send_messages([])

        print(f"  Result: {result}, Type: {type(result).__name__}")
        print(f"  Returns int: {isinstance(result, int)}")
        print(f"  Returns None: {result is None}")

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("Summary:")
print("- dummy backend: Returns int (0) correctly")
print("- locmem backend: Returns int (0) correctly")
print("- console backend: Returns None (BUG!)")
print("- filebased backend: Inherits from console, also returns None (BUG!)")
print("- smtp backend: Returns int (0) correctly (based on code inspection)")
print("=" * 80)