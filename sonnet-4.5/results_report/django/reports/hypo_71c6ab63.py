#!/usr/bin/env python3
"""Property-based test for Django UploadedFile backslash vulnerability"""

import os
import sys

# Add Django to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup minimal Django settings
import django
from django.conf import settings
settings.configure(DEBUG=True, SECRET_KEY='test')
django.setup()

from hypothesis import given, strategies as st, assume
from django.core.files.uploadedfile import SimpleUploadedFile

@given(st.text(min_size=1), st.binary())
def test_uploaded_file_name_sanitization(name, content):
    assume(name not in {'', '.', '..'})
    f = SimpleUploadedFile(name, content)
    assert '/' not in f.name, f"Forward slash found in sanitized name: {f.name!r}"
    assert '\\' not in f.name, f"Backslash found in sanitized name: {f.name!r}"

if __name__ == "__main__":
    # Run the test
    test_uploaded_file_name_sanitization()