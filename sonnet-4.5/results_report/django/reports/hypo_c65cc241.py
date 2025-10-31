#!/usr/bin/env python3
"""Property-based test for Django filebased email backend None path handling"""

from hypothesis import given, strategies as st, settings as hypo_settings
from django.core.mail.backends.filebased import EmailBackend
from django.conf import settings
import traceback

# Configure Django with EMAIL_FILE_PATH set to None
if not settings.configured:
    settings.configure(EMAIL_FILE_PATH=None, DEFAULT_CHARSET='utf-8')

@given(st.just(None))
@hypo_settings(max_examples=1, deadline=None)
def test_filebased_none_path_handling(file_path):
    """Test that filebased backend handles None file_path gracefully"""
    print(f"Testing with file_path={file_path}")
    try:
        backend = EmailBackend(file_path=file_path)
        print("ERROR: Backend created successfully - expected TypeError!")
        assert False, "Should raise exception for None file_path"
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
        traceback.print_exc()
        # Expected behavior with current bug

if __name__ == "__main__":
    # Run the property test
    print("Running property-based test for filebased email backend with None path...")
    test_filebased_none_path_handling()
    print("\nTest completed - TypeError was caught as expected (demonstrating the bug)")