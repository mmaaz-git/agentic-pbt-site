#!/usr/bin/env python3
"""Hypothesis test for DictionarySerializer bug"""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

import django
django.setup()

from hypothesis import given, strategies as st, example
from django.db.migrations.serializer import serializer_factory

@given(st.dictionaries(
    st.one_of(st.integers(), st.text()),
    st.integers(),
    min_size=2,
    max_size=5
))
@example({1: 10, 'a': 20})
def test_dict_serializer_mixed_keys(value):
    """Test that DictionarySerializer can handle mixed-type keys"""
    try:
        serialized, imports = serializer_factory(value).serialize()
        # Deserialize to check round-trip
        exec_globals = {}
        for imp in imports:
            exec(imp, exec_globals)
        deserialized = eval(serialized, exec_globals)
        assert deserialized == value
        print(f"✓ Success with dict: {value}")
    except TypeError as e:
        if "not supported between instances of" in str(e):
            print(f"✗ TypeError with dict {value}: {e}")
            raise
        else:
            raise
    except Exception as e:
        print(f"✗ Unexpected error with dict {value}: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    print("Running Hypothesis test...")
    print("=" * 60)

    # Run the test - it will raise if there are any failures
    try:
        test_dict_serializer_mixed_keys()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")