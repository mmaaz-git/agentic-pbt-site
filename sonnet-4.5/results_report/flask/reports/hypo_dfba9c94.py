#!/usr/bin/env python3
"""Property-based test for Flask Config.from_prefixed_env type collision bug"""
from hypothesis import given, strategies as st, settings
from flask import Config
import tempfile
import os

# Strategy for valid environment variable keys (uppercase letters and underscores)
st_key = st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ_', min_size=1, max_size=20)

@settings(max_examples=100)
@given(
    key1=st_key.filter(lambda x: '__' not in x),
    value1=st.integers(),
    value2=st.integers()
)
def test_from_prefixed_env_collision_flat_then_nested(key1, value1, value2):
    """Test that from_prefixed_env handles flat+nested key collisions gracefully"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(tmpdir)

        env_key_flat = f"FLASK_{key1}"
        env_key_nested = f"FLASK_{key1}__SUBKEY"

        # Save old values if they exist
        old_flat = os.environ.get(env_key_flat)
        old_nested = os.environ.get(env_key_nested)

        try:
            # Set both flat and nested keys
            os.environ[env_key_flat] = str(value1)
            os.environ[env_key_nested] = str(value2)

            # This should either work or fail with a clear error
            config.from_prefixed_env()

            # If it works, verify the config is sensible
            assert key1 in config, f"Key {key1} should be in config"

        except TypeError as e:
            # This is the bug - TypeError is not a clear error
            print(f"\n❌ BUG FOUND with inputs: key1={key1!r}, value1={value1}, value2={value2}")
            print(f"   Environment variables: {env_key_flat}={value1}, {env_key_nested}={value2}")
            print(f"   Error: {e}")
            raise  # Re-raise to let Hypothesis catch it

        finally:
            # Restore original environment
            if old_flat is None:
                os.environ.pop(env_key_flat, None)
            else:
                os.environ[env_key_flat] = old_flat
            if old_nested is None:
                os.environ.pop(env_key_nested, None)
            else:
                os.environ[env_key_nested] = old_nested

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for Flask Config.from_prefixed_env...")
    print("=" * 60)
    test_from_prefixed_env_collision_flat_then_nested()