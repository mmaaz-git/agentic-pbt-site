from hypothesis import given, strategies as st, settings
from flask import Config
import tempfile
import os

st_key = st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ_', min_size=1, max_size=20)

@settings(max_examples=100)
@given(
    key1=st_key.filter(lambda x: '__' not in x),
    value1=st.integers(),
    value2=st.integers()
)
def test_from_prefixed_env_collision_flat_then_nested(key1, value1, value2):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(tmpdir)

        env_key_flat = f"FLASK_{key1}"
        env_key_nested = f"FLASK_{key1}__SUBKEY"

        old_flat = os.environ.get(env_key_flat)
        old_nested = os.environ.get(env_key_nested)
        try:
            os.environ[env_key_flat] = str(value1)
            os.environ[env_key_nested] = str(value2)
            config.from_prefixed_env()
            print(f"Success with key1={key1!r}, value1={value1}, value2={value2}")
        except TypeError as e:
            print(f"TypeError with key1={key1!r}, value1={value1}, value2={value2}: {e}")
            raise
        finally:
            if old_flat is None:
                os.environ.pop(env_key_flat, None)
            else:
                os.environ[env_key_flat] = old_flat
            if old_nested is None:
                os.environ.pop(env_key_nested, None)
            else:
                os.environ[env_key_nested] = old_nested

# Test the specific failing input
if __name__ == "__main__":
    print("Testing specific failing input: key1='A', value1=0, value2=0")
    test_from_prefixed_env_collision_flat_then_nested('A', 0, 0)