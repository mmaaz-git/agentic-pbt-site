import os
import json
from hypothesis import given, strategies as st
from flask.config import Config

@given(
    prefix=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
    key=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
    simple_value=st.integers(),
    nested_value=st.text()
)
def test_from_prefixed_env_conflicting_keys(prefix, key, simple_value, nested_value):
    config = Config(root_path='/')

    old_env = os.environ.copy()
    try:
        # Clear any existing prefixed env vars
        for k in list(os.environ.keys()):
            if k.startswith(f'{prefix}_'):
                del os.environ[k]

        os.environ[f'{prefix}_{key}'] = json.dumps(simple_value)
        os.environ[f'{prefix}_{key}__NESTED'] = json.dumps(nested_value)

        config.from_prefixed_env(prefix=prefix)

        assert isinstance(config.get(key), dict) or key not in config
    finally:
        os.environ.clear()
        os.environ.update(old_env)

# Run the test
test_from_prefixed_env_conflicting_keys()