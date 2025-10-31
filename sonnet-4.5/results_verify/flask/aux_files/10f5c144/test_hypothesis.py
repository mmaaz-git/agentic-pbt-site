from flask import Flask
from hypothesis import given, strategies as st

@given(
    st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.integers(),
        min_size=1
    )
)
def test_config_update_only_uppercase(mapping):
    app = Flask(__name__)
    config = app.config

    initial_keys = set(config.keys())
    config.update(mapping)

    for key in mapping:
        if key.isupper():
            assert key in config
            assert config[key] == mapping[key]
        else:
            if key not in initial_keys:
                assert key not in config

# Manual test function
def manual_test():
    app = Flask(__name__)
    config = app.config

    initial_keys = set(config.keys())

    # Test with non-uppercase key
    mapping = {'0': 0}
    config.update(mapping)

    for key in mapping:
        if key.isupper():
            assert key in config
            assert config[key] == mapping[key]
        else:
            if key not in initial_keys:
                assert key not in config, f"Non-uppercase key '{key}' should not be in config after update()"

if __name__ == "__main__":
    try:
        manual_test()
        print("Test passed - update() does not accept non-uppercase keys")
    except AssertionError as e:
        print(f"Test failed - {e}")
        print("This shows that update() DOES accept non-uppercase keys")