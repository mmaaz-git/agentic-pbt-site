from hypothesis import given, strategies as st
from flask import Flask
import flask.json

app = Flask(__name__)

@given(st.dictionaries(st.text(), st.integers(), min_size=2))
def test_dumps_consistent_with_without_context(d):
    without_context = flask.json.dumps(d)

    with app.app_context():
        with_context = flask.json.dumps(d)

    assert without_context == with_context, (
        f"flask.json.dumps() behavior should not depend on app context. "
        f"Got different results: {without_context} vs {with_context}"
    )

if __name__ == "__main__":
    test_dumps_consistent_with_without_context()