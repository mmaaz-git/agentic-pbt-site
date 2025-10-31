from flask import Flask
import flask.json

app = Flask(__name__)

test_dict = {'z': 1, 'a': 2, 'b': 3}

without_context = flask.json.dumps(test_dict)
print(f"Without context: {without_context}")

with app.app_context():
    with_context = flask.json.dumps(test_dict)
    print(f"With context:    {with_context}")