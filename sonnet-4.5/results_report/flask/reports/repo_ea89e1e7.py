import os
import json
from flask.config import Config

# Clear environment to ensure clean test
for key in list(os.environ.keys()):
    if key.startswith('FLASK_'):
        del os.environ[key]

config = Config(root_path='/')

os.environ['FLASK_DB'] = json.dumps("sqlite")
os.environ['FLASK_DB__NAME'] = json.dumps("mydb")

config.from_prefixed_env(prefix="FLASK")
print("Config:", dict(config))