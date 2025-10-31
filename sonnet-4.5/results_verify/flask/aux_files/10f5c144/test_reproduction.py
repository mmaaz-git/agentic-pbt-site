from flask import Flask

app = Flask(__name__)
config = app.config

config.update({'lowercase_key': 'value1', 'UPPERCASE_KEY': 'value2'})

print("'lowercase_key' in config:", 'lowercase_key' in config)
print("'UPPERCASE_KEY' in config:", 'UPPERCASE_KEY' in config)

app2 = Flask(__name__)
config2 = app2.config

config2.from_mapping({'lowercase_key': 'value1', 'UPPERCASE_KEY': 'value2'})

print("'lowercase_key' in config2:", 'lowercase_key' in config2)
print("'UPPERCASE_KEY' in config2:", 'UPPERCASE_KEY' in config2)