import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.utils import ConnectionHandler

test_cases = [
    ('empty string', ''),
    ('integer', 42),
    ('boolean True', True),
    ('boolean False', False),
    ('list', []),
    ('None', None),
    ('float', 3.14),
]

for name, test_value in test_cases:
    handler = ConnectionHandler()
    databases = {'default': {'TEST': test_value}}
    try:
        result = handler.configure_settings(databases)
        print(f"{name}: SUCCESS - TEST value after configuration: {result['default']['TEST']}")
    except AttributeError as e:
        print(f"{name}: FAILED with AttributeError: {e}")
    except Exception as e:
        print(f"{name}: FAILED with {type(e).__name__}: {e}")