import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.utils import ConnectionHandler

handler = ConnectionHandler()
databases = {'default': {'TEST': ''}}
try:
    result = handler.configure_settings(databases)
    print("No error occurred")
except AttributeError as e:
    print(f"AttributeError: {e}")
    import traceback
    traceback.print_exc()