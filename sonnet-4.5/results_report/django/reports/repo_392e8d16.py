import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.utils import ConnectionHandler

handler = ConnectionHandler()
databases = {'default': {'TEST': ''}}
result = handler.configure_settings(databases)