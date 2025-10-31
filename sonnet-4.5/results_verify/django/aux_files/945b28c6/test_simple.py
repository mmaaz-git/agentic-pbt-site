from django.db.utils import ConnectionHandler

handler = ConnectionHandler()
databases = {'default': {'TEST': ''}}
try:
    configured = handler.configure_settings(databases)
    print("Success - no error occurred")
except AttributeError as e:
    print(f"AttributeError caught: {e}")
    print(f"Error type: {type(e).__name__}")