#!/usr/bin/env python3
"""
Minimal reproduction of the Django ConnectionHandler bug with non-dictionary TEST values.
"""

from django.db.utils import ConnectionHandler

# Create a ConnectionHandler instance
handler = ConnectionHandler()

# Set up a database configuration with TEST as an empty string (not a dictionary)
databases = {'default': {'TEST': ''}}

# This should configure settings but will crash with AttributeError
try:
    configured = handler.configure_settings(databases)
    print("Configuration succeeded (unexpected)")
    print("Configured databases:", configured)
except AttributeError as e:
    print(f"AttributeError caught: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()