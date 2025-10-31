#!/usr/bin/env python3
"""Minimal reproduction case for ModelFormMixin.get_success_url() KeyError bug"""

from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin

# Create a ModelFormMixin instance
mixin = ModelFormMixin()

# Set a success_url with a placeholder
mixin.success_url = "/object/{id}/success"

# Create a mock object with an empty __dict__ (no 'id' attribute)
mock_obj = Mock()
mock_obj.__dict__ = {}
mixin.object = mock_obj

# This should raise a KeyError
try:
    result = mixin.get_success_url()
    print(f"Success URL: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error args: {e.args}")
    import traceback
    traceback.print_exc()