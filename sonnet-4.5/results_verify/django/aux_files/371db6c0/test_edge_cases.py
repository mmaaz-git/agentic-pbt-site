#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.servers.basehttp import ServerHandler
from io import BytesIO

test_cases = [
    ("-1", "negative integer string"),
    ("-100", "large negative integer string"),
    ("0", "zero"),
    ("100", "positive integer"),
    ("abc", "non-numeric string"),
    ("", "empty string"),
    (None, "None value"),
    ("1.5", "float string"),
]

for content_length_value, description in test_cases:
    stdin = BytesIO(b"test data")
    stdout = BytesIO()
    stderr = BytesIO()

    if content_length_value is None:
        environ = {"REQUEST_METHOD": "POST"}
    else:
        environ = {"CONTENT_LENGTH": content_length_value, "REQUEST_METHOD": "POST"}

    try:
        handler = ServerHandler(stdin, stdout, stderr, environ)
        print(f"{description:30} -> limit: {handler.stdin.limit}")
    except Exception as e:
        print(f"{description:30} -> Error: {e}")