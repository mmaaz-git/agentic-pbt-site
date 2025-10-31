import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'test_settings'

import django
from django.conf import settings

settings.configure(
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
    INSTALLED_APPS=['django.contrib.contenttypes'],
)
django.setup()

from django.db import models
from django.db.models import Q


class TestModel(models.Model):
    x = models.IntegerField()
    class Meta:
        app_label = 'test_app'


q = Q(x=0)
q_and = q & q
q_or = q | q

print(f"q: {q}")
print(f"q & q: {q_and}")
print(f"q | q: {q_or}")
print(f"q == (q & q): {q == q_and}")
print(f"q == (q | q): {q == q_or}")

print(f"\nSQL for q: {TestModel.objects.filter(q).query}")
print(f"SQL for q & q: {TestModel.objects.filter(q_and).query}")
print(f"SQL for q | q: {TestModel.objects.filter(q_or).query}")