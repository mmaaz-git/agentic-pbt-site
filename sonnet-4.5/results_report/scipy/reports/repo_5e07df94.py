import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from io import StringIO
from django.core.serializers.base import ProgressBar

output = StringIO()
progress = ProgressBar(output, total_count=0)
progress.update(0)