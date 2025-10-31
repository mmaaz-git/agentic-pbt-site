import io
from django.core.serializers.base import ProgressBar

# Create a ProgressBar with total_count=0 (simulating an empty database)
output = io.StringIO()
pb = ProgressBar(output, total_count=0)

# Attempt to update the progress bar
# This will cause a ZeroDivisionError at line 59 of base.py
pb.update(0)