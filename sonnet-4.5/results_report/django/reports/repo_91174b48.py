from io import StringIO
from django.core.serializers.base import ProgressBar

# Create a ProgressBar with total_count=0
output = StringIO()
pb = ProgressBar(output, total_count=0)

# Attempt to update the progress bar
# This will trigger a ZeroDivisionError on line 59 of base.py
pb.update(1)