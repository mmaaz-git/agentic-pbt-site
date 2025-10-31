#!/usr/bin/env python3
"""Test script to reproduce the unreachable code bug in TruncBase.convert_value"""

import sys
import os
import traceback

# Add Django environment to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings

# Configure minimal Django settings
settings.configure(
    USE_TZ=False,
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    INSTALLED_APPS=[],
    SECRET_KEY='test-secret-key'
)

django.setup()

from datetime import datetime
from django.db.models.functions.datetime import TruncBase
from django.db.models.fields import DateField, TimeField
from django.db.models.expressions import Value

print("=== Testing TruncBase.convert_value unreachable code ===\n")

# Test 1: Simple demonstration of the logical impossibility
print("Test 1: Demonstrating logical impossibility")
print("-" * 50)
value = datetime(2023, 1, 15, 12, 30, 45)

print(f"value = {value}")
print(f"type(value) = {type(value)}")
print(f"isinstance(value, datetime) = {isinstance(value, datetime)}")
print(f"value is None = {value is None}")

if isinstance(value, datetime):
    if value is None:
        print("This line is UNREACHABLE - datetime instance cannot be None")
    else:
        print("This line ALWAYS executes when isinstance(value, datetime) is True")
print()

# Test 2: Test TruncBase.convert_value with datetime value
print("Test 2: Testing TruncBase.convert_value with datetime")
print("-" * 50)
try:
    trunc = TruncBase(Value(datetime.now()))
    trunc.output_field = DateField()

    test_datetime = datetime(2023, 1, 15, 12, 30, 45)
    result = trunc.convert_value(test_datetime, None, None)

    print(f"Input datetime: {test_datetime}")
    print(f"Output after convert_value: {result}")
    print(f"Output type: {type(result)}")
    print()
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    print()

# Test 3: Check the actual code path in convert_value
print("Test 3: Analyzing code path in convert_value")
print("-" * 50)
print("Looking at the convert_value method code at lines 357-363:")
print("""
elif isinstance(value, datetime):
    if value is None:  # Line 358 - THIS IS UNREACHABLE
        pass
    elif isinstance(self.output_field, DateField):
        value = value.date()
    elif isinstance(self.output_field, TimeField):
        value = value.time()
""")
print("\nAnalysis:")
print("1. The outer condition checks: isinstance(value, datetime)")
print("2. If this is True, value is a datetime instance")
print("3. The inner condition checks: value is None")
print("4. A datetime instance can NEVER be None")
print("5. Therefore, the 'if value is None: pass' block is unreachable")
print()

# Test 4: Verify with multiple datetime values
print("Test 4: Testing with multiple datetime values")
print("-" * 50)
test_values = [
    datetime(2020, 1, 1),
    datetime(2023, 6, 15, 14, 30),
    datetime(2025, 12, 31, 23, 59, 59),
]

for dt_value in test_values:
    print(f"Testing {dt_value}:")
    print(f"  isinstance(dt_value, datetime) = {isinstance(dt_value, datetime)}")
    print(f"  dt_value is None = {dt_value is None}")

    # Simulate the problematic code
    if isinstance(dt_value, datetime):
        if dt_value is None:
            print("  -> Entered 'if value is None' block (IMPOSSIBLE)")
        else:
            print("  -> Skipped 'if value is None' block (ALWAYS happens)")
print()

# Test 5: Property-based test using hypothesis
print("Test 5: Property-based test with hypothesis")
print("-" * 50)
try:
    from hypothesis import given, strategies as st

    @st.composite
    def datetime_values(draw):
        year = draw(st.integers(min_value=1900, max_value=2100))
        month = draw(st.integers(min_value=1, max_value=12))
        day = draw(st.integers(min_value=1, max_value=28))
        hour = draw(st.integers(min_value=0, max_value=23))
        minute = draw(st.integers(min_value=0, max_value=59))
        second = draw(st.integers(min_value=0, max_value=59))
        return datetime(year, month, day, hour, minute, second)

    @given(dt=datetime_values())
    def test_convert_value_never_reaches_none_check(dt):
        trunc = TruncBase(Value(dt))
        trunc.output_field = DateField()

        result = trunc.convert_value(dt, None, None)

        assert result == dt.date()

        # Additional check: verify the logical impossibility
        assert isinstance(dt, datetime)
        assert dt is not None

        # This mimics the problematic code
        reached_none_block = False
        if isinstance(dt, datetime):
            if dt is None:
                reached_none_block = True

        assert not reached_none_block, "Should never reach the 'if value is None' block"

    # Run the property-based test
    test_convert_value_never_reaches_none_check()
    print("Property-based test passed - confirmed that 'if value is None' is never reached")

except ImportError:
    print("Hypothesis not installed, skipping property-based test")
except Exception as e:
    print(f"Property-based test error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("CONCLUSION: The code at line 358 'if value is None:' is unreachable")
print("because it's inside an 'elif isinstance(value, datetime):' block.")
print("A datetime instance can never be None, making this dead code.")
print("=" * 60)