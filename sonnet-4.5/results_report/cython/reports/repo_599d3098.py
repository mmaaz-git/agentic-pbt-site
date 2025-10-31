#!/usr/bin/env python3
"""Minimal reproduction of Cython TypeSlots assertion error bug."""

from Cython.Compiler.TypeSlots import get_slot_by_name

# Try to get a non-existent slot name
# This should raise a proper exception, not AssertionError
get_slot_by_name('nonexistent_slot', {})