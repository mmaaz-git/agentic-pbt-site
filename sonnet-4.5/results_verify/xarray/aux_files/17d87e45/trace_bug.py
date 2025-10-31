import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

def trace_collapsible_section(name, inline_details="", details="", n_items=None, enabled=True, collapsed=False):
    """Tracing through the bug logic"""
    import uuid

    print(f"Initial parameters: enabled={enabled} (type: {type(enabled).__name__}), n_items={n_items}")

    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    print(f"has_items = {has_items}")

    n_items_span = "" if n_items is None else f" <span>({n_items})</span>"

    # This is line 181 - the problematic reassignment
    enabled = "" if enabled and has_items else "disabled"
    print(f"After line 181: enabled={repr(enabled)} (type: {type(enabled).__name__})")

    collapsed = "" if collapsed or not has_items else "checked"

    # This is line 183 - checking the string value
    tip = " title='Expand/collapse section'" if enabled else ""
    print(f"Line 183: enabled={repr(enabled)}, bool(enabled)={bool(enabled)}, tip={repr(tip)}")

    return enabled, tip

# Test case 1: Should have tooltip (enabled checkbox)
print("=== Test 1: Enabled checkbox (should have tooltip) ===")
enabled_attr, tip = trace_collapsible_section("Test", n_items=5, enabled=True)
print(f"Result: enabled_attr={repr(enabled_attr)}, tip={repr(tip)}")
print()

# Test case 2: Should NOT have tooltip (disabled checkbox)
print("=== Test 2: Disabled checkbox (should NOT have tooltip) ===")
enabled_attr, tip = trace_collapsible_section("Test", n_items=0, enabled=True)
print(f"Result: enabled_attr={repr(enabled_attr)}, tip={repr(tip)}")
print()

# Test case 3: explicit enabled=False
print("=== Test 3: Explicitly disabled (enabled=False) ===")
enabled_attr, tip = trace_collapsible_section("Test", n_items=5, enabled=False)
print(f"Result: enabled_attr={repr(enabled_attr)}, tip={repr(tip)}")