import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

disabled_string = "plugin1, plugin2, plugin3"
plugin_names = disabled_string.split(',')

print(f"Split result: {plugin_names}")

assert plugin_names == ['plugin1', ' plugin2', ' plugin3']

if 'plugin2' in plugin_names:
    print("Plugin2 found (expected)")
else:
    print("BUG: Plugin2 NOT found due to leading space")
    print("Actual values:", [repr(name) for name in plugin_names])