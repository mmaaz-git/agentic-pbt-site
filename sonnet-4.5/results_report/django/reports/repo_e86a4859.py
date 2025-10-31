import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
import tempfile
from pathlib import Path
from django.conf import Settings

zoneinfo_root = Path("/usr/share/zoneinfo")

with tempfile.TemporaryDirectory() as tmpdir:
    external_file = Path(tmpdir) / "fake_timezone"
    external_file.touch()

    relative_path = os.path.relpath(external_file, zoneinfo_root)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(f"TIME_ZONE = {repr(relative_path)}\n")
        f.write("SECRET_KEY = 'test'\n")
        settings_file = f.name

    try:
        module_name = Path(settings_file).stem
        sys.path.insert(0, str(Path(settings_file).parent))

        try:
            settings_obj = Settings(module_name)
            print(f"BUG: Django accepted TIME_ZONE = {repr(relative_path)}")
            print(f"This path points to: {zoneinfo_root.joinpath(*relative_path.split('/')).resolve()}")
            print(f"Which is outside: {zoneinfo_root.resolve()}")
        finally:
            sys.path.remove(str(Path(settings_file).parent))
            if module_name in sys.modules:
                del sys.modules[module_name]
    finally:
        os.unlink(settings_file)