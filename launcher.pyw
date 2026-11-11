"""Start Whispa with no console window.

Run with the *base* interpreter's pythonw.exe:  pythonw.exe -I -S launcher.pyw <site-packages>
(Some virtual-environment launchers start console python.exe, which makes Windows open a
terminal window. This avoids them: it adds the environment's packages itself.)
"""

import os
import site
import sys

root = os.path.dirname(os.path.abspath(__file__))
packages = os.path.join(root, ".venv", "Lib", "site-packages")
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    packages = sys.argv.pop(1)
site.addsitedir(packages)  # also processes .pth files (editable install, nvidia DLL paths)
sys.path.insert(0, root)
sys.argv[0] = "whispa"

from whispa.__main__ import main  # noqa: E402

sys.exit(main())
