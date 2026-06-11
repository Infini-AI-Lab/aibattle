"""Ensure tests import the in-repo ``aibattle`` package.

This repository may coexist with a separate editable install of ``aibattle``
pointing elsewhere; prepend this repo's ``src`` so tests always exercise the
local source tree rather than the installed copy.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
