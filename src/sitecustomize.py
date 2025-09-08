"""
Ensure vendored dependencies are importable without per-module sys.path hacks.

This file is auto-imported by Python if present on sys.path. We add the
`vendor/circuit_tracer` directory so `import circuit_tracer` works seamlessly
in this project without modifying each module.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_VENDOR_CT = os.path.join(_ROOT, "vendor", "circuit_tracer")

if os.path.isdir(_VENDOR_CT) and _VENDOR_CT not in sys.path:
    sys.path.append(_VENDOR_CT)

