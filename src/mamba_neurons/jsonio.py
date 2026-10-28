"""UTF-8 JSON read/write helpers shared by library modules and pipeline phases.

Global context: every small result file (results/<model>/*.json, site data) is written through `save_json`.
"""

from __future__ import annotations

import json


def save_json(obj, path) -> None:
    """Pretty UTF-8 JSON writer (ensure_ascii=False keeps token strings readable)."""
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    """Inverse of `save_json`."""
    return json.loads(path.read_text(encoding="utf-8"))
