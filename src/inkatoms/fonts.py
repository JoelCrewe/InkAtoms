"""Make every face of a font collection (.ttc) available to matplotlib.

matplotlib before 3.11 only reads the first face of a .ttc/.otc collection.
On macOS, Helvetica, Helvetica Neue and others ship as collections, so asking
for bold Helvetica silently gives the regular face. use_collection_faces()
extracts the other faces into a local cache folder (for this machine only,
never copied anywhere) and registers them, so fontweight="bold" works again.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _cache_dir():
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "inkatoms" / "fonts"
    return Path.home() / ".cache" / "inkatoms" / "fonts"


def _needs_fix():
    import matplotlib

    major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    return (major, minor) < (3, 11)


def use_collection_faces(families=("Helvetica", "Helvetica Neue"), cache_dir=None, force=False):
    """Register all faces of the system font collections for these families.

    Does nothing on matplotlib >= 3.11 (which reads collections itself) unless
    force=True. Returns the list of font files registered.
    """
    if not (force or _needs_fix()):
        return []
    import matplotlib.font_manager as fm

    try:
        from fontTools.ttLib import TTCollection
    except ImportError:          # fontTools ships with matplotlib, but be safe
        return []
    families = {families} if isinstance(families, str) else set(families)
    sources = sorted({e.fname for e in fm.fontManager.ttflist
                      if e.name in families and e.fname.lower().endswith((".ttc", ".otc"))})
    cache = Path(cache_dir) if cache_dir else _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    known = {e.fname for e in fm.fontManager.ttflist}
    added = []
    for src in sources:
        try:
            collection = TTCollection(src)
        except Exception:
            continue
        for k, face in enumerate(collection.fonts):
            if k == 0:
                continue                 # matplotlib already has the first face
            target = cache / f"{Path(src).stem}-face{k}.ttf"
            if not target.exists():
                face.save(str(target))
            if str(target) not in known:
                fm.fontManager.addfont(str(target))
            added.append(target)
    # forget any lookups made before the new faces were known
    if hasattr(fm.fontManager, "_findfont_cached"):
        fm.fontManager._findfont_cached.cache_clear()
    if hasattr(fm, "_get_font"):
        fm._get_font.cache_clear()
    return added
