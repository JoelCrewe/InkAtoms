"""Work around matplotlib < 3.11 only reading the first face of .ttc fonts.

On macOS this makes bold Helvetica come out regular. The other faces are written
to a local cache and registered with the font manager.
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
    except ImportError:
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
                continue                 # already known to matplotlib
            target = cache / f"{Path(src).stem}-face{k}.ttf"
            if not target.exists():
                face.save(str(target))
            if str(target) not in known:
                fm.fontManager.addfont(str(target))
            added.append(target)
    if hasattr(fm.fontManager, "_findfont_cached"):
        fm.fontManager._findfont_cached.cache_clear()
    if hasattr(fm, "_get_font"):
        fm._get_font.cache_clear()
    return added
