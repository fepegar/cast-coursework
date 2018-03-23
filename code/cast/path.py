from pathlib import Path


def ensure_dir(path):
    """Make sure that the directory and its parents exists"""
    path = Path(path)
    if path.exists():
        return
    is_dir = not path.suffixes
    if is_dir:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
