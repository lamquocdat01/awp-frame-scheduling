# -*- coding: utf-8 -*-
"""v3_io.py -- atomic file output for every v3 script (C7 root fix).

Every result file is first written to <name>.tmp in the same folder and then moved over the target with
os.replace (atomic on NTFS and POSIX), so an interrupted or failing writer never leaves a half-written
JSON/CSV/TeX/PNG behind and a concurrent reader sees either the old or the new file, never a prefix.
Text is always UTF-8 unless stated (never the Windows locale code page).
"""
import contextlib
import os
import shutil
from pathlib import Path


def _tmp(path):
    path = Path(path)
    return path, path.with_name(path.name + ".tmp")


def _commit(tmp, path):
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def atomic_open(path, mode="w", newline="", encoding="utf-8"):
    """with atomic_open(p) as fh: csv.writer(fh)...   (text mode; binary with mode='wb')."""
    path, tmp = _tmp(path)
    kw = {} if "b" in mode else {"newline": newline, "encoding": encoding}
    try:
        with open(tmp, mode, **kw) as fh:
            yield fh
        _commit(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def atomic_write_text(path, text, encoding="utf-8"):
    with atomic_open(path, newline=None, encoding=encoding) as fh:
        fh.write(text)


def atomic_write_bytes(path, data):
    with atomic_open(path, mode="wb") as fh:
        fh.write(data)


def atomic_copy(src, dst):
    with open(src, "rb") as fi, atomic_open(dst, mode="wb") as fo:
        shutil.copyfileobj(fi, fo)
    shutil.copymode(src, dst)


def save_fig(fig, path, **kw):
    """fig.savefig through a temporary file (format taken from the target suffix)."""
    path = Path(path)
    kw.setdefault("format", path.suffix.lstrip("."))
    with atomic_open(path, mode="wb") as fh:
        fig.savefig(fh, **kw)
