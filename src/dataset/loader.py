"""Loading and parsing of the raw Flickr8k text files and image directory.

The Flickr8k caption file (``Flickr8k.token.txt``) stores one caption per
line in the format::

    <image_filename>#<caption_index>\t<caption text>

Example line::

    1000268201_693b08cb0e.jpg#0\tA child in a pink dress is climbing up a set of stairs in an entry way .

The three split files list one image filename per line and together define
the official train/validation/test partition. This module only *reads and
parses*; every judgement about the parsed data lives in ``validate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

CAPTION_COLUMNS = ["image_id", "caption_id", "caption"]


@dataclass
class CaptionParseResult:
    """Parsed contents of ``Flickr8k.token.txt``.

    Attributes:
        captions: DataFrame with columns ``image_id``, ``caption_id``,
            ``caption``. Caption text is preserved verbatim (original casing
            and punctuation) because later phases evaluate against these
            human references.
        malformed_lines: Human-readable descriptions of lines that could not
            be parsed, including the original line number.
        duplicate_keys: ``image_id#caption_id`` keys that appeared more than
            once; only the first occurrence is kept.
    """

    captions: pd.DataFrame
    malformed_lines: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)


@dataclass
class ImageListResult:
    """Parsed contents of one split file.

    Attributes:
        names: Unique image filenames in file order.
        duplicates: Filenames listed more than once in the file.
    """

    names: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)


def load_caption_file(path: Path) -> CaptionParseResult:
    """Parse ``Flickr8k.token.txt`` into a tidy DataFrame.

    Rows with structurally valid keys but empty caption text are kept (the
    validation layer reports them); structurally broken lines are collected
    in ``malformed_lines`` instead of being silently dropped.

    Args:
        path: Path to the token file.

    Returns:
        A :class:`CaptionParseResult`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Caption file not found: {path}")

    records: list[dict] = []
    malformed: list[str] = []
    seen_keys: set[str] = set()
    duplicates: list[str] = []

    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            parsed = _parse_caption_line(line)
            if isinstance(parsed, str):
                malformed.append(f"line {line_number}: {parsed}")
                continue

            image_id, caption_id, caption = parsed
            key = f"{image_id}#{caption_id}"
            if key in seen_keys:
                duplicates.append(key)
                continue
            seen_keys.add(key)
            records.append(
                {"image_id": image_id, "caption_id": caption_id, "caption": caption}
            )

    frame = pd.DataFrame.from_records(records, columns=CAPTION_COLUMNS)
    return CaptionParseResult(
        captions=frame, malformed_lines=malformed, duplicate_keys=duplicates
    )


def _parse_caption_line(line: str) -> tuple[str, int, str] | str:
    """Parse one caption line.

    Returns:
        An ``(image_id, caption_id, caption)`` tuple on success, otherwise a
        human-readable description of the problem.
    """
    if "\t" not in line:
        return f"missing tab separator: {line[:60]!r}"
    key, caption = line.split("\t", 1)
    if "#" not in key:
        return f"missing '#<index>' suffix: {key[:60]!r}"
    image_id, suffix = key.rsplit("#", 1)
    if not image_id.strip():
        return f"empty image id: {key[:60]!r}"
    if not suffix.isdigit():
        return f"non-numeric caption index {suffix!r}: {key[:60]!r}"
    return image_id, int(suffix), caption.strip()


def read_image_list(path: Path) -> ImageListResult:
    """Read one split file (one image filename per line).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    names: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            name = raw_line.strip()
            if not name:
                continue
            if name in seen:
                duplicates.append(name)
                continue
            seen.add(name)
            names.append(name)
    return ImageListResult(names=names, duplicates=duplicates)


def list_image_files(images_dir: Path, extensions: Iterable[str] = (".jpg",)) -> list[str]:
    """List image filenames in ``images_dir`` matching ``extensions``.

    Raises:
        NotADirectoryError: If ``images_dir`` does not exist.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {images_dir}")
    allowed = {ext.lower() for ext in extensions}
    return sorted(
        entry.name
        for entry in images_dir.iterdir()
        if entry.is_file() and entry.suffix.lower() in allowed
    )


def find_missing_images(
    images_dir: Path, image_ids: Iterable[str], extensions: Iterable[str] = (".jpg",)
) -> list[str]:
    """Return the sorted IDs from ``image_ids`` that have no file on disk."""
    present = set(list_image_files(images_dir, extensions))
    return sorted(set(image_ids) - present)


def find_unreadable_images(images_dir: Path, image_ids: Iterable[str]) -> list[str]:
    """Return sorted IDs whose file exists on disk but cannot be opened.

    Uses :meth:`PIL.Image.Image.verify`, which parses the file header and
    internal consistency markers without decoding pixel data, so it is fast
    enough to run over the whole Flickr8k collection.
    """
    images_dir = Path(images_dir)
    unreadable: list[str] = []
    for image_id in sorted(set(image_ids)):
        file_path = images_dir / image_id
        if not file_path.exists():
            continue
        try:
            with Image.open(file_path) as image:
                image.verify()
        except Exception:  # noqa: BLE001 - PIL raises several exception types
            unreadable.append(image_id)
    return unreadable
