"""Download, verify and safely extract the Bang & Kim dataset.

Design points (each one is something that can silently go wrong with a 566 MiB file):

* **Integrity:** the file's size and SHA-256 are checked against the values Mendeley
  publishes; a mismatch deletes nothing but aborts before extraction.
* **Resumable:** bytes are streamed to ``<name>.part`` and a re-run continues with an HTTP
  ``Range`` request, so a dropped Colab connection does not restart the download.
* **Disk gate:** refuses to start if the download plus extraction would leave less than
  ``min_free_gb`` free (the project rule on this laptop is 3 GB).
* **Safe extraction:** archive members whose names are absolute or contain ``..`` are
  rejected (zip-slip), because the archive is untrusted input.
* **Idempotent:** if the extracted dataset is already complete, nothing is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DATASET_URL = (
    "https://data.mendeley.com/public-files/datasets/4h68fmktwh/files/"
    "c7d5c355-15eb-4138-8eba-981c32e38429/file_downloaded"
)
ZIP_NAME = "image_with_captions.zip"
ZIP_SIZE = 593_575_014
ZIP_SHA256 = "d1860ae77b885ad87dfee0f530606a2b227dfc776a50dc55a20024e9a40f7333"
VIA_NAME = "via_region_data_final.json"
IMAGES_SUBDIR = "UAVdata"
EXPECTED_IMAGES = 1431

CHUNK = 1 << 20
USER_AGENT = "dataset-downloader/1.0"  # generic on purpose: no project or user name


class DownloadError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check_free_space(directory: Path, needed_bytes: int, min_free_gb: float) -> None:
    """Raise unless ``needed_bytes`` can be written leaving ``min_free_gb`` free."""
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    remaining = free - needed_bytes
    if remaining < min_free_gb * 1e9:
        raise DownloadError(
            f"not enough disk: {free / 1e9:.2f} GB free, need {needed_bytes / 1e9:.2f} GB, "
            f"would leave {remaining / 1e9:.2f} GB (< {min_free_gb} GB floor)"
        )


def download(url: str, dest: Path, expected_size: int, expected_sha256: str) -> None:
    """Stream ``url`` to ``dest`` (resuming a ``.part`` file) and verify size + SHA-256."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > expected_size:
        part.unlink()
        have = 0
    if have < expected_size:
        # Mendeley's CDN answers the default "Python-urllib/x" client with HTTP 403 (found when
        # this first ran against the real server on Colab), so identify with a neutral name.
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req) as resp:
            if have and resp.status != 206:  # server ignored Range: start over, do not append
                have = 0
            with part.open("ab" if have else "wb") as out:
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
    size = part.stat().st_size
    if size != expected_size:
        raise DownloadError(f"size mismatch: got {size}, expected {expected_size}")
    digest = sha256_file(part)
    if digest != expected_sha256:
        raise DownloadError(f"sha256 mismatch: got {digest}, expected {expected_sha256}")
    part.replace(dest)


def safe_extract(zip_path: Path, out_dir: Path) -> int:
    """Extract ``zip_path`` into ``out_dir``, rejecting unsafe member names. Returns file count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for m in members:
            p = PurePosixPath(m.filename)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in m.filename
                or (p.parts and ":" in p.parts[0])
            ):
                raise DownloadError(f"unsafe path in archive: {m.filename!r}")
            target = (root / m.filename).resolve()
            if root != target and root not in target.parents:
                raise DownloadError(f"path escapes target directory: {m.filename!r}")
        zf.extractall(root)
    return sum(1 for m in members if not m.is_dir())


def is_complete(extracted: Path, expected_images: int = EXPECTED_IMAGES) -> bool:
    images = extracted / IMAGES_SUBDIR
    return (extracted / VIA_NAME).is_file() and (
        images.is_dir() and sum(1 for _ in images.glob("*.jpg")) == expected_images
    )


def ensure_dataset(
    data_dir: Path,
    *,
    url: str = DATASET_URL,
    expected_size: int = ZIP_SIZE,
    expected_sha256: str = ZIP_SHA256,
    expected_images: int = EXPECTED_IMAGES,
    min_free_gb: float = 3.0,
    keep_zip: bool = False,
) -> Path:
    """Make sure ``data_dir/extracted`` holds the full dataset; return that directory."""
    extracted = data_dir / "extracted"
    if is_complete(extracted, expected_images):
        return extracted
    raw = data_dir / "raw"
    zip_path = raw / ZIP_NAME
    if not zip_path.exists():
        # Extraction needs roughly the zip size again (JPEGs barely compress).
        check_free_space(raw, 2 * expected_size, min_free_gb)
        download(url, zip_path, expected_size, expected_sha256)
    elif sha256_file(zip_path) != expected_sha256:
        raise DownloadError(f"{zip_path} exists but its sha256 does not match; delete it")
    safe_extract(zip_path, extracted)
    if not is_complete(extracted, expected_images):
        raise DownloadError("extraction finished but the dataset is incomplete")
    if not keep_zip:
        zip_path.unlink()
    return extracted


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch and verify the UAV captioning dataset.")
    p.add_argument("--data-dir", type=Path, required=True, help="folder OUTSIDE the repo")
    p.add_argument("--min-free-gb", type=float, default=3.0)
    p.add_argument("--keep-zip", action="store_true", help="do not delete the zip after extraction")
    args = p.parse_args(argv)
    try:
        extracted = ensure_dataset(
            args.data_dir, min_free_gb=args.min_free_gb, keep_zip=args.keep_zip
        )
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"dataset ready: {extracted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
