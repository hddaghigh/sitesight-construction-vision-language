"""Downloader tests against a real local HTTP server serving a real zip (no mocks)."""

from __future__ import annotations

import hashlib
import re
import threading
import zipfile
from collections.abc import Iterator
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sitesight.data.download import (
    DownloadError,
    check_free_space,
    download,
    ensure_dataset,
    safe_extract,
    sha256_file,
)
from tests.conftest import build_fixture


class _Handler(BaseHTTPRequestHandler):
    """Serves one file; honours ``Range: bytes=N-`` unless ``ignore_range`` is set."""

    def __init__(self, payload: bytes, ignore_range: bool, *args: object, **kw: object) -> None:
        self.payload = payload
        self.ignore_range = ignore_range
        super().__init__(*args, **kw)  # type: ignore[arg-type]

    def do_GET(self) -> None:
        # Behave like the real host: the default "Python-urllib/x" client gets 403 Forbidden.
        if self.headers.get("User-Agent", "").startswith("Python-urllib"):
            self.send_error(403, "Forbidden")
            return
        m = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
        if m and not self.ignore_range:
            start = int(m.group(1))
            body = self.payload[start:]
            self.send_response(206)
        else:
            body = self.payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # silence test output
        pass


def _serve(payload: bytes, ignore_range: bool = False) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Handler, payload, ignore_range))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/file"


@pytest.fixture
def dataset_zip(tmp_path: Path) -> Iterator[tuple[bytes, Path]]:
    """A zip shaped like the real archive (UAVdata/*.jpg + the VIA json)."""
    src = tmp_path / "src"
    src.mkdir()
    via, images = build_fixture(src)
    zpath = tmp_path / "made.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(via, via.name)
        for img in sorted(images.glob("*.jpg")):
            zf.write(img, f"UAVdata/{img.name}")
    yield zpath.read_bytes(), zpath


def test_server_rejects_default_python_client_but_downloader_is_accepted(
    tmp_path: Path, dataset_zip: tuple[bytes, Path]
) -> None:
    """Regression test for the 403 seen on the real host: our requests must not look like
    the default urllib client."""
    import urllib.error
    import urllib.request

    payload, _ = dataset_zip
    server, url = _serve(payload)
    try:
        default_client = urllib.request.Request(url, headers={"User-Agent": "Python-urllib/3.13"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(default_client)
        assert err.value.code == 403  # the test server really does reject that client
        dest = tmp_path / "ok.zip"
        download(url, dest, len(payload), hashlib.sha256(payload).hexdigest())  # must succeed
    finally:
        server.shutdown()
    assert dest.read_bytes() == payload


def test_download_verifies_and_renames(tmp_path: Path, dataset_zip: tuple[bytes, Path]) -> None:
    payload, _ = dataset_zip
    server, url = _serve(payload)
    try:
        dest = tmp_path / "d" / "f.zip"
        dest.parent.mkdir()
        download(url, dest, len(payload), hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
    assert dest.read_bytes() == payload
    assert not dest.with_name("f.zip.part").exists()


def test_download_resumes_with_range(tmp_path: Path, dataset_zip: tuple[bytes, Path]) -> None:
    payload, _ = dataset_zip
    dest = tmp_path / "f.zip"
    dest.with_name("f.zip.part").write_bytes(payload[:100])  # interrupted earlier
    server, url = _serve(payload)
    try:
        download(url, dest, len(payload), hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
    assert dest.read_bytes() == payload  # would be corrupt if the tail were appended twice


def test_download_restarts_if_server_ignores_range(
    tmp_path: Path, dataset_zip: tuple[bytes, Path]
) -> None:
    payload, _ = dataset_zip
    dest = tmp_path / "f.zip"
    dest.with_name("f.zip.part").write_bytes(payload[:100])
    server, url = _serve(payload, ignore_range=True)
    try:
        download(url, dest, len(payload), hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
    assert dest.read_bytes() == payload


def test_download_rejects_wrong_hash_and_size(
    tmp_path: Path, dataset_zip: tuple[bytes, Path]
) -> None:
    payload, _ = dataset_zip
    server, url = _serve(payload)
    try:
        with pytest.raises(DownloadError, match="sha256 mismatch"):
            download(url, tmp_path / "a.zip", len(payload), "0" * 64)
        with pytest.raises(DownloadError, match="size mismatch"):
            download(url, tmp_path / "b.zip", len(payload) + 1, hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
    assert not (tmp_path / "a.zip").exists()  # a bad file is never promoted to its final name


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "x")
    with pytest.raises(DownloadError, match="unsafe path"):
        safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_absolute_and_drive_paths_rejected(tmp_path: Path) -> None:
    for name in ("/abs.txt", "C:/win.txt"):
        z = tmp_path / "z.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr(name, "x")
        with pytest.raises(DownloadError, match="unsafe path"):
            safe_extract(z, tmp_path / "out")


def test_free_space_gate(tmp_path: Path) -> None:
    check_free_space(tmp_path, 1, min_free_gb=0.0)  # passes
    with pytest.raises(DownloadError, match="not enough disk"):
        check_free_space(tmp_path, 1, min_free_gb=1e9)  # a billion GB floor can never be met


def test_ensure_dataset_end_to_end_and_idempotent(
    tmp_path: Path, dataset_zip: tuple[bytes, Path]
) -> None:
    payload, _ = dataset_zip
    sha = hashlib.sha256(payload).hexdigest()
    server, url = _serve(payload)
    data_dir = tmp_path / "data"
    try:
        extracted = ensure_dataset(
            data_dir,
            url=url,
            expected_size=len(payload),
            expected_sha256=sha,
            expected_images=3,
            min_free_gb=0.0,
        )
    finally:
        server.shutdown()
    assert (extracted / "via_region_data_final.json").is_file()
    assert len(list((extracted / "UAVdata").glob("*.jpg"))) == 3
    assert not (data_dir / "raw" / "image_with_captions.zip").exists()  # deleted by default
    # Server is gone: a second call must not touch the network.
    again = ensure_dataset(
        data_dir,
        url="http://127.0.0.1:1/unreachable",
        expected_size=len(payload),
        expected_sha256=sha,
        expected_images=3,
        min_free_gb=0.0,
    )
    assert again == extracted


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()
