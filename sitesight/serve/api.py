"""HTTP API (FastAPI) in front of any backend that can describe regions and ground phrases.

Endpoints (all upload the image as ``multipart/form-data``):

* ``GET  /health``: status plus the backend description (engine, precision, model).
* ``POST /v1/describe``: ``image`` + ``boxes`` (JSON ``[[x, y, w, h], ...]`` in pixels) ->
  one caption per box, the parsed fields, a confidence, and a per-resource tally.
* ``POST /v1/ground``: ``image`` + ``phrase`` -> boxes for that phrase.

Honest-output rules (same as the rest of the project):

* ``confidence`` is the geometric-mean token probability of the generated caption. It is the
  model's own certainty, **not a calibrated probability** (ECE 0.43 on validation, see
  ``reports/m5``). Use it for ranking and for the ``review`` flag, not as "p(correct)".
* ``review`` is true when the caption could not be parsed into fields or its confidence is below
  ``review_threshold`` (a default of 0.5 that was not tuned on any data).
* Limits are reported, never silent: oversized or non-image uploads, too many boxes and boxes
  outside the image are rejected with a reason; boxes that merely touch the edge are clamped and
  reported in ``warnings``.
* The ``summary`` counts *described regions* per resource, not objects on the site (one box can
  stand for a whole pile).
"""

from __future__ import annotations

import io
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from sitesight.data.caption_fields import parse_caption


class Backend(Protocol):
    """What the API needs from an inference engine (ONNX, PyTorch or a test double)."""

    def describe_region(self, image: Image.Image, bbox: list[float]) -> tuple[str, float]: ...

    def ground(self, image: Image.Image, phrase: str) -> list[tuple[list[float], float]]: ...

    def info(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Limits:
    max_upload_bytes: int = 15_000_000
    max_pixels: int = 40_000_000
    max_boxes: int = 64
    max_phrase_chars: int = 80
    review_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> Limits:
        d = cls()
        return cls(
            max_upload_bytes=int(os.environ.get("SITESIGHT_MAX_UPLOAD_BYTES", d.max_upload_bytes)),
            max_pixels=int(os.environ.get("SITESIGHT_MAX_PIXELS", d.max_pixels)),
            max_boxes=int(os.environ.get("SITESIGHT_MAX_BOXES", d.max_boxes)),
            max_phrase_chars=int(os.environ.get("SITESIGHT_MAX_PHRASE", d.max_phrase_chars)),
            review_threshold=float(
                os.environ.get("SITESIGHT_REVIEW_THRESHOLD", d.review_threshold)
            ),
        )


class ParsedFields(BaseModel):
    resource: str
    count: int | None
    colour: str | None
    state: str


class Region(BaseModel):
    bbox: list[float]
    caption: str
    confidence: float
    fields: ParsedFields | None
    review: bool
    review_reasons: list[str]


class DescribeResponse(BaseModel):
    image_width: int
    image_height: int
    regions: list[Region]
    summary: dict[str, int]
    warnings: list[str]


class GroundBox(BaseModel):
    bbox: list[float]
    confidence: float


class GroundResponse(BaseModel):
    image_width: int
    image_height: int
    phrase: str
    boxes: list[GroundBox]


def decode_image(data: bytes, limits: Limits) -> Image.Image:
    """Open an upload as RGB, rejecting oversized files, non-images and decompression bombs."""
    if len(data) > limits.max_upload_bytes:
        raise HTTPException(413, f"upload larger than {limits.max_upload_bytes} bytes")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.size[0] * im.size[1] > limits.max_pixels:  # header only: nothing decoded yet
                raise HTTPException(413, f"image larger than {limits.max_pixels} pixels")
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, f"not a readable image: {e}") from e


def parse_boxes(
    raw: str, width: int, height: int, limits: Limits
) -> tuple[list[list[float]], list[str]]:
    """Validate ``[[x, y, w, h], ...]``; clamp boxes that overhang the image and say so."""
    try:
        boxes = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"boxes is not valid JSON: {e}") from e
    if not isinstance(boxes, list) or not boxes:
        raise HTTPException(422, "boxes must be a non-empty list of [x, y, w, h]")
    if len(boxes) > limits.max_boxes:
        raise HTTPException(422, f"{len(boxes)} boxes exceeds the limit of {limits.max_boxes}")
    out: list[list[float]] = []
    warnings: list[str] = []
    for i, b in enumerate(boxes):
        if (
            not isinstance(b, list)
            or len(b) != 4
            or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in b)
        ):
            raise HTTPException(422, f"box {i} must be four numbers [x, y, w, h]")
        x, y, w, h = (float(v) for v in b)
        x0, y0, x1, y1 = (
            max(x, 0.0),
            max(y, 0.0),
            min(x + w, float(width)),
            min(y + h, float(height)),
        )
        if w <= 0 or h <= 0 or x1 <= x0 or y1 <= y0:
            raise HTTPException(422, f"box {i} is empty or outside the {width}x{height} image")
        if (x0, y0, x1, y1) != (x, y, x + w, y + h):
            warnings.append(f"box {i} was clamped to the image")
        out.append([x0, y0, x1 - x0, y1 - y0])
    return out, warnings


def describe_regions(
    backend: Backend, img: Image.Image, boxes: list[list[float]], lim: Limits
) -> list[Region]:
    """Caption every (already validated) box; shared by the API and the batch CLI."""
    regions: list[Region] = []
    for box in boxes:
        caption, conf = backend.describe_region(img, box)
        parsed = parse_caption(caption).fields
        reasons = []
        if parsed is None:
            reasons.append("caption_not_parsed")
        if conf < lim.review_threshold:
            reasons.append("low_confidence")
        fields = (
            ParsedFields(
                resource=parsed.resource,
                count=parsed.count,
                colour=parsed.colour,
                state=parsed.state,
            )
            if parsed
            else None
        )
        regions.append(
            Region(
                bbox=box,
                caption=caption,
                confidence=conf,
                fields=fields,
                review=bool(reasons),
                review_reasons=reasons,
            )
        )
    return regions


def summarise(regions: list[Region]) -> Counter[str]:
    """Number of described regions per parsed resource (not a count of objects)."""
    return Counter(r.fields.resource for r in regions if r.fields)


def create_app(backend: Backend, limits: Limits | None = None) -> FastAPI:
    lim = limits or Limits()
    app = FastAPI(title="SiteSight", version="0.7")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "backend": backend.info()}

    @app.post("/v1/describe", response_model=DescribeResponse)
    async def describe(
        image: Annotated[UploadFile, File()], boxes: Annotated[str, Form()]
    ) -> DescribeResponse:
        img = decode_image(await image.read(lim.max_upload_bytes + 1), lim)
        w, h = img.size
        valid, warnings = parse_boxes(boxes, w, h, lim)
        regions = describe_regions(backend, img, valid, lim)
        tally = summarise(regions)
        return DescribeResponse(
            image_width=w,
            image_height=h,
            regions=regions,
            summary=dict(sorted(tally.items())),
            warnings=warnings,
        )

    @app.post("/v1/ground", response_model=GroundResponse)
    async def ground(
        image: Annotated[UploadFile, File()], phrase: Annotated[str, Form()]
    ) -> GroundResponse:
        phrase = phrase.strip()
        if not phrase or len(phrase) > lim.max_phrase_chars:
            raise HTTPException(422, f"phrase must be 1 to {lim.max_phrase_chars} characters")
        img = decode_image(await image.read(lim.max_upload_bytes + 1), lim)
        found = backend.ground(img, phrase)
        return GroundResponse(
            image_width=img.size[0],
            image_height=img.size[1],
            phrase=phrase,
            boxes=[GroundBox(bbox=b, confidence=s) for b, s in found],
        )

    return app
