"""Qt-free document model of the review tool: layers, edits, undo, audit, export.

Keeping this free of Qt means every rule (clipping, flag refresh, blind-mode hiding, undo) is
unit-tested without a display; the GUI only forwards user gestures to these methods.

Layers
------
* ``original``: the curated ground truth, never modified.
* ``edit``: a deep copy the user edits. In *review* mode (and plain editing) it starts as a copy
  of the original; in *blind* audit mode the sample images start **empty** and the original is
  hidden for them until the image is marked done, so the auditor labels independently.
* predictions: any number of :class:`PredictionSet` (one per model), read-only.

Every edit goes through :class:`Document` so that the result always satisfies
:func:`sitesight.data.validate.validate_dataset`: boxes are rounded to integers and clipped to
the image, captions are normalised and re-parsed (category and fields refreshed, ``unparsed`` /
``empty_caption`` / ``duplicate`` flags recomputed). Undo/redo stores whole-image snapshots,
which is simple and obviously correct (an image has at most a few dozen regions).
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sitesight.data.audit import AuditFile, AuditImage
from sitesight.data.caption_fields import CATEGORY_IDS, CATEGORY_NAMES, parse_caption
from sitesight.data.predictions import PredictionSet, validate_predictions
from sitesight.data.schema import (
    FLAG_CLIPPED,
    FLAG_DUPLICATE,
    FLAG_EMPTY_CAPTION,
    FLAG_UNPARSED,
    AnnotationRecord,
    CategoryRecord,
    Dataset,
    ImageRecord,
)
from sitesight.data.validate import validate_dataset

MIN_BOX = 2  # smallest box side the tool allows, in pixels
SOURCE_ORIGINAL = "original"
SOURCE_EDIT = "edit"
PRED_PREFIX = "pred:"


class DocumentError(Exception):
    pass


def clip_box(bbox: tuple[float, float, float, float], width: int, height: int) -> list[int]:
    """Round to integers and clip to the image; enforce a minimum size.

    Raises :class:`DocumentError` if nothing of the box remains inside the image.
    """
    x0, y0 = round(bbox[0]), round(bbox[1])
    x1, y1 = round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
        raise DocumentError(f"box too small or outside the image: {bbox}")
    return [x0, y0, x1 - x0, y1 - y0]


def refresh_annotation(a: AnnotationRecord) -> None:
    """Recompute everything derived from the caption (category, fields, caption flags)."""
    a.caption = " ".join(a.caption_raw.split())
    a.flags = [f for f in a.flags if f not in (FLAG_EMPTY_CAPTION, FLAG_UNPARSED, FLAG_CLIPPED)]
    a.bbox_raw = None
    a.area = a.bbox[2] * a.bbox[3]
    a.category_id, a.fields = None, None
    if not a.caption:
        a.flags.append(FLAG_EMPTY_CAPTION)
        return
    res = parse_caption(a.caption)
    if res.fields is None:
        a.flags.append(FLAG_UNPARSED)
    else:
        a.category_id = CATEGORY_IDS[res.fields.resource]
        a.fields = asdict(res.fields)


def refresh_duplicates(annotations: list[AnnotationRecord]) -> None:
    """Re-derive the ``duplicate`` flag for the annotations of ONE image (first one is kept)."""
    seen: set[tuple[tuple[int, ...], str]] = set()
    for a in annotations:
        a.flags = [f for f in a.flags if f != FLAG_DUPLICATE]
        key = (tuple(a.bbox), a.caption)
        if key in seen:
            a.flags.append(FLAG_DUPLICATE)
        seen.add(key)


class Document:
    def __init__(self, original: Dataset, images_dir: Path | None = None) -> None:
        self.original = original
        self.edit: Dataset = copy.deepcopy(original)
        if not self.edit.categories:
            self.edit.categories = [CategoryRecord(id=i, name=n) for n, i in CATEGORY_IDS.items()]
        self.images_dir = images_dir
        self.predictions: dict[str, PredictionSet] = {}
        self.audit: AuditFile | None = None
        self._by_name = {i.file_name: i for i in self.edit.images}
        self._undo: list[tuple[str, list[AnnotationRecord]]] = []
        self._redo: list[tuple[str, list[AnnotationRecord]]] = []
        self.dirty = False

    # --- lookups ---------------------------------------------------------------------

    def image(self, file_name: str) -> ImageRecord:
        try:
            return self._by_name[file_name]
        except KeyError:
            raise DocumentError(f"unknown image {file_name}") from None

    def regions(self, file_name: str, source: str) -> list[AnnotationRecord]:
        """Annotations of an image from a source layer (blind-mode hiding enforced here)."""
        img = self.image(file_name)
        if source == SOURCE_EDIT:
            return [a for a in self.edit.annotations if a.image_id == img.id]
        if source == SOURCE_ORIGINAL:
            if not self.original_visible(file_name):
                return []
            return [a for a in self.original.annotations if a.image_id == img.id]
        raise DocumentError(f"unknown annotation source {source!r}")

    def original_visible(self, file_name: str) -> bool:
        """False for a not-yet-finished image in blind audit mode (the auditor must not see it)."""
        if self.audit is None or self.audit.mode != "blind" or file_name not in self.audit.sample:
            return True
        return self.audit.images.get(file_name, AuditImage()).status == "done"

    def sources(self) -> list[str]:
        return [SOURCE_ORIGINAL, SOURCE_EDIT, *[PRED_PREFIX + m for m in sorted(self.predictions)]]

    # --- predictions -----------------------------------------------------------------

    def add_predictions(self, pset: PredictionSet) -> None:
        problems = validate_predictions(pset, self.original)
        if problems:
            raise DocumentError(f"invalid predictions ({len(problems)}): {problems[0]}")
        self.predictions[pset.model_name] = pset

    # --- editing ---------------------------------------------------------------------

    def _snapshot(self, file_name: str) -> list[AnnotationRecord]:
        return copy.deepcopy(self.regions(file_name, SOURCE_EDIT))

    def _replace(self, file_name: str, anns: list[AnnotationRecord]) -> None:
        img = self.image(file_name)
        self.edit.annotations = [a for a in self.edit.annotations if a.image_id != img.id] + anns
        self.edit.annotations.sort(key=lambda a: a.id)

    def _commit(self, file_name: str, before: list[AnnotationRecord]) -> None:
        after = self.regions(file_name, SOURCE_EDIT)
        refresh_duplicates(after)
        self._undo.append((file_name, before))
        self._redo.clear()
        self.dirty = True

    def _find(self, ann_id: int) -> tuple[str, AnnotationRecord]:
        for a in self.edit.annotations:
            if a.id == ann_id:
                name = next(i.file_name for i in self.edit.images if i.id == a.image_id)
                return name, a
        raise DocumentError(f"no annotation {ann_id} in the edit layer")

    def add_region(
        self, file_name: str, bbox: tuple[float, float, float, float], caption: str
    ) -> int:
        img = self.image(file_name)
        before = self._snapshot(file_name)
        a = AnnotationRecord(
            id=max((x.id for x in self.edit.annotations), default=0) + 1,
            image_id=img.id,
            bbox=clip_box(bbox, img.width, img.height),
            area=0,
            caption="",
            caption_raw=caption,
            source={"origin": "viewer"},
        )
        refresh_annotation(a)
        self.edit.annotations.append(a)
        self._commit(file_name, before)
        return a.id

    def delete_region(self, ann_id: int) -> None:
        name, a = self._find(ann_id)
        before = self._snapshot(name)
        self.edit.annotations.remove(a)
        self._commit(name, before)

    def set_caption(self, ann_id: int, caption: str) -> None:
        name, a = self._find(ann_id)
        before = self._snapshot(name)
        a.caption_raw = caption
        refresh_annotation(a)
        a.source = {**a.source, "edited": "caption"}
        self._commit(name, before)

    def set_bbox(self, ann_id: int, bbox: tuple[float, float, float, float]) -> None:
        name, a = self._find(ann_id)
        img = self.image(name)
        new = clip_box(bbox, img.width, img.height)
        before = self._snapshot(name)
        a.bbox = new
        refresh_annotation(a)
        a.source = {**a.source, "edited": "box"}
        self._commit(name, before)

    def undo(self) -> bool:
        if not self._undo:
            return False
        name, before = self._undo.pop()
        self._redo.append((name, self._snapshot(name)))
        self._replace(name, before)
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        name, after = self._redo.pop()
        self._undo.append((name, self._snapshot(name)))
        self._replace(name, after)
        self.dirty = True
        return True

    # --- audit -----------------------------------------------------------------------

    def start_audit(self, audit: AuditFile) -> None:
        """Begin (or resume) an audit. Blind mode empties the edit layer for sample images that
        are not finished yet, so the auditor draws from scratch."""
        for name in audit.sample:
            self.image(name)  # fail early on an unknown image
        self.audit = audit
        for name in audit.sample:
            entry = audit.images.setdefault(name, AuditImage())
            if entry.annotations or entry.status != "todo":
                # Resuming: restore exactly what the auditor saved (even an empty list).
                self._load_audit_annotations(name, entry)
            elif audit.mode == "blind":
                self._replace(name, [])  # fresh blind image: start from nothing
            # review mode, untouched image: keep the copy of the original to correct
        self._undo.clear()
        self._redo.clear()

    def _load_audit_annotations(self, name: str, entry: AuditImage) -> None:
        img = self.image(name)
        next_id = max((a.id for a in self.edit.annotations), default=0) + 1
        anns = []
        for k, x in enumerate(entry.annotations):
            a = AnnotationRecord(
                id=next_id + k,
                image_id=img.id,
                bbox=[round(v) for v in x["bbox"]],
                area=0,
                caption="",
                caption_raw=x["caption"],
                source={"origin": "audit"},
            )
            refresh_annotation(a)
            anns.append(a)
        refresh_duplicates(anns)
        self._replace(name, anns)

    def _require_audit(self) -> AuditFile:
        if self.audit is None:
            raise DocumentError("no audit loaded")
        return self.audit

    def sync_audit_image(self, file_name: str) -> None:
        """Copy the edit-layer regions of an image into the audit file."""
        audit = self._require_audit()
        entry = audit.images.setdefault(file_name, AuditImage())
        entry.annotations = [
            {"bbox": list(a.bbox), "caption": a.caption_raw}
            for a in self.regions(file_name, SOURCE_EDIT)
        ]

    def mark_audit(self, file_name: str, status: str, notes: str | None = None) -> None:
        if status not in ("todo", "done", "skipped"):
            raise DocumentError(f"bad status {status!r}")
        audit = self._require_audit()
        if file_name not in audit.sample:
            raise DocumentError(f"{file_name} is not in the audit sample")
        self.sync_audit_image(file_name)
        entry = audit.images[file_name]
        entry.status = status
        if notes is not None:
            entry.notes = notes
        self.dirty = True

    def next_todo(self, after: str | None = None) -> str | None:
        audit = self._require_audit()
        order = audit.sample
        start = order.index(after) + 1 if after in order else 0
        for name in order[start:] + order[:start]:
            if audit.images.get(name, AuditImage()).status == "todo":
                return name
        return None

    def audit_progress(self) -> tuple[int, int]:
        audit = self._require_audit()
        done = sum(1 for n in audit.sample if audit.images.get(n, AuditImage()).status != "todo")
        return done, len(audit.sample)

    # --- export ----------------------------------------------------------------------

    def export_edit(self, path: Path) -> list[str]:
        """Write the whole edit layer in the SiteSight schema. Returns validation problems
        (the file is only written when there are none)."""
        problems = [f"{v.code}: {v.detail}" for v in validate_dataset(self.edit)]
        if problems:
            return problems
        self.edit.save(path)
        self.dirty = False
        return []

    def export_audit_dataset(self, path: Path) -> list[str]:
        """Write only the auditor's regions for finished images, in the SiteSight schema."""
        audit = self._require_audit()
        names = set(audit.done())
        for n in names:
            self.sync_audit_image(n)
        sub = Dataset(
            info={**self.edit.info, "audit_mode": audit.mode}, categories=self.edit.categories
        )
        sub.images = [i for i in self.edit.images if i.file_name in names]
        ids = {i.id for i in sub.images}
        sub.annotations = [a for a in self.edit.annotations if a.image_id in ids]
        problems = [f"{v.code}: {v.detail}" for v in validate_dataset(sub)]
        if problems:
            return problems
        sub.save(path)
        return []


# --- listing / filtering ------------------------------------------------------------------


def list_images(
    doc: Document,
    split: str | None = None,
    text: str | None = None,
    sample_only: bool = False,
    status: str | None = None,
) -> list[str]:
    """File names matching the filters, in numeric order."""
    names = []
    for img in doc.original.images:
        if split and img.split != split:
            continue
        if text and text.lower() not in img.file_name.lower():
            continue
        if sample_only and (doc.audit is None or img.file_name not in doc.audit.sample):
            continue
        if status and (
            doc.audit is None
            or img.file_name not in doc.audit.sample  # only sample images have an audit status
            or doc.audit.images.get(img.file_name, AuditImage()).status != status
        ):
            continue
        names.append(img.file_name)
    return sorted(names, key=lambda n: (int(Path(n).stem) if Path(n).stem.isdigit() else 10**9, n))


def caption_vocabulary(ds: Dataset, limit: int = 1000) -> list[str]:
    """Most frequent distinct captions (for the caption editor's autocomplete)."""
    c = Counter(a.caption for a in ds.annotations if a.caption and FLAG_UNPARSED not in a.flags)
    return [cap for cap, _ in c.most_common(limit)]


def resource_names() -> list[str]:
    return list(CATEGORY_NAMES)
