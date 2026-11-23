"""Document-model tests (no Qt). Built on the real parser path from the M1 fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from sitesight.data.audit import AuditFile, AuditImage
from sitesight.data.caption_fields import apply_caption_fields
from sitesight.data.predictions import Prediction, PredictionSet
from sitesight.data.schema import FLAG_DUPLICATE, FLAG_EMPTY_CAPTION, FLAG_UNPARSED, Dataset
from sitesight.data.validate import validate_dataset
from sitesight.data.via_parse import parse_via
from sitesight.viewer.state import (
    PRED_PREFIX,
    SOURCE_EDIT,
    SOURCE_ORIGINAL,
    Document,
    DocumentError,
    caption_vocabulary,
    clip_box,
    list_images,
)

W = "A worker is located on soil"


@pytest.fixture
def doc(via_fixture: tuple[Path, Path]) -> Document:
    ds, _ = parse_via(*via_fixture)
    apply_caption_fields(ds)
    for i, img in enumerate(ds.images):
        img.split = ("train", "val", "test")[i]
    return Document(ds, via_fixture[1])


def test_clip_box() -> None:
    assert clip_box((-5, 50, 30, 20), 100, 60) == [0, 50, 25, 10]
    # rounding: x0=1, y0=3, x1=round(11.6)=12, y1=round(12.2)=12 -> [1, 3, 11, 9]
    assert clip_box((1.4, 2.6, 10.2, 9.6), 100, 60) == [1, 3, 11, 9]
    with pytest.raises(DocumentError):
        clip_box((10, 10, 1, 1), 100, 60)  # smaller than the 2 px minimum
    with pytest.raises(DocumentError):
        clip_box((200, 10, 20, 20), 100, 60)  # entirely outside


def test_initial_layers(doc: Document) -> None:
    assert len(doc.regions("1.jpg", SOURCE_ORIGINAL)) == 4  # r0, r1, r2, r3 survived parsing
    assert doc.regions("1.jpg", SOURCE_EDIT) == doc.regions("1.jpg", SOURCE_ORIGINAL)
    assert doc.regions("1.jpg", SOURCE_EDIT) is not doc.regions("1.jpg", SOURCE_ORIGINAL)
    assert validate_dataset(doc.edit) == [] and not doc.dirty
    with pytest.raises(DocumentError):
        doc.image("nope.jpg")
    with pytest.raises(DocumentError):
        doc.regions("1.jpg", "bogus")


def test_add_region_derives_everything_from_the_caption(doc: Document) -> None:
    n_orig = len(doc.original.annotations)
    aid = doc.add_region(
        "2.jpg", (10.4, 10.2, 20, 20), "  Two  orange excavators are digging soil "
    )
    a = next(x for x in doc.edit.annotations if x.id == aid)
    assert a.bbox == [10, 10, 20, 20] and a.area == 400
    assert a.caption == "Two orange excavators are digging soil" and a.caption_raw.startswith(
        "  Two"
    )
    assert a.fields is not None and a.fields["count"] == 2 and a.category_id is not None
    assert a.source == {"origin": "viewer"} and doc.dirty
    assert len(doc.original.annotations) == n_orig  # the original layer is untouched
    assert validate_dataset(doc.edit) == []


def test_unparsed_and_empty_captions_are_flagged_not_rejected(doc: Document) -> None:
    g = doc.add_region("2.jpg", (0, 0, 10, 10), "some garbage")
    e = doc.add_region("2.jpg", (20, 0, 10, 10), "   ")
    by_id = {a.id: a for a in doc.edit.annotations}
    assert FLAG_UNPARSED in by_id[g].flags and by_id[g].category_id is None
    assert by_id[e].flags == [FLAG_EMPTY_CAPTION] and by_id[e].category_id is None
    assert validate_dataset(doc.edit) == []


def test_set_caption_set_bbox_and_delete(doc: Document) -> None:
    aid = doc.regions("1.jpg", SOURCE_EDIT)[1].id  # the clipped fence box (bbox_raw set)
    assert doc.regions("1.jpg", SOURCE_EDIT)[1].bbox_raw is not None
    doc.set_bbox(aid, (-3, 48, 40, 30))
    a = next(x for x in doc.edit.annotations if x.id == aid)
    assert a.bbox == [0, 48, 37, 12] and a.bbox_raw is None and "clipped" not in a.flags
    assert a.area == 37 * 12 and a.source["edited"] == "box"
    doc.set_caption(aid, "Wood fences are installed in a line")
    a = next(x for x in doc.edit.annotations if x.id == aid)
    assert a.fields is not None and a.fields["resource"] == "wood_fence"
    n = len(doc.edit.annotations)
    doc.delete_region(aid)
    assert len(doc.edit.annotations) == n - 1 and validate_dataset(doc.edit) == []
    with pytest.raises(DocumentError):
        doc.delete_region(aid)
    with pytest.raises(DocumentError):
        doc.set_bbox(doc.regions("1.jpg", SOURCE_EDIT)[0].id, (500, 500, 5, 5))


def test_duplicate_flag_follows_edits(doc: Document) -> None:
    first = doc.add_region("2.jpg", (5, 5, 20, 20), W)
    second = doc.add_region("2.jpg", (5, 5, 20, 20), W)
    by_id = {a.id: a for a in doc.edit.annotations}
    assert FLAG_DUPLICATE not in by_id[first].flags and FLAG_DUPLICATE in by_id[second].flags
    doc.delete_region(first)
    assert FLAG_DUPLICATE not in {a.id: a for a in doc.edit.annotations}[second].flags
    assert validate_dataset(doc.edit) == []


def test_undo_redo(doc: Document) -> None:
    before = Dataset.from_dict(doc.edit.to_dict()).to_dict()
    assert doc.undo() is False
    aid = doc.add_region("2.jpg", (5, 5, 20, 20), W)
    after_add = doc.edit.to_dict()
    doc.set_caption(aid, "Two workers are located on soil")
    doc.delete_region(aid)
    assert doc.undo() and doc.undo() and doc.edit.to_dict() == after_add
    assert doc.undo() and doc.edit.to_dict() == before
    assert doc.undo() is False
    assert doc.redo() and doc.edit.to_dict() == after_add
    doc.add_region("2.jpg", (30, 30, 10, 10), W)  # a new edit invalidates redo
    assert doc.redo() is False
    assert validate_dataset(doc.edit) == []


def test_export_edit_roundtrip_and_refusal(doc: Document, tmp_path: Path) -> None:
    doc.add_region("2.jpg", (5, 5, 20, 20), W)
    out = tmp_path / "edited.json"
    assert doc.export_edit(out) == [] and not doc.dirty
    assert Dataset.load(out).to_dict() == doc.edit.to_dict()
    doc.edit.annotations[0].bbox = [0, 0, 500, 500]  # corrupt: outside the 100x60 image
    bad = tmp_path / "bad.json"
    problems = doc.export_edit(bad)
    assert problems and any("box_out_of_bounds" in p for p in problems) and not bad.exists()


def test_predictions(doc: Document) -> None:
    ok = PredictionSet("m1", [Prediction("1.jpg", [10, 10, 20, 20], 0.9, W)])
    doc.add_predictions(ok)
    doc.add_predictions(PredictionSet("a-model", [Prediction("2.jpg", [0, 0, 5, 5], 0.2)]))
    assert doc.sources() == [
        SOURCE_ORIGINAL,
        SOURCE_EDIT,
        PRED_PREFIX + "a-model",
        PRED_PREFIX + "m1",
    ]
    with pytest.raises(DocumentError, match="invalid predictions"):
        doc.add_predictions(PredictionSet("bad", [Prediction("1.jpg", [0, 0, 500, 5], 0.5)]))
    assert "bad" not in doc.predictions


def test_blind_audit_hides_original_until_done(doc: Document, tmp_path: Path) -> None:
    af = AuditFile(mode="blind", sample=["1.jpg", "10.jpg"])
    doc.start_audit(af)
    assert doc.regions("1.jpg", SOURCE_EDIT) == [] and doc.regions("10.jpg", SOURCE_EDIT) == []
    assert not doc.original_visible("1.jpg") and doc.regions("1.jpg", SOURCE_ORIGINAL) == []
    assert doc.original_visible("2.jpg")  # not in the sample
    assert doc.next_todo() == "1.jpg" and doc.audit_progress() == (0, 2)
    doc.add_region("1.jpg", (10, 10, 20, 20), W)
    doc.mark_audit("1.jpg", "done", notes="clear image")
    assert doc.original_visible("1.jpg") and len(doc.regions("1.jpg", SOURCE_ORIGINAL)) == 4
    entry = af.images["1.jpg"]
    assert entry.status == "done" and entry.notes == "clear image"
    assert entry.annotations == [{"bbox": [10, 10, 20, 20], "caption": W}]
    assert doc.next_todo(after="1.jpg") == "10.jpg" and doc.audit_progress() == (1, 2)
    out = tmp_path / "audit_ds.json"
    assert doc.export_audit_dataset(out) == []
    sub = Dataset.load(out)
    assert [i.file_name for i in sub.images] == ["1.jpg"] and len(sub.annotations) == 1
    assert sub.info["audit_mode"] == "blind" and validate_dataset(sub) == []


def test_review_audit_starts_from_original_and_resume_restores(
    doc: Document, via_fixture: tuple[Path, Path]
) -> None:
    af = AuditFile(mode="review", sample=["1.jpg"])
    doc.start_audit(af)
    assert len(doc.regions("1.jpg", SOURCE_EDIT)) == 4 and doc.original_visible("1.jpg")
    for a in list(doc.regions("1.jpg", SOURCE_EDIT)):
        doc.delete_region(a.id)  # the auditor decides nothing there is correct
    doc.mark_audit("1.jpg", "done")
    assert af.images["1.jpg"].annotations == [] and af.images["1.jpg"].status == "done"
    # A fresh Document resuming the saved audit must restore the EMPTY list, not the original.
    ds2, _ = parse_via(*via_fixture)
    apply_caption_fields(ds2)
    doc2 = Document(ds2)
    doc2.start_audit(AuditFile.from_dict(af.to_dict()))
    assert doc2.regions("1.jpg", SOURCE_EDIT) == []
    # And a saved non-empty blind audit is restored with parsed fields.
    af3 = AuditFile(
        mode="blind",
        sample=["1.jpg"],
        images={"1.jpg": AuditImage("done", [{"bbox": [10, 10, 20, 20], "caption": W}])},
    )
    doc3 = Document(parse_via(*via_fixture)[0])
    doc3.start_audit(af3)
    restored = doc3.regions("1.jpg", SOURCE_EDIT)
    assert (
        len(restored) == 1
        and restored[0].bbox == [10, 10, 20, 20]
        and restored[0].fields is not None
    )


def test_audit_errors(doc: Document) -> None:
    with pytest.raises(DocumentError):
        doc.mark_audit("1.jpg", "done")  # no audit loaded
    with pytest.raises(DocumentError):
        doc.start_audit(AuditFile(mode="blind", sample=["nope.jpg"]))
    doc.start_audit(AuditFile(mode="review", sample=["1.jpg"]))
    with pytest.raises(DocumentError):
        doc.mark_audit("2.jpg", "done")  # not in the sample
    with pytest.raises(DocumentError):
        doc.mark_audit("1.jpg", "finished")


def test_list_images_filters_and_vocabulary(doc: Document) -> None:
    assert list_images(doc) == ["1.jpg", "2.jpg", "10.jpg"]  # numeric order
    assert list_images(doc, split="val") == ["2.jpg"]
    assert list_images(doc, text="10") == ["10.jpg"]
    assert list_images(doc, sample_only=True) == []
    af = AuditFile(mode="review", sample=["10.jpg", "1.jpg"])
    doc.start_audit(af)
    doc.mark_audit("1.jpg", "done")
    assert list_images(doc, sample_only=True) == ["1.jpg", "10.jpg"]
    assert list_images(doc, status="done") == ["1.jpg"] and list_images(doc, status="todo") == [
        "10.jpg"
    ]
    vocab = caption_vocabulary(doc.original)
    assert vocab[0] in {
        W,
        "Fences are installed in a line",
        "Two orange excavators are digging soil",
    }
    assert "" not in vocab
