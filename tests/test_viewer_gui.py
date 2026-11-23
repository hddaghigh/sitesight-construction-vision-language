"""GUI tests: the real PyQt6 window on Qt's offscreen platform, driven with real mouse events.

Opt-in (needs PyQt6):  pytest -m viewer
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from sitesight.data.audit import AuditFile
from sitesight.data.caption_fields import apply_caption_fields
from sitesight.data.predictions import Prediction, PredictionSet
from sitesight.data.validate import validate_dataset
from sitesight.data.via_parse import parse_via
from sitesight.viewer.app import MainWindow
from sitesight.viewer.canvas import ImageView
from sitesight.viewer.state import SOURCE_EDIT, SOURCE_ORIGINAL, Document

pytestmark = pytest.mark.viewer
W = "A worker is located on soil"


@pytest.fixture(scope="module")
def qapp() -> Iterator[QApplication]:
    existing: object = QApplication.instance()
    yield existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def win(qapp: QApplication, via_fixture: tuple[Path, Path]) -> Iterator[MainWindow]:
    ds, _ = parse_via(*via_fixture)
    apply_caption_fields(ds)
    for i, img in enumerate(ds.images):
        img.split = ("train", "val", "test")[i]
    w = MainWindow(Document(ds, via_fixture[1]))
    w.resize(1400, 800)
    w.show()
    qapp.processEvents()
    w.goto("1.jpg")
    qapp.processEvents()
    yield w
    w.close()


def vp(view: ImageView, x: float, y: float) -> QPoint:
    """Viewport pixel of an image coordinate."""
    return view.mapFromScene(QPointF(x, y))


def drag(
    view: ImageView, a: tuple[float, float], b: tuple[float, float], app: QApplication
) -> None:
    pa, pb = vp(view, *a), vp(view, *b)
    port = view.viewport()
    assert port is not None
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    # The PyQt6 stubs declare these static QTest functions as if they needed a QTest instance.
    QTest.mousePress(port, left, modifier=none, pos=pa)  # type: ignore[call-overload]
    app.processEvents()
    QTest.mouseMove(port, pos=pb)  # type: ignore[call-overload]
    app.processEvents()
    QTest.mouseRelease(port, left, modifier=none, pos=pb)  # type: ignore[call-overload]
    app.processEvents()


def test_window_lists_images_and_shows_boxes(win: MainWindow) -> None:
    assert win.image_list.count() == 3
    assert win.panel_a.source == SOURCE_EDIT and win.panel_b.isHidden()
    assert win.table.rowCount() == 4 and len(win.panel_a.view._boxes) == 4
    win.split_filter.setCurrentText("val")
    only = win.image_list.item(0)
    assert win.image_list.count() == 1 and only is not None and only.text() == "2.jpg"


def test_drawing_a_box_with_the_mouse_adds_a_region(win: MainWindow, qapp: QApplication) -> None:
    assert win.doc is not None
    n = len(win.doc.edit.annotations)
    win.caption_box.setEditText(W)
    drag(win.panel_a.view, (60, 30), (90, 50), qapp)  # empty area, bottom right of the 100x60 image
    assert len(win.doc.edit.annotations) == n + 1
    new = win.doc.edit.annotations[-1]
    assert abs(new.bbox[0] - 60) <= 1 and abs(new.bbox[1] - 30) <= 1
    assert abs(new.bbox[2] - 30) <= 2 and abs(new.bbox[3] - 20) <= 2
    assert new.caption == W and new.fields is not None
    assert validate_dataset(win.doc.edit) == []
    assert win.table.rowCount() == 5 and "unsaved" in win.status_label.text()
    win.undo()
    assert len(win.doc.edit.annotations) == n and win.table.rowCount() == 4


def test_tiny_drag_is_a_click_not_a_box(win: MainWindow, qapp: QApplication) -> None:
    assert win.doc is not None
    n = len(win.doc.edit.annotations)
    drag(win.panel_a.view, (70, 40), (71, 41), qapp)
    assert len(win.doc.edit.annotations) == n


def test_moving_and_resizing_a_box(win: MainWindow, qapp: QApplication) -> None:
    assert win.doc is not None
    view = win.panel_a.view
    # The fixture's regions 0 and 2 are exact duplicates (the mouse would grab the top copy), so
    # use the unique 10x10 box at [30, 30, 10, 10].
    target = next(a for a in win.doc.regions("1.jpg", SOURCE_EDIT) if a.bbox == [30, 30, 10, 10])
    drag(view, (35, 35), (45, 40), qapp)  # grab the middle, move by (+10, +5)
    moved = next(a for a in win.doc.edit.annotations if a.id == target.id)
    assert all(abs(a - b) <= 1 for a, b in zip(moved.bbox, [40, 35, 10, 10], strict=True))
    # Resize from the bottom-right corner (50, 45) by (+10, +5): top-left stays, size grows.
    x, y, w, h = moved.bbox
    drag(view, (x + w, y + h), (x + w + 10, y + h + 5), qapp)
    resized = next(a for a in win.doc.edit.annotations if a.id == target.id)
    assert resized.bbox[0] == moved.bbox[0] and resized.bbox[1] == moved.bbox[1]
    assert all(abs(a - b) <= 2 for a, b in zip(resized.bbox[2:], [w + 10, h + 5], strict=True))
    assert validate_dataset(win.doc.edit) == [] and resized.source["edited"] == "box"


def test_boxes_cannot_leave_the_image(win: MainWindow, qapp: QApplication) -> None:
    assert win.doc is not None
    view = win.panel_a.view
    first = next(a for a in win.doc.regions("1.jpg", SOURCE_EDIT) if a.bbox == [30, 30, 10, 10])
    drag(view, (35, 35), (200, 200), qapp)  # drag far outside the 100x60 image
    box = next(a for a in win.doc.edit.annotations if a.id == first.id).bbox
    assert box != [30, 30, 10, 10]  # it did move
    assert box == [90, 50, 10, 10]  # clamped to the bottom-right corner of the 100x60 image
    assert box[0] + box[2] <= 100 and box[1] + box[3] <= 60 and box[0] >= 0 and box[1] >= 0
    assert validate_dataset(win.doc.edit) == []


def test_apply_caption_delete_and_table_selection(win: MainWindow) -> None:
    assert win.doc is not None
    win.table.selectRow(0)
    assert win._selected_ann == win.doc.regions("1.jpg", SOURCE_EDIT)[0].id
    assert win.caption_box.currentText() == W  # selecting a region loads its caption
    win.caption_box.setEditText("Two workers are located on soil")
    win.apply_caption()
    assert win.doc.regions("1.jpg", SOURCE_EDIT)[0].fields["count"] == 2  # type: ignore[index]
    win.delete_selected()
    assert win.table.rowCount() == 3 and win._selected_ann is None


def test_compare_mode_syncs_zoom_between_panels(win: MainWindow, qapp: QApplication) -> None:
    win.compare.setChecked(True)
    qapp.processEvents()
    assert not win.panel_b.isHidden() and win.panel_b.source == SOURCE_ORIGINAL
    a, b = win.panel_a.view, win.panel_b.view
    assert a.transform().m11() == pytest.approx(b.transform().m11(), rel=0.05)  # equal fit zoom
    before = b.transform().m11()
    ev = QWheelEvent(
        QPointF(50, 50),
        QPointF(50, 50),
        QPoint(0, 0),
        QPoint(0, 240),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    a.wheelEvent(ev)
    assert a.transform().m11() > before * 1.2
    assert b.transform().m11() == pytest.approx(a.transform().m11())  # zoom followed
    win.fit_views()
    assert a.transform().m11() == pytest.approx(before, rel=0.05)


def test_predictions_are_shown_and_filtered_by_confidence(
    win: MainWindow, qapp: QApplication, tmp_path: Path
) -> None:
    pset = PredictionSet(
        "model-a",
        [
            Prediction("1.jpg", [10, 10, 20, 20], 0.9, W),
            Prediction("1.jpg", [50, 30, 20, 20], 0.3, "A blue tarpaulin is located on soil"),
        ],
        {"notes": "test"},
    )
    path = tmp_path / "p.json"
    pset.save(path)
    win.open_predictions(path)
    win.panel_a.source_box.setCurrentText("pred:model-a")
    qapp.processEvents()
    assert len(win.panel_a.view._boxes) == 2 and win.table.rowCount() == 2
    win.conf_slider.setValue(50)  # keep only score >= 0.50
    assert len(win.panel_a.view._boxes) == 1 and win.conf_label.text() == "0.50"
    win.panel_a.view.editable = False
    # predictions are read-only: a drag in a prediction panel must not create regions
    assert win.doc is not None
    n = len(win.doc.edit.annotations)
    drag(win.panel_a.view, (60, 35), (90, 55), qapp)
    assert len(win.doc.edit.annotations) == n


def test_command_line_entry_point_with_audit(
    qapp: QApplication,
    via_fixture: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sitesight.viewer import app as app_module

    via, images = via_fixture
    ds, _ = parse_via(via, images)
    apply_caption_fields(ds)
    ann = tmp_path / "ann.json"
    ds.save(ann)
    pred = tmp_path / "pred.json"
    PredictionSet("m1", [Prediction("1.jpg", [10, 10, 20, 20], 0.7)]).save(pred)
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"images": [{"file_name": "10.jpg"}, {"file_name": "1.jpg"}]}))
    audit = tmp_path / "audit.json"
    monkeypatch.setattr(QApplication, "exec", staticmethod(lambda: 0))  # do not block
    argv = ["--annotations", str(ann), "--images", str(images), "--predictions", str(pred)]
    argv += ["--audit-sample", str(sample), "--audit-file", str(audit), "--audit-mode", "review"]
    assert app_module.main(argv) == 0
    saved = AuditFile.load(audit)
    assert saved.mode == "review" and saved.sample == ["10.jpg", "1.jpg"]
    with pytest.raises(SystemExit):  # a NEW audit file needs --audit-sample
        app_module.main(
            [
                "--annotations",
                str(ann),
                "--images",
                str(images),
                "--audit-file",
                str(tmp_path / "new.json"),
            ]
        )


def test_blind_audit_flow_hides_original_then_saves(
    win: MainWindow, qapp: QApplication, tmp_path: Path
) -> None:
    assert win.doc is not None
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"images": [{"file_name": "1.jpg"}, {"file_name": "10.jpg"}]}))
    audit_path = tmp_path / "audit.json"
    win.start_audit(sample, audit_path, "blind")
    qapp.processEvents()
    assert win.current == "1.jpg" and audit_path.exists()
    win.panel_a.source_box.setCurrentText(SOURCE_ORIGINAL)
    assert len(win.panel_a.view._boxes) == 0  # blind: the original is hidden
    win.panel_a.source_box.setCurrentText(SOURCE_EDIT)
    assert win.table.rowCount() == 0
    win.caption_box.setEditText(W)
    drag(win.panel_a.view, (10, 10), (40, 40), qapp)
    win.notes.setText("clear image")
    win.mark_done()  # saves, reveals the original, moves to the next unfinished image
    qapp.processEvents()
    saved = AuditFile.load(audit_path)
    assert saved.images["1.jpg"].status == "done" and saved.images["1.jpg"].notes == "clear image"
    assert len(saved.images["1.jpg"].annotations) == 1
    assert win.current == "10.jpg" and "audit (blind): 1/2" in win.status_label.text()
    win.goto("1.jpg")
    win.panel_a.source_box.setCurrentText(SOURCE_ORIGINAL)
    assert len(win.panel_a.view._boxes) == 4  # revealed after the image is done
    win.mark_skipped()
    assert AuditFile.load(audit_path).images["1.jpg"].status == "skipped"
