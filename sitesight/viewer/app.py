"""Main window of the SiteSight review/debug tool.

Run:  python -m sitesight.viewer.app --annotations curated/annotations_curated.json \\
          --images extracted/UAVdata [--predictions model_a.json model_b.json] \\
          [--audit-sample reports/m3/audit_sample.json --audit-file audit.json --audit-mode blind]

Layout: image list (left), one or two synchronised image panels (centre), region table and
caption editor (right). Each panel shows one *source*: the original ground truth, your edit
layer, or a model's predictions. Turn on "Compare" to put two sources side by side.
Editing happens on panels whose source is ``edit``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sitesight.data.audit import AuditFile
from sitesight.data.predictions import PredictionSet
from sitesight.data.schema import Dataset
from sitesight.viewer.canvas import DisplayRegion, ImageView
from sitesight.viewer.state import (
    PRED_PREFIX,
    SOURCE_EDIT,
    SOURCE_ORIGINAL,
    Document,
    DocumentError,
    caption_vocabulary,
    list_images,
)

COLOURS = {SOURCE_ORIGINAL: "#2e9e5b", SOURCE_EDIT: "#2f6fdf"}
PRED_COLOURS = ["#d97706", "#9333ea", "#0d9488", "#dc2626"]


def _file_name(list_text: str) -> str:
    """List rows are '<marker> <file name>' for audit-sample images; file names have no spaces."""
    return list_text.rsplit(" ", 1)[-1]


def _box(b: Sequence[float]) -> tuple[float, float, float, float]:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


class Panel(QWidget):
    """A source selector above an image view."""

    def __init__(self, default_source: str) -> None:
        super().__init__()
        self.default_source = default_source
        self.source_box = QComboBox()
        self.view = ImageView()
        self.header = QLabel("")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        top = QHBoxLayout()
        top.addWidget(QLabel("Source:"))
        top.addWidget(self.source_box, 1)
        top.addWidget(self.header)
        layout.addLayout(top)
        layout.addWidget(self.view, 1)

    @property
    def source(self) -> str:
        return self.source_box.currentText()

    def set_sources(self, sources: list[str]) -> None:
        current = self.source or self.default_source
        self.source_box.blockSignals(True)
        self.source_box.clear()
        self.source_box.addItems(sources)
        if current in sources:
            self.source_box.setCurrentText(current)
        self.source_box.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self, doc: Document | None = None) -> None:
        super().__init__()
        self.setWindowTitle("SiteSight review tool")
        self.resize(1500, 900)
        self.doc: Document | None = None
        self.current: str | None = None
        self.audit_path: Path | None = None
        self._selected_ann: int | None = None

        self.panel_a, self.panel_b = Panel(SOURCE_EDIT), Panel(SOURCE_ORIGINAL)
        self.panel_a.view.editable = True
        self.panel_b.view.editable = False
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.panel_a)
        self.splitter.addWidget(self.panel_b)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.panel_b.hide()
        self.setCentralWidget(self.splitter)

        self._build_left_dock()
        self._build_right_dock()
        self._build_menus()
        docks = self.findChildren(QDockWidget)
        self.resizeDocks(docks, [210] * len(docks), Qt.Orientation.Horizontal)
        right = [d for d in docks if d.windowTitle() == "Regions"]
        if right:
            self.resizeDocks(right, [430], Qt.Orientation.Horizontal)
        self.status_label = QLabel("")
        status = self.statusBar()
        if status is not None:
            status.addPermanentWidget(self.status_label)

        for p in (self.panel_a, self.panel_b):
            p.source_box.currentTextChanged.connect(self._render)
            p.view.region_selected.connect(self._on_box_selected)
            p.view.region_drawn.connect(lambda x, y, w, h, p=p: self._on_drawn(p, x, y, w, h))
            p.view.region_changed.connect(
                lambda i, x, y, w, h, p=p: self._on_changed(p, i, x, y, w, h)
            )
        self.panel_a.view.view_changed.connect(
            lambda: self.panel_b.view.sync_from(self.panel_a.view)
        )
        self.panel_b.view.view_changed.connect(
            lambda: self.panel_a.view.sync_from(self.panel_b.view)
        )

        if doc is not None:
            self.set_document(doc)

    def _build_left_dock(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.split_filter = QComboBox()
        self.split_filter.addItems(["all splits", "train", "val", "test"])
        self.text_filter = QLineEdit()
        self.text_filter.setPlaceholderText("file name contains…")
        self.sample_only = QCheckBox("audit sample only")
        self.status_filter = QComboBox()
        self.status_filter.addItems(["any status", "todo", "done", "skipped"])
        self.image_list = QListWidget()
        for widget in (
            self.split_filter,
            self.text_filter,
            self.sample_only,
            self.status_filter,
            self.image_list,
        ):
            lay.addWidget(widget)
        self.split_filter.currentTextChanged.connect(self._refresh_list)
        self.text_filter.textChanged.connect(self._refresh_list)
        self.sample_only.toggled.connect(self._refresh_list)
        self.status_filter.currentTextChanged.connect(self._refresh_list)
        self.image_list.currentItemChanged.connect(self._on_list_selection)
        dock = QDockWidget("Images", self)
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_right_dock(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.compare = QCheckBox("Compare two sources side by side")
        self.compare.toggled.connect(self._on_compare)
        lay.addWidget(self.compare)
        lay.addWidget(QLabel("Min confidence (predictions):"))
        row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_label = QLabel("0.00")
        self.conf_slider.valueChanged.connect(self._on_conf)
        row.addWidget(self.conf_slider, 1)
        row.addWidget(self.conf_label)
        lay.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["id", "caption", "resource", "flags / score"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        lay.addWidget(self.table, 1)
        lay.addWidget(QLabel("Caption (used for new boxes; Apply edits the selected one):"))
        self.caption_box = QComboBox()
        self.caption_box.setEditable(True)
        self.caption_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        completer = QCompleter(self.caption_box.model(), self.caption_box)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.caption_box.setCompleter(completer)
        lay.addWidget(self.caption_box)
        buttons = QHBoxLayout()
        apply_btn, del_btn = QPushButton("Apply caption"), QPushButton("Delete box")
        apply_btn.clicked.connect(self.apply_caption)
        del_btn.clicked.connect(self.delete_selected)
        buttons.addWidget(apply_btn)
        buttons.addWidget(del_btn)
        lay.addLayout(buttons)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("audit notes for this image…")
        lay.addWidget(self.notes)
        dock = QDockWidget("Regions", self)
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _action(
        self, menu_text: str, menu: object, slot: object, shortcut: str | None = None
    ) -> QAction:
        act = QAction(menu_text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)  # type: ignore[arg-type]
        menu.addAction(act)  # type: ignore[attr-defined]
        return act

    def _build_menus(self) -> None:
        bar = self.menuBar()
        assert bar is not None
        f = bar.addMenu("&File")
        e = bar.addMenu("&Edit")
        v = bar.addMenu("&View")
        a = bar.addMenu("&Audit")
        assert f is not None and e is not None and v is not None and a is not None
        self._action("Open dataset…", f, self._dlg_open_dataset)
        self._action("Open predictions…", f, self._dlg_open_predictions)
        self._action("Export edits (schema JSON)…", f, self._dlg_export_edits, "Ctrl+S")
        self._action("Export audit dataset…", f, self._dlg_export_audit)
        self._action("Undo", e, self.undo, "Ctrl+Z")
        self._action("Redo", e, self.redo, "Ctrl+Y")
        self._action("Delete selected box", e, self.delete_selected, "Delete")
        self._action("Previous image", v, self.prev_image, "Ctrl+Left")
        self._action("Next image", v, self.next_image, "Ctrl+Right")
        self._action("Fit to window", v, self.fit_views, "Ctrl+0")
        self._action("Start / resume audit…", a, self._dlg_start_audit)
        self._action("Mark image done and go to next", a, self.mark_done, "Ctrl+Return")
        self._action("Skip image", a, self.mark_skipped)
        self._action("Next unfinished image", a, self.goto_next_todo, "Ctrl+J")

    def set_document(self, doc: Document) -> None:
        self.doc = doc
        self.caption_box.clear()
        self.caption_box.addItems(caption_vocabulary(doc.original))
        self.caption_box.setEditText("")
        self._refresh_sources()
        self._refresh_list()
        first = self.image_list.item(0)
        if first is not None:
            self.image_list.setCurrentItem(first)

    def _refresh_sources(self) -> None:
        if self.doc is None:
            return
        for p in (self.panel_a, self.panel_b):
            p.set_sources(self.doc.sources())

    def _refresh_list(self) -> None:
        if self.doc is None:
            return
        split = self.split_filter.currentText()
        status = self.status_filter.currentText()
        names = list_images(
            self.doc,
            split=None if split.startswith("all") else split,
            text=self.text_filter.text() or None,
            sample_only=self.sample_only.isChecked(),
            status=None if status.startswith("any") else status,
        )
        keep = self.current
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for n in names:
            marker = ""
            if self.doc.audit is not None and n in self.doc.audit.sample:
                st = self.doc.audit.images[n].status if n in self.doc.audit.images else "todo"
                marker = {"done": "✓ ", "skipped": "- ", "todo": "• "}[st]
            self.image_list.addItem(QListWidgetItem(marker + n))
            if n == keep:
                self.image_list.setCurrentRow(self.image_list.count() - 1)
        self.image_list.blockSignals(False)

    def _on_list_selection(
        self, item: QListWidgetItem | None, _prev: QListWidgetItem | None = None
    ) -> None:
        if item is not None:
            self.show_image(_file_name(item.text()))

    def image_path(self, name: str) -> Path | None:
        if self.doc is None or self.doc.images_dir is None:
            return None
        return self.doc.images_dir / name

    def show_image(self, name: str) -> None:
        if self.doc is None:
            return
        if (
            self.doc.audit is not None
            and self.current in self.doc.audit.images
            and self.current is not None
        ):
            self.doc.audit.images[self.current].notes = self.notes.text()
        self.current = name
        self._selected_ann = None
        path = self.image_path(name)
        for p in (self.panel_a, self.panel_b):
            p.view.set_image(path)
        entry = self.doc.audit.images.get(name) if self.doc.audit else None
        self.notes.setText(entry.notes if entry else "")
        self._render()

    def _display_regions(self, source: str, name: str) -> list[DisplayRegion]:
        assert self.doc is not None
        if source.startswith(PRED_PREFIX):
            model = source[len(PRED_PREFIX) :]
            names = sorted(self.doc.predictions)
            colour = (
                PRED_COLOURS[names.index(model) % len(PRED_COLOURS)]
                if model in names
                else PRED_COLOURS[0]
            )
            preds = (
                self.doc.predictions[model].by_image(self.conf_slider.value() / 100).get(name, [])
            )
            return [
                DisplayRegion(_box(p.bbox), p.caption or "", colour, None, p.score) for p in preds
            ]
        editable = source == SOURCE_EDIT
        if source == SOURCE_ORIGINAL and not self.doc.original_visible(name):
            return []
        return [
            DisplayRegion(_box(a.bbox), a.caption, COLOURS[source], a.id, None, editable)
            for a in self.doc.regions(name, source)
        ]

    def _render(self) -> None:
        if self.doc is None or self.current is None:
            return
        for p in (self.panel_a, self.panel_b):
            p.view.editable = p.source == SOURCE_EDIT
            p.view.set_regions(self._display_regions(p.source, self.current))
            p.view.select(self._selected_ann)
            img = self.doc.image(self.current)
            p.header.setText(f"{self.current}  [{img.split or 'no split'}]")
        self._fill_table()
        self._update_status()

    def _fill_table(self) -> None:
        assert self.doc is not None and self.current is not None
        src = self.panel_a.source
        rows: list[tuple[str, str, str, str]] = []
        self._row_ids: list[int | None] = []
        if src.startswith(PRED_PREFIX):
            for p in (
                self.doc.predictions[src[len(PRED_PREFIX) :]]
                .by_image(self.conf_slider.value() / 100)
                .get(self.current, [])
            ):
                rows.append(("", p.caption or "", "", f"{p.score:.2f}"))
                self._row_ids.append(None)
        elif not (src == SOURCE_ORIGINAL and not self.doc.original_visible(self.current)):
            names = {c.id: c.name for c in self.doc.edit.categories}
            for a in self.doc.regions(self.current, src):
                rows.append(
                    (str(a.id), a.caption, names.get(a.category_id or -1, ""), ",".join(a.flags))
                )
                self._row_ids.append(a.id)
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnToContents(0)
        self.table.blockSignals(False)
        if self._selected_ann in self._row_ids:
            self.table.selectRow(self._row_ids.index(self._selected_ann))

    def _update_status(self) -> None:
        assert self.doc is not None and self.current is not None
        n_o = (
            len(self.doc.regions(self.current, SOURCE_ORIGINAL))
            if self.doc.original_visible(self.current)
            else "hidden"
        )
        n_e = len(self.doc.regions(self.current, SOURCE_EDIT))
        text = f"{self.current} | original: {n_o} | edit: {n_e}"
        if self.doc.audit is not None:
            done, total = self.doc.audit_progress()
            text += f" | audit ({self.doc.audit.mode}): {done}/{total}"
        if self.doc.dirty:
            text += " | unsaved edits"
        self.status_label.setText(text)

    def _notify(self, message: str) -> None:
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message, 6000)

    def _on_compare(self, on: bool) -> None:
        self.panel_b.setVisible(on)
        if on:
            self.splitter.setSizes([1000, 1000])  # equal widths so zoom levels match
            self.panel_a.view.fit()
            self.panel_b.view.fit()

    def _on_conf(self, value: int) -> None:
        self.conf_label.setText(f"{value / 100:.2f}")
        self._render()

    def _on_box_selected(self, ann_id: int) -> None:
        self._selected_ann = ann_id
        for p in (self.panel_a, self.panel_b):
            p.view.select(ann_id)
        if ann_id in getattr(self, "_row_ids", []):
            self.table.blockSignals(True)
            self.table.selectRow(self._row_ids.index(ann_id))
            self.table.blockSignals(False)
        if self.doc is not None and self.current is not None:
            for a in self.doc.regions(self.current, SOURCE_EDIT):
                if a.id == ann_id:
                    self.caption_box.setEditText(a.caption_raw)

    def _on_table_selection(self) -> None:
        model = self.table.selectionModel()
        rows = model.selectedRows() if model is not None else []
        if rows and rows[0].row() < len(self._row_ids):
            ann = self._row_ids[rows[0].row()]
            if ann is not None:
                self._on_box_selected(ann)

    def _on_drawn(self, panel: Panel, x: float, y: float, w: float, h: float) -> None:
        if self.doc is None or self.current is None or panel.source != SOURCE_EDIT:
            return
        try:
            self._selected_ann = self.doc.add_region(
                self.current, (x, y, w, h), self.caption_box.currentText()
            )
        except DocumentError as exc:
            self._notify(str(exc))
            return
        self._render()

    def _on_changed(
        self, panel: Panel, ann_id: int, x: float, y: float, w: float, h: float
    ) -> None:
        if self.doc is None or panel.source != SOURCE_EDIT:
            return
        try:
            self.doc.set_bbox(ann_id, (x, y, w, h))
        except DocumentError as exc:
            self._notify(str(exc))
        self._render()

    def apply_caption(self) -> None:
        if self.doc is None or self._selected_ann is None:
            self._notify("select a box in an 'edit' panel first")
            return
        try:
            self.doc.set_caption(self._selected_ann, self.caption_box.currentText())
        except DocumentError as exc:
            self._notify(str(exc))
        self._render()

    def delete_selected(self) -> None:
        if self.doc is None or self._selected_ann is None:
            return
        try:
            self.doc.delete_region(self._selected_ann)
        except DocumentError as exc:
            self._notify(str(exc))
            return
        self._selected_ann = None
        self._render()

    def undo(self) -> None:
        if self.doc is not None and self.doc.undo():
            self._selected_ann = None
            self._render()

    def redo(self) -> None:
        if self.doc is not None and self.doc.redo():
            self._selected_ann = None
            self._render()

    def fit_views(self) -> None:
        self.panel_a.view.fit()
        self.panel_b.view.fit()

    def _step(self, delta: int) -> None:
        row = self.image_list.currentRow() + delta
        if 0 <= row < self.image_list.count():
            self.image_list.setCurrentRow(row)

    def next_image(self) -> None:
        self._step(1)

    def prev_image(self) -> None:
        self._step(-1)

    def goto(self, name: str) -> None:
        for r in range(self.image_list.count()):
            item = self.image_list.item(r)
            if item is not None and _file_name(item.text()) == name:
                self.image_list.setCurrentRow(r)
                return
        self.show_image(name)

    def start_audit(self, sample_json: Path, audit_path: Path, mode: str) -> None:
        """Load the sample and resume ``audit_path`` if it exists, else create it."""
        if self.doc is None:
            return
        if audit_path.exists():
            audit = AuditFile.load(audit_path)
        else:
            rows = json.loads(sample_json.read_text(encoding="utf-8"))["images"]
            audit = AuditFile(mode=mode, sample=[r["file_name"] for r in rows])
        self.doc.start_audit(audit)
        self.audit_path = audit_path
        audit.save(audit_path)
        self._refresh_list()
        todo = self.doc.next_todo()
        if todo is not None:
            self.goto(todo)
        else:
            self._render()

    def _save_audit(self) -> None:
        if self.doc is not None and self.doc.audit is not None and self.audit_path is not None:
            if self.current is not None and self.current in self.doc.audit.images:
                self.doc.audit.images[self.current].notes = self.notes.text()
            self.doc.audit.save(self.audit_path)

    def mark_done(self) -> None:
        self._set_status("done")

    def mark_skipped(self) -> None:
        self._set_status("skipped")

    def _set_status(self, status: str) -> None:
        if self.doc is None or self.doc.audit is None or self.current is None:
            self._notify("no audit loaded")
            return
        try:
            self.doc.mark_audit(self.current, status, self.notes.text())
        except DocumentError as exc:
            self._notify(str(exc))
            return
        self._save_audit()
        finished = self.current
        self._refresh_list()
        nxt = self.doc.next_todo(after=finished)
        if nxt is not None:
            self.goto(nxt)
        else:
            self._render()
            self._notify("audit complete: every sampled image is done or skipped")

    def goto_next_todo(self) -> None:
        if self.doc is not None and self.doc.audit is not None:
            nxt = self.doc.next_todo(after=self.current)
            if nxt is not None:
                self.goto(nxt)

    # Dialogs: thin wrappers; the logic lives in the methods above and in Document.

    def open_dataset(self, annotations: Path, images_dir: Path) -> None:
        self.set_document(Document(Dataset.load(annotations), images_dir))

    def open_predictions(self, path: Path) -> None:
        if self.doc is None:
            return
        self.doc.add_predictions(PredictionSet.load(path))
        self._refresh_sources()
        self._render()

    def _dlg_open_dataset(self) -> None:
        ann, _ = QFileDialog.getOpenFileName(self, "Annotations JSON", "", "JSON (*.json)")
        folder = QFileDialog.getExistingDirectory(self, "Images folder") if ann else ""
        if ann and folder:
            self.open_dataset(Path(ann), Path(folder))

    def _dlg_open_predictions(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Predictions JSON", "", "JSON (*.json)")
        if path:
            try:
                self.open_predictions(Path(path))
            except (DocumentError, KeyError, ValueError) as exc:
                QMessageBox.warning(self, "Predictions", str(exc))

    def _dlg_export_edits(self) -> None:
        if self.doc is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export edits", "annotations_edited.json", "JSON (*.json)"
        )
        if path:
            problems = self.doc.export_edit(Path(path))
            if problems:
                QMessageBox.warning(self, "Not saved", "\n".join(problems[:10]))
            else:
                self._notify(f"saved {path}")
                self._update_status()

    def _dlg_export_audit(self) -> None:
        if self.doc is None or self.doc.audit is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audit dataset", "audit_annotations.json", "JSON (*.json)"
        )
        if path:
            problems = self.doc.export_audit_dataset(Path(path))
            if problems:
                QMessageBox.warning(self, "Not saved", "\n".join(problems[:10]))

    def _dlg_start_audit(self) -> None:
        sample, _ = QFileDialog.getOpenFileName(self, "Audit sample JSON", "", "JSON (*.json)")
        if not sample:
            return
        audit, _ = QFileDialog.getSaveFileName(
            self, "Audit file (resumed if it exists)", "audit.json", "JSON (*.json)"
        )
        mode, ok = QInputDialog.getItem(self, "Audit mode", "Mode:", ["blind", "review"], 0, False)
        if audit and ok:
            self.start_audit(Path(sample), Path(audit), mode)

    def closeEvent(self, event: object) -> None:
        self._save_audit()
        super().closeEvent(event)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SiteSight review/debug tool")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--predictions", type=Path, nargs="*", default=[])
    p.add_argument("--audit-sample", type=Path)
    p.add_argument("--audit-file", type=Path)
    p.add_argument("--audit-mode", choices=["blind", "review"], default="blind")
    args = p.parse_args(argv)

    existing: object = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv[:1])
    win = MainWindow()
    win.open_dataset(args.annotations, args.images)
    for pred in args.predictions:
        win.open_predictions(pred)
    if args.audit_file is not None:
        if args.audit_sample is None and not args.audit_file.exists():
            p.error("--audit-sample is required to create a new --audit-file")
        win.start_audit(args.audit_sample or args.audit_file, args.audit_file, args.audit_mode)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
