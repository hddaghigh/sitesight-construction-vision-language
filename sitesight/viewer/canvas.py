"""Image canvas: zoomable view with editable boxes, a drawing gesture and view syncing.

Coordinates: the scene IS the image (1 scene unit = 1 image pixel), so a box item's rectangle
is exactly the ``[x, y, w, h]`` stored in the dataset and no conversion can drift.

Gestures: wheel = zoom under the cursor; middle-button drag = pan; left-drag on empty image =
draw a new box (only when the view is editable); left-drag inside a box = move it; left-drag on
a box edge/corner = resize it; click = select.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)

MIN_DRAW = 4.0  # scene units: smaller drags are treated as clicks
EDGE_TOL = 7.0  # device pixels: grab distance for resize handles
MARGIN = 40.0  # scene margin around the image so labels above the top edge stay visible


@dataclass
class DisplayRegion:
    """One box to draw. ``ann_id`` is None for read-only predictions."""

    bbox: tuple[float, float, float, float]
    text: str
    colour: str
    ann_id: int | None = None
    score: float | None = None
    editable: bool = False


class LabelItem(QGraphicsItem):
    """Caption chip drawn above a box, constant size on screen regardless of zoom."""

    def __init__(self, text: str, colour: QColor, parent: QGraphicsItem) -> None:
        super().__init__(parent=parent)  # type: ignore[arg-type]  # stub bug: expects LabelItem
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._text, self._colour = text, colour
        self._font = QFont()
        self._font.setPointSize(9)
        fm = QFontMetrics(self._font)
        self._w, self._h = fm.horizontalAdvance(text) + 8, fm.height() + 2

    def boundingRect(self) -> QRectF:
        return QRectF(0, -self._h, self._w, self._h)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ) -> None:
        if painter is None:
            return
        bg = QColor(self._colour)
        bg.setAlpha(215)
        painter.fillRect(self.boundingRect(), bg)
        painter.setFont(self._font)
        painter.setPen(QColor("white"))
        painter.drawText(
            self.boundingRect().adjusted(4, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, self._text
        )


class BoxItem(QGraphicsRectItem):
    """A box that can be selected and (if editable) moved/resized with the mouse."""

    def __init__(self, region: DisplayRegion, view: ImageView) -> None:
        x, y, w, h = region.bbox
        super().__init__(QRectF(x, y, w, h))
        self.region, self._view = region, view
        self._colour = QColor(region.colour)
        pen = QPen(self._colour, 2)
        pen.setCosmetic(True)  # stays 2 px wide at any zoom
        self.setPen(pen)
        self.setAcceptHoverEvents(region.editable)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, region.ann_id is not None)
        self.setZValue(1 if region.editable else 0)
        text = region.text if region.score is None else f"{region.text}  {region.score:.2f}"
        self._label = LabelItem(text or "(no caption)", self._colour, self)
        self._label.setPos(x, y)
        self._mode: str | None = None
        self._press_scene = QPointF()
        self._press_rect = QRectF()

    def set_selected_look(self, selected: bool) -> None:
        pen = QPen(self._colour, 4 if selected else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(self._colour)
        fill.setAlpha(45 if selected else 0)
        self.setBrush(QBrush(fill))

    def _edges(self, p: QPointF) -> str:
        """Which edges of the rect the point is near ('', 'l', 'tr', ...). Tolerance in pixels."""
        tol = EDGE_TOL / max(self._view.transform().m11(), 1e-6)
        r = self.rect()
        e = ""
        if abs(p.y() - r.top()) <= tol:
            e += "t"
        elif abs(p.y() - r.bottom()) <= tol:
            e += "b"
        if abs(p.x() - r.left()) <= tol:
            e += "l"
        elif abs(p.x() - r.right()) <= tol:
            e += "r"
        return e

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent | None) -> None:
        if event is None:
            return
        e = self._edges(event.pos())
        cursor = {
            "": Qt.CursorShape.SizeAllCursor,
            "l": Qt.CursorShape.SizeHorCursor,
            "r": Qt.CursorShape.SizeHorCursor,
            "t": Qt.CursorShape.SizeVerCursor,
            "b": Qt.CursorShape.SizeVerCursor,
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
        }[e]
        self.setCursor(cursor)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent | None) -> None:
        if event is None:
            return
        if self.region.ann_id is not None:
            self._view.select(self.region.ann_id, emit=True)
        if not self.region.editable:
            event.accept()
            return
        self._mode = self._edges(event.pos()) or "move"
        self._press_scene = event.scenePos()
        self._press_rect = QRectF(self.rect())
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent | None) -> None:
        if event is None or self._mode is None:
            return
        d = event.scenePos() - self._press_scene
        r = QRectF(self._press_rect)
        bounds = self._view.image_rect()
        if self._mode == "move":
            r.translate(d)
            r.moveLeft(min(max(r.left(), bounds.left()), bounds.right() - r.width()))
            r.moveTop(min(max(r.top(), bounds.top()), bounds.bottom() - r.height()))
        else:
            if "l" in self._mode:
                r.setLeft(min(max(r.left() + d.x(), bounds.left()), r.right() - MIN_DRAW))
            if "r" in self._mode:
                r.setRight(max(min(r.right() + d.x(), bounds.right()), r.left() + MIN_DRAW))
            if "t" in self._mode:
                r.setTop(min(max(r.top() + d.y(), bounds.top()), r.bottom() - MIN_DRAW))
            if "b" in self._mode:
                r.setBottom(max(min(r.bottom() + d.y(), bounds.bottom()), r.top() + MIN_DRAW))
        self.setRect(r)
        self._label.setPos(r.topLeft())

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent | None) -> None:
        if event is None:
            return
        mode, self._mode = self._mode, None
        if mode is not None and self.region.ann_id is not None and self.rect() != self._press_rect:
            r = self.rect()
            self._view.region_changed.emit(self.region.ann_id, r.x(), r.y(), r.width(), r.height())


class ImageView(QGraphicsView):
    # The PyQt6 stubs wrongly demand an `arguments=` keyword for typed signals.
    region_drawn = pyqtSignal(float, float, float, float)  # type: ignore[call-arg]
    region_changed = pyqtSignal(int, float, float, float, float)  # type: ignore[call-arg]
    region_selected = pyqtSignal(int)  # type: ignore[call-arg]
    view_changed = pyqtSignal()  # type: ignore[call-arg]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.editable = False
        self._pixmap: QGraphicsPixmapItem | None = None
        self._image_rect = QRectF()
        self._boxes: dict[int, BoxItem] = {}
        self._selected: int | None = None
        self._rubber: QGraphicsRectItem | None = None
        self._draw_start: QPointF | None = None
        self._pan_last: QPointF | None = None
        self._syncing = False
        self._auto_fit = True
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            if bar is not None:
                bar.valueChanged.connect(self._emit_view_changed)

    def image_rect(self) -> QRectF:
        return QRectF(self._image_rect)

    def set_image(self, path: Path | None, keep_view: bool = False) -> bool:
        self._scene.clear()
        self._boxes.clear()
        self._pixmap, self._selected = None, None
        if path is None:
            return False
        pm = QPixmap(str(path))
        if pm.isNull():
            return False
        self._pixmap = self._scene.addPixmap(pm)
        self._image_rect = QRectF(0, 0, pm.width(), pm.height())
        self._scene.setSceneRect(self._image_rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN))
        if not keep_view:
            self.fit()
        return True

    def set_regions(self, regions: list[DisplayRegion]) -> None:
        for item in [i for i in self._scene.items() if isinstance(i, BoxItem)]:
            self._scene.removeItem(item)
        self._boxes.clear()
        for k, r in enumerate(regions):
            item = BoxItem(r, self)
            self._scene.addItem(item)
            self._boxes[r.ann_id if r.ann_id is not None else -(k + 1)] = item
        if self._selected is not None:
            self.select(self._selected, emit=False)

    def select(self, ann_id: int | None, emit: bool = False) -> None:
        if self._selected is not None and self._selected in self._boxes:
            self._boxes[self._selected].set_selected_look(False)
        self._selected = ann_id
        if ann_id in self._boxes:
            self._boxes[ann_id].set_selected_look(True)
        if emit and ann_id is not None:
            self.region_selected.emit(ann_id)

    def fit(self) -> None:
        """Fit the whole image in the view and re-enable automatic refitting on resize."""
        self._auto_fit = True
        self._fit_quietly()

    def _fit_quietly(self) -> None:
        """Fit without telling the peer view: each view refits itself, so an automatic refit
        never overwrites the peer's zoom (only user zoom/pan is synchronised)."""
        if not self._image_rect.isValid():
            return
        self._syncing = True
        try:
            self.fitInView(self._image_rect, Qt.AspectRatioMode.KeepAspectRatio)
        finally:
            self._syncing = False

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self._fit_quietly()

    def _emit_view_changed(self) -> None:
        if not self._syncing:
            self.view_changed.emit()

    def sync_from(self, other: ImageView) -> None:
        """Copy zoom and scroll position from another view (side-by-side comparison)."""
        self._syncing = True
        self._auto_fit = other._auto_fit
        try:
            self.setTransform(other.transform())
            for mine, theirs in (
                (self.horizontalScrollBar(), other.horizontalScrollBar()),
                (self.verticalScrollBar(), other.verticalScrollBar()),
            ):
                if mine is not None and theirs is not None:
                    mine.setValue(theirs.value())
        finally:
            self._syncing = False

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        self._auto_fit = False  # the user chose a zoom level: stop refitting on resize
        self.scale(factor, factor)
        self.view_changed.emit()

    def _over_box(self, pos: QPointF) -> bool:
        return any(isinstance(i, BoxItem) for i in self.items(pos.toPoint()))

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.editable
            and not self._over_box(event.position())
        ):
            p = self.mapToScene(event.position().toPoint())
            if self._image_rect.contains(p):
                self._draw_start = p
                self._rubber = QGraphicsRectItem(QRectF(p, p))
                pen = QPen(QColor("#ffd400"), 2, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                self._rubber.setPen(pen)
                self._rubber.setZValue(5)
                self._scene.addItem(self._rubber)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self._pan_last is not None:
            d = event.position() - self._pan_last
            self._pan_last = event.position()
            h, v = self.horizontalScrollBar(), self.verticalScrollBar()
            if h is not None and v is not None:
                h.setValue(h.value() - round(d.x()))
                v.setValue(v.value() - round(d.y()))
            return
        if self._rubber is not None and self._draw_start is not None:
            p = self.mapToScene(event.position().toPoint())
            self._rubber.setRect(
                QRectF(self._draw_start, p).normalized().intersected(self._image_rect)
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self.unsetCursor()
            return
        if self._rubber is not None:
            r = self._rubber.rect()
            self._scene.removeItem(self._rubber)
            self._rubber, self._draw_start = None, None
            if r.width() >= MIN_DRAW and r.height() >= MIN_DRAW:
                self.region_drawn.emit(r.x(), r.y(), r.width(), r.height())
            return
        super().mouseReleaseEvent(event)
