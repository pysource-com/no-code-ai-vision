"""VisoNode desktop GUI (PySide6).

A native replacement for the old browser editor. The node graph, inspector,
events, terminal and live preview all run inside one Qt window, while the heavy
lifting (capture + YOLO26 / SAM 3 inference + drawing) is handled by the shared
engine in ``core``/``main``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from PySide6.QtCore import (
    QMimeData,
    QObject,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDrag,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import core
import main as engine

ROOT = Path(__file__).resolve().parent
WORKFLOW_FILE = ROOT / "workflow.json"
INPUT_NODE_ID = core.INPUT_NODE_ID
NODE_MIME_TYPE = "application/x-visonode-node-type"

NODE_W = 222
NODE_H = 86
PORT_Y = 43
PORT_R = 7

NODE_ICONS = {
    "input": "🎬",
    "detector": "🔍",
    "segmenter": "✂️",
    "classifier": "🏷️",
    "filter": "⚗️",
    "preview": "🖥️",
    "alert": "🔔",
}

DETECTION_MODEL_OPTIONS = [
    ("yolo26n.pt", "YOLO26 nano"),
    ("yolo26s.pt", "YOLO26 small"),
    ("yolo26m.pt", "YOLO26 medium"),
    ("yolo26l.pt", "YOLO26 large"),
    ("yolo26x.pt", "YOLO26 extra large"),
]
SEGMENTATION_MODEL_OPTIONS = [
    ("yolo26n-seg.pt", "YOLO26 nano segmentation"),
    ("yolo26s-seg.pt", "YOLO26 small segmentation"),
    ("yolo26m-seg.pt", "YOLO26 medium segmentation"),
    ("yolo26l-seg.pt", "YOLO26 large segmentation"),
    ("yolo26x-seg.pt", "YOLO26 extra large segmentation"),
]
RFDETR_DETECTION_MODEL_OPTIONS = [
    ("rfdetr-nano", "RF-DETR nano"),
    ("rfdetr-small", "RF-DETR small"),
    ("rfdetr-medium", "RF-DETR medium"),
    ("rfdetr-large", "RF-DETR large"),
]
RFDETR_SEGMENTATION_MODEL_OPTIONS = [
    ("rfdetr-seg-nano", "RF-DETR-Seg nano"),
    ("rfdetr-seg-small", "RF-DETR-Seg small"),
    ("rfdetr-seg-medium", "RF-DETR-Seg medium"),
    ("rfdetr-seg-large", "RF-DETR-Seg large"),
]
CLASSIFICATION_MODEL_OPTIONS = [
    ("yolo26n-cls.pt", "YOLO26 nano classification"),
    ("yolo26s-cls.pt", "YOLO26 small classification"),
    ("yolo26m-cls.pt", "YOLO26 medium classification"),
    ("yolo26l-cls.pt", "YOLO26 large classification"),
    ("yolo26x-cls.pt", "YOLO26 extra large classification"),
]

NODE_BLUEPRINTS = {
    "input": {
        "title": "Input",
        "subtitle": "Camera or file source",
        "x": 40,
        "y": 60,
        "config": {"sourceType": "camera", "source": 0, "width": 1280, "height": 720, "filePath": "", "loop": True},
    },
    "detector": {
        "title": "Object Detection",
        "subtitle": "YOLO26 or RF-DETR",
        "x": 330,
        "y": 60,
        "config": {"engine": "yolo26", "threshold": 0.55, "yoloModel": "yolo26n.pt", "rfdetrModel": "rfdetr-nano", "rfdetrCheckpoint": "", "device": "auto", "imgsz": 640, "end2end": True},
    },
    "segmenter": {
        "title": "Object Segmentation",
        "subtitle": "YOLO26, RF-DETR, or SAM 3 masks",
        "x": 620,
        "y": 60,
        "config": {"engine": "yolo26", "threshold": 0.55, "yoloModel": "yolo26n-seg.pt", "rfdetrModel": "rfdetr-seg-nano", "rfdetrCheckpoint": "", "samCheckpoint": "", "concepts": "person", "device": "auto", "imgsz": 640, "end2end": True},
    },
    "classifier": {
        "title": "Object Classification",
        "subtitle": "YOLO26 image label",
        "x": 620,
        "y": 230,
        "config": {"engine": "yolo26", "threshold": 0.25, "yoloModel": "yolo26n-cls.pt", "device": "auto", "imgsz": 224},
    },
    "filter": {
        "title": "Class Filter",
        "subtitle": "Only pass selected labels",
        "x": 330,
        "y": 230,
        "config": {"classes": "person, car, dog, cat, bottle", "minCount": 1},
    },
    "preview": {
        "title": "Live Preview",
        "subtitle": "Native display window",
        "x": 40,
        "y": 400,
        "config": {"showBoxes": True, "showLabels": True, "showMasks": True, "maskOpacity": 0.35, "useFilter": False},
    },
    "alert": {
        "title": "Alert Output",
        "subtitle": "Event log",
        "x": 330,
        "y": 400,
        "config": {"cooldownSeconds": 5, "message": "Detected target object"},
    },
}

# ---- theme ----------------------------------------------------------------
C_BG = "#0b1020"
C_CANVAS = "#0f172a"
C_PANEL = "#111827"
C_CARD = "#182235"
C_CARD_HOVER = "#251c23"
C_BORDER = "#263244"
C_BORDER_STRONG = "#4a5872"
C_TEXT = "#e5e7eb"
C_MUTED = "#94a3b8"
C_ACCENT = "#ff6d5a"
C_ACCENT2 = "#8da2fb"
C_DANGER = "#f97066"
C_WARN = "#f79009"
C_SUCCESS = "#12b76a"

STATUS_COLORS = {
    "idle": "#64748b",
    "starting": C_ACCENT2,
    "running": C_SUCCESS,
    "error": C_DANGER,
    "off": "#475569",
    "unlinked": C_WARN,
    "stopped": "#64748b",
}

NODE_TYPE_COLORS = {
    "input": "#ff6d5a",
    "detector": "#2e90fa",
    "segmenter": "#7a5af8",
    "classifier": "#12b76a",
    "filter": "#f79009",
    "preview": "#06aed4",
    "alert": "#f04438",
}

STYLESHEET = f"""
* {{ font-family: "Segoe UI Variable", "Segoe UI", sans-serif; }}
QMainWindow, QWidget#root {{ background: {C_BG}; }}
QWidget {{ color: {C_TEXT}; font-size: 13px; }}
QLabel#brandMark {{
    background: transparent; color: {C_ACCENT}; font-weight: 900;
    border: 2px solid {C_ACCENT}; border-radius: 10px; padding: 3px 7px; font-size: 14px;
}}
QLabel#brandTitle {{ color: {C_TEXT}; font-size: 18px; font-weight: 750; }}
QLabel#versionBadge {{
    color: {C_MUTED}; background: {C_CARD}; border: 1px solid {C_BORDER};
    border-radius: 8px; padding: 2px 8px; font-size: 11px;
}}
QLabel#panelHeading {{ color: {C_MUTED}; font-size: 11px; font-weight: 800; letter-spacing: 1px; }}
QFrame#panel {{ background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 16px; }}
QFrame#topbar {{ background: {C_PANEL}; border-bottom: 1px solid {C_BORDER}; }}
QFrame#hsep {{ background: {C_BORDER}; max-height: 1px; min-height: 1px; }}
QSplitter {{ background: {C_BG}; }}
QSplitter::handle {{ background: {C_BG}; }}

QPushButton {{
    background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 9px;
    padding: 8px 14px; color: {C_TEXT}; font-weight: 600;
}}
QPushButton:hover {{ background: {C_CARD}; border-color: {C_BORDER_STRONG}; }}
QPushButton#primary {{ background: {C_ACCENT}; color: white; border: none; font-weight: 800; }}
QPushButton#primary:hover {{ background: #ff826f; }}
QPushButton#primary:disabled {{ background: #5a2b2c; color: #9f6a66; }}
QPushButton#danger {{ background: {C_PANEL}; border: 1px solid {C_BORDER}; }}
QPushButton#danger:hover {{ border-color: {C_DANGER}; color: {C_DANGER}; }}
QPushButton#palette {{
    text-align: left; padding: 10px 12px; font-weight: 650;
    background: transparent; border-color: transparent; border-radius: 10px;
}}
QPushButton#palette:hover {{ background: {C_CARD_HOVER}; border-color: #6b363c; color: #ffad9f; }}

QComboBox, QLineEdit, QSpinBox, QTextEdit, QPlainTextEdit {{
    background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 9px;
    padding: 6px 9px; selection-background-color: {C_ACCENT2};
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QTextEdit:focus {{ border-color: {C_ACCENT2}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {C_PANEL}; border: 1px solid {C_BORDER}; selection-background-color: #24304a;
    color: {C_TEXT}; outline: none;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 6px; border: 1px solid {C_BORDER}; background: {C_PANEL}; }}
QCheckBox::indicator:checked {{ background: {C_ACCENT}; border-color: {C_ACCENT}; }}

QSlider::groove:horizontal {{ height: 5px; background: #253145; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {C_PANEL}; border: 1px solid {C_ACCENT}; width: 15px; height: 15px; margin: -6px 0; border-radius: 8px; }}

QListWidget {{ background: transparent; border: none; }}
QListWidget::item {{
    background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 10px;
    padding: 7px 10px; margin-bottom: 6px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #344258; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QGraphicsView {{ border: 1px solid {C_BORDER}; border-radius: 16px; background: {C_CANVAS}; }}

QLabel#metric {{ font-size: 13px; color: {C_MUTED}; }}
QLabel#metricValue {{ font-size: 15px; font-weight: 850; color: {C_TEXT}; }}
QPlainTextEdit#terminal {{
    font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px;
    background: {C_PANEL}; color: {C_TEXT}; border-color: {C_BORDER};
}}
"""


# ---- workflow helpers -----------------------------------------------------
def make_node(node_type: str, **overrides) -> dict:
    bp = NODE_BLUEPRINTS[node_type]
    return {
        "id": overrides.get("id", node_type),
        "type": node_type,
        "title": bp["title"],
        "subtitle": bp["subtitle"],
        "x": overrides.get("x", bp["x"]),
        "y": overrides.get("y", bp["y"]),
        "enabled": overrides.get("enabled", True),
        "status": "idle",
        "config": {**bp["config"], **overrides.get("config", {})},
    }


def default_workflow() -> dict:
    return {
        "nodes": [make_node("input"), make_node("detector"), make_node("filter"), make_node("preview"), make_node("alert")],
        "edges": [[INPUT_NODE_ID, "detector"], ["detector", "filter"], ["filter", "preview"], ["filter", "alert"]],
    }


def input_subtitle(node: dict) -> str:
    return "File source" if (node.get("config", {}).get("sourceType") or "camera") == "file" else "Camera capture"


def normalize_gui(workflow: dict) -> dict:
    """Apply blueprint defaults / titles so older saved files stay valid."""
    engine.normalize_workflow(workflow)
    for node in workflow.get("nodes", []):
        bp = NODE_BLUEPRINTS.get(node["type"])
        if not bp:
            continue
        node["title"] = node.get("customTitle") or bp["title"]
        node["config"] = {**bp["config"], **(node.get("config") or {})}
        node["subtitle"] = input_subtitle(node) if node["type"] == "input" else bp["subtitle"]
        node.setdefault("enabled", True)
        node.setdefault("status", "idle")
    return workflow


def load_workflow() -> dict:
    try:
        if WORKFLOW_FILE.exists():
            data = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
            if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
                return normalize_gui(data)
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return normalize_gui(default_workflow())


def save_workflow(workflow: dict) -> None:
    try:
        WORKFLOW_FILE.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    except OSError:
        pass


# ---- palette drag source --------------------------------------------------
class PaletteNodeButton(QPushButton):
    def __init__(self, node_type: str, label: str) -> None:
        super().__init__(label)
        self.node_type = node_type
        self._drag_start = None
        self.setObjectName("palette")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag onto the canvas, or click to add/select this node.")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton) or self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        if (event.position().toPoint() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        mime = QMimeData()
        mime.setData(NODE_MIME_TYPE, self.node_type.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        self.render(pixmap)
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.position().toPoint())
        drag.exec(Qt.DropAction.CopyAction)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_start = None

    def mouseReleaseEvent(self, event) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_start = None
        super().mouseReleaseEvent(event)


# ---- graphics: edges & nodes ----------------------------------------------
class EdgeItem(QGraphicsPathItem):
    def __init__(self, from_id: str, to_id: str) -> None:
        super().__init__()
        self.from_id = from_id
        self.to_id = to_id
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        pen = QPen(QColor("#5d6b82"), 2.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

    def update_path(self, p1: QPointF, p2: QPointF) -> None:
        path = QPainterPath(p1)
        dx = max(70.0, abs(p2.x() - p1.x()) / 2)
        direction = 1 if p2.x() >= p1.x() else -1
        path.cubicTo(p1.x() + dx * direction, p1.y(), p2.x() - dx * direction, p2.y(), p2.x(), p2.y())
        self.setPath(path)

    def paint(self, painter, option, widget=None) -> None:
        selected = self.isSelected()
        pen = QPen(QColor(C_ACCENT if selected else "#5d6b82"), 3.2 if selected else 2.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        super().paint(painter, option, widget)


class NodeItem(QGraphicsObject):
    def __init__(self, node: dict, editor: "NodeEditor") -> None:
        super().__init__()
        self.node = node
        self.editor = editor
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPos(node["x"], node["y"])
        self.setZValue(1)
        self._toggle_rect = QRectF(NODE_W - 52, 10, 20, 20)
        self._delete_rect = QRectF(NODE_W - 28, 10, 20, 20)

    def boundingRect(self) -> QRectF:
        return QRectF(-9, -9, NODE_W + 18, NODE_H + 18)

    def input_pos(self) -> QPointF:
        return self.mapToScene(0, PORT_Y)

    def output_pos(self) -> QPointF:
        return self.mapToScene(NODE_W, PORT_Y)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node["x"] = round(self.pos().x())
            self.node["y"] = round(self.pos().y())
            self.editor.update_edges()
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged and value:
            self.editor.select_node(self.node["id"])
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        pos = event.pos()
        node_id = self.node["id"]
        # Defer mutations that rebuild the scene: removing/toggling can delete
        # this very item, so don't do it while inside its own event handler.
        if self._toggle_rect.contains(pos):
            QTimer.singleShot(0, lambda: self.editor.toggle_node(node_id))
            event.accept()
            return
        if self._delete_rect.contains(pos):
            QTimer.singleShot(0, lambda: self.editor.remove_node(node_id))
            event.accept()
            return
        self.editor.select_node(self.node["id"])
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.editor.persist()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        node = self.node
        enabled = node.get("enabled", True)
        selected = node["id"] == self.editor.selected_id
        status = node.get("status", "idle") if enabled else "off"
        accent = QColor(STATUS_COLORS.get(status, C_MUTED))
        node_color = QColor(NODE_TYPE_COLORS.get(node["type"], C_ACCENT2))

        body = QRectF(0, 0, NODE_W, NODE_H)
        path = QPainterPath()
        path.addRoundedRect(body, 13, 13)

        shadow_path = QPainterPath()
        shadow_path.addRoundedRect(body.translated(0, 4), 13, 13)
        shadow = QColor(0, 0, 0, 80 if enabled else 45)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillPath(shadow_path, QBrush(shadow))

        painter.setPen(QPen(QColor(C_ACCENT if selected else C_BORDER), 2.2 if selected else 1.2))
        painter.setBrush(QBrush(QColor(C_CARD if enabled else "#141c2b")))
        painter.drawPath(path)

        # n8n-like node type badge.
        icon_box = QRectF(15, 17, 40, 40)
        icon_path = QPainterPath()
        icon_path.addRoundedRect(icon_box, 12, 12)
        icon_bg = QColor(node_color)
        icon_bg.setAlpha(42 if enabled else 20)
        painter.fillPath(icon_path, QBrush(icon_bg))
        painter.setPen(QPen(node_color, 1.4))
        painter.drawPath(icon_path)

        self._paint_node_symbol(painter, icon_box.adjusted(8, 8, -8, -8), node["type"], node_color if enabled else QColor(C_MUTED))

        # title + subtitle
        painter.setPen(QColor(C_TEXT if enabled else C_MUTED))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(QRectF(66, 18, NODE_W - 128, 22), Qt.AlignmentFlag.AlignVCenter, node["title"])
        painter.setPen(QColor(C_MUTED))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(66, 41, NODE_W - 86, 18), Qt.AlignmentFlag.AlignVCenter, node.get("subtitle", ""))

        # status pill
        label = (node.get("status", "idle") if enabled else "off").capitalize()
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        pill_w = painter.fontMetrics().horizontalAdvance(label) + 18
        pill = QRectF(66, 61, pill_w, 18)
        pp = QPainterPath()
        pp.addRoundedRect(pill, 9, 9)
        pc = QColor(accent)
        pc.setAlpha(32)
        painter.fillPath(pp, QBrush(pc))
        painter.setPen(accent)
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, label)

        # toolbar buttons (toggle / delete)
        self._paint_glyph(painter, self._toggle_rect, "⏻", QColor(C_MUTED))
        self._paint_glyph(painter, self._delete_rect, "🗑", QColor("#c66"))

        # ports
        in_active = node["id"] != INPUT_NODE_ID
        if in_active:
            self._paint_port(painter, QPointF(0, PORT_Y))
        self._paint_port(painter, QPointF(NODE_W, PORT_Y))

    def _paint_glyph(self, painter: QPainter, rect: QRectF, glyph: str, color: QColor) -> None:
        painter.setFont(QFont("Segoe UI Symbol", 8))
        painter.setPen(color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, glyph)

    def _paint_node_symbol(self, painter: QPainter, rect: QRectF, node_type: str, color: QColor) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(color, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        if node_type == "input":
            painter.drawRoundedRect(QRectF(x + 2, y + 5, w - 4, h - 10), 4, 4)
            play = QPainterPath()
            play.moveTo(x + w * 0.42, y + h * 0.34)
            play.lineTo(x + w * 0.42, y + h * 0.66)
            play.lineTo(x + w * 0.68, y + h * 0.5)
            play.closeSubpath()
            painter.fillPath(play, QBrush(color))
        elif node_type == "detector":
            painter.drawEllipse(QRectF(x + 2, y + 2, w * 0.58, h * 0.58))
            painter.drawLine(QPointF(x + w * 0.62, y + h * 0.62), QPointF(x + w - 2, y + h - 2))
        elif node_type == "segmenter":
            painter.drawEllipse(QRectF(x + 2, y + 3, 7, 7))
            painter.drawEllipse(QRectF(x + 2, y + h - 10, 7, 7))
            painter.drawLine(QPointF(x + 9, y + 8), QPointF(x + w - 2, y + h - 3))
            painter.drawLine(QPointF(x + 9, y + h - 8), QPointF(x + w - 2, y + 3))
        elif node_type == "classifier":
            tag = QPainterPath()
            tag.moveTo(x + 2, y + 5)
            tag.lineTo(x + w * 0.62, y + 5)
            tag.lineTo(x + w - 2, y + h * 0.5)
            tag.lineTo(x + w * 0.62, y + h - 5)
            tag.lineTo(x + 2, y + h - 5)
            tag.closeSubpath()
            painter.drawPath(tag)
            painter.drawEllipse(QRectF(x + w * 0.55, y + h * 0.41, 3.5, 3.5))
        elif node_type == "filter":
            for yy, knob_x in ((y + 5, x + w * 0.35), (y + h * 0.5, x + w * 0.68), (y + h - 5, x + w * 0.48)):
                painter.drawLine(QPointF(x + 2, yy), QPointF(x + w - 2, yy))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(knob_x, yy), 2.8, 2.8)
                painter.setBrush(Qt.BrushStyle.NoBrush)
        elif node_type == "preview":
            painter.drawRoundedRect(QRectF(x + 2, y + 3, w - 4, h - 8), 3, 3)
            painter.drawLine(QPointF(x + w * 0.38, y + h - 2), QPointF(x + w * 0.62, y + h - 2))
        elif node_type == "alert":
            bell = QPainterPath()
            bell.moveTo(x + w * 0.22, y + h * 0.68)
            bell.cubicTo(x + w * 0.26, y + h * 0.42, x + w * 0.3, y + h * 0.22, x + w * 0.5, y + h * 0.22)
            bell.cubicTo(x + w * 0.7, y + h * 0.22, x + w * 0.74, y + h * 0.42, x + w * 0.78, y + h * 0.68)
            painter.drawPath(bell)
            painter.drawLine(QPointF(x + w * 0.18, y + h * 0.68), QPointF(x + w * 0.82, y + h * 0.68))
            painter.drawEllipse(QPointF(x + w * 0.5, y + h * 0.8), 2, 2)
        else:
            painter.drawEllipse(rect.adjusted(3, 3, -3, -3))
        painter.restore()

    def _paint_port(self, painter: QPainter, center: QPointF) -> None:
        painter.setPen(QPen(QColor(C_BORDER_STRONG), 1.6))
        painter.setBrush(QBrush(QColor(C_PANEL)))
        painter.drawEllipse(center, PORT_R, PORT_R)
        painter.setBrush(QBrush(QColor(C_ACCENT)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, 2.8, 2.8)


class NodeEditor(QGraphicsView):
    nodeSelected = Signal(str)
    workflowChanged = Signal()

    def __init__(self, workflow: dict) -> None:
        self.scene_obj = QGraphicsScene()
        super().__init__(self.scene_obj)
        self.workflow = workflow
        self.selected_id = INPUT_NODE_ID
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []
        self._connecting_from: str | None = None
        self._temp_edge: QGraphicsPathItem | None = None

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setSceneRect(0, 0, 1400, 900)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.rebuild()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor(C_CANVAS))

        major_step = 32
        minor_step = 16
        left = int(rect.left()) - (int(rect.left()) % minor_step)
        top = int(rect.top()) - (int(rect.top()) % minor_step)
        dot = QColor("#253044")
        major_dot = QColor("#334157")

        painter.setPen(Qt.PenStyle.NoPen)
        y = top
        while y < rect.bottom():
            x = left
            while x < rect.right():
                is_major = x % major_step == 0 and y % major_step == 0
                painter.setBrush(QBrush(major_dot if is_major else dot))
                painter.drawEllipse(QPointF(x, y), 1.15 if is_major else 0.8, 1.15 if is_major else 0.8)
                x += minor_step
            y += minor_step

    # -- building ---
    def rebuild(self) -> None:
        self.scene_obj.clear()
        self.node_items.clear()
        self.edge_items.clear()
        for node in self.workflow["nodes"]:
            item = NodeItem(node, self)
            self.scene_obj.addItem(item)
            self.node_items[node["id"]] = item
        for edge in self.workflow["edges"]:
            self._add_edge_item(edge[0], edge[1])
        self.refresh_statuses()
        self.update_edges()

    def _add_edge_item(self, from_id: str, to_id: str) -> None:
        if from_id not in self.node_items or to_id not in self.node_items:
            return
        edge = EdgeItem(from_id, to_id)
        self.scene_obj.addItem(edge)
        self.edge_items.append(edge)

    def update_edges(self) -> None:
        for edge in self.edge_items:
            a = self.node_items.get(edge.from_id)
            b = self.node_items.get(edge.to_id)
            if a and b:
                edge.update_path(a.output_pos(), b.input_pos())

    def refresh_statuses(self, running: bool = False, override: str | None = None) -> None:
        source_id = engine.source_node_id(self.workflow)
        for node in self.workflow["nodes"]:
            nid = node["id"]
            active = node.get("enabled", True) and (
                nid == INPUT_NODE_ID or engine.has_active_path(self.workflow, source_id, nid)
            )
            if nid != INPUT_NODE_ID and not active:
                node["status"] = "unlinked"
            elif override:
                node["status"] = override
            else:
                node["status"] = "running" if running else "idle"
        for item in self.node_items.values():
            item.update()

    # -- selection / mutation ---
    def select_node(self, node_id: str) -> None:
        self.selected_id = node_id
        for item in self.node_items.values():
            item.update()
        self.nodeSelected.emit(node_id)

    def persist(self) -> None:
        save_workflow(self.workflow)
        self.workflowChanged.emit()

    def get_node(self, node_id: str) -> dict | None:
        return next((n for n in self.workflow["nodes"] if n["id"] == node_id), None)

    def add_node(self, node_type: str, x: float | None = None, y: float | None = None) -> None:
        existing = next((n for n in self.workflow["nodes"] if n["type"] == node_type), None)
        if existing:
            if x is not None and y is not None:
                existing["x"] = round(x)
                existing["y"] = round(y)
                item = self.node_items.get(existing["id"])
                if item:
                    item.setPos(existing["x"], existing["y"])
                self.update_edges()
                self.persist()
            self.select_node(existing["id"])
            return
        bp = NODE_BLUEPRINTS[node_type]
        node = make_node(node_type, x=x if x is not None else bp["x"], y=y if y is not None else bp["y"])
        self.workflow["nodes"].append(node)
        self.rebuild()
        self.select_node(node["id"])
        self.persist()

    def _drop_node_type(self, event) -> str | None:
        mime = event.mimeData()
        if not mime.hasFormat(NODE_MIME_TYPE):
            return None
        node_type = bytes(mime.data(NODE_MIME_TYPE)).decode("utf-8")
        return node_type if node_type in NODE_BLUEPRINTS else None

    def _drop_scene_top_left(self, event) -> QPointF:
        scene_pos = self.mapToScene(event.position().toPoint())
        scene_rect = self.sceneRect()
        x = min(max(scene_pos.x() - NODE_W / 2, scene_rect.left()), scene_rect.right() - NODE_W)
        y = min(max(scene_pos.y() - NODE_H / 2, scene_rect.top()), scene_rect.bottom() - NODE_H)
        return QPointF(x, y)

    def dragEnterEvent(self, event) -> None:
        if self._drop_node_type(event):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._drop_node_type(event):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        node_type = self._drop_node_type(event)
        if not node_type:
            super().dropEvent(event)
            return
        pos = self._drop_scene_top_left(event)
        self.add_node(node_type, pos.x(), pos.y())
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()

    def remove_node(self, node_id: str) -> None:
        self.workflow["nodes"] = [n for n in self.workflow["nodes"] if n["id"] != node_id]
        self.workflow["edges"] = [e for e in self.workflow["edges"] if e[0] != node_id and e[1] != node_id]
        if self.selected_id == node_id:
            self.selected_id = self.workflow["nodes"][0]["id"] if self.workflow["nodes"] else None
        self.rebuild()
        if self.selected_id:
            self.select_node(self.selected_id)
        else:
            self.nodeSelected.emit("")
        self.persist()

    def toggle_node(self, node_id: str) -> None:
        node = self.get_node(node_id)
        if not node:
            return
        node["enabled"] = not node.get("enabled", True)
        self.refresh_statuses()
        self.select_node(node_id)
        self.persist()

    def rename_node(self, node_id: str) -> None:
        node = self.get_node(node_id)
        if not node:
            return
        current = node.get("customTitle") or node.get("title") or ""
        text, ok = QInputDialog.getText(self, "Rename node", "Node name:", text=current)
        if not ok:
            return
        text = text.strip()
        if text:
            node["customTitle"] = text
            node["title"] = text
        else:
            node.pop("customTitle", None)
            node["title"] = NODE_BLUEPRINTS[node["type"]]["title"]
        self.node_items[node_id].update()
        self.select_node(node_id)
        self.persist()

    def connect_nodes(self, from_id: str, to_id: str) -> None:
        if not from_id or not to_id or from_id == to_id:
            return
        if any(e[0] == from_id and e[1] == to_id for e in self.workflow["edges"]):
            return
        self.workflow["edges"].append([from_id, to_id])
        self._add_edge_item(from_id, to_id)
        self.refresh_statuses()
        self.update_edges()
        self.persist()

    # -- port hit testing & connections ---
    def _port_at(self, scene_pos: QPointF, kind: str) -> str | None:
        for nid, item in self.node_items.items():
            if kind == "out":
                center = item.output_pos()
            else:
                if nid == INPUT_NODE_ID:
                    continue
                center = item.input_pos()
            if (scene_pos - center).manhattanLength() < PORT_R * 2.2:
                return nid
        return None

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        out_id = self._port_at(scene_pos, "out")
        if out_id is not None:
            self._connecting_from = out_id
            self._temp_edge = QGraphicsPathItem()
            pen = QPen(QColor(C_ACCENT), 2.5, Qt.PenStyle.DashLine)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._temp_edge.setPen(pen)
            self._temp_edge.setZValue(5)
            self.scene_obj.addItem(self._temp_edge)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._temp_edge and self._connecting_from:
            start = self.node_items[self._connecting_from].output_pos()
            end = self.mapToScene(event.position().toPoint())
            path = QPainterPath(start)
            dx = max(70.0, abs(end.x() - start.x()) / 2)
            d = 1 if end.x() >= start.x() else -1
            path.cubicTo(start.x() + dx * d, start.y(), end.x() - dx * d, end.y(), end.x(), end.y())
            self._temp_edge.setPath(path)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._temp_edge and self._connecting_from:
            scene_pos = self.mapToScene(event.position().toPoint())
            to_id = self._port_at(scene_pos, "in")
            self.scene_obj.removeItem(self._temp_edge)
            self._temp_edge = None
            if to_id:
                self.connect_nodes(self._connecting_from, to_id)
            self._connecting_from = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # delete selected edges first, else selected node
            selected_edges = [e for e in self.edge_items if e.isSelected()]
            if selected_edges:
                for edge in selected_edges:
                    self.workflow["edges"] = [
                        e for e in self.workflow["edges"]
                        if not (e[0] == edge.from_id and e[1] == edge.to_id)
                    ]
                self.rebuild()
                if self.selected_id:
                    self.select_node(self.selected_id)
                self.persist()
                return
            if self.selected_id:
                self.remove_node(self.selected_id)
                return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        target = None
        for nid, item in self.node_items.items():
            if item.sceneBoundingRect().contains(scene_pos):
                target = nid
                break
        if not target:
            super().contextMenuEvent(event)
            return
        node = self.get_node(target)
        self.select_node(target)
        menu = QMenu(self)
        menu.addAction("Rename", lambda: self.rename_node(target))
        menu.addAction("Disable" if node.get("enabled", True) else "Enable", lambda: self.toggle_node(target))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.remove_node(target))
        menu.exec(event.globalPos())


# ---- inspector field widgets ----------------------------------------------
class FloatSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, value: float, lo: float, hi: float, step: float) -> None:
        super().__init__()
        self._lo, self._step = lo, step
        steps = round((hi - lo) / step)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._label = QLabel(f"{value:.2f}")
        self._label.setStyleSheet(f"color: {C_ACCENT}; font-weight: 700;")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(steps)
        self._slider.setValue(round((value - lo) / step))
        self._slider.valueChanged.connect(self._on_change)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._slider, 1)
        row.addWidget(self._label)
        layout.addLayout(row)

    def _on_change(self, raw: int) -> None:
        value = self._lo + raw * self._step
        self._label.setText(f"{value:.2f}")
        self.valueChanged.emit(value)


class Inspector(QScrollArea):
    def __init__(self, editor: NodeEditor, get_devices) -> None:
        super().__init__()
        self.editor = editor
        self.get_devices = get_devices
        self.setWidgetResizable(True)
        self._host = QWidget()
        self._layout = QVBoxLayout(self._host)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(12)
        self._layout.addStretch(1)
        self.setWidget(self._host)
        self.current_id: str | None = None

    def show_node(self, node_id: str) -> None:
        self.current_id = node_id
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        node = self.editor.get_node(node_id)
        if not node:
            empty = QLabel("Select a node, or add one from the palette.")
            empty.setStyleSheet(f"color: {C_MUTED};")
            empty.setWordWrap(True)
            self._layout.addWidget(empty)
            self._layout.addStretch(1)
            return

        self._add(self._field("Enabled", self._checkbox(node.get("enabled", True), lambda v: self._set_enabled(node, v))))
        cfg = node["config"]
        t = node["type"]

        if t == "input":
            self._add(self._field("Input type", self._combo([("camera", "Camera"), ("file", "File")], cfg.get("sourceType", "camera"), lambda v: self._set_source_type(node, v))))
            if (cfg.get("sourceType") or "camera") == "file":
                self._add(self._field("File path", self._file_row(cfg.get("filePath", ""), lambda v: self._set(node, "filePath", v))))
                self._add(self._field("Loop file", self._checkbox(cfg.get("loop", True), lambda v: self._set(node, "loop", v))))
            else:
                self._add(self._field("OpenCV source", self._line(str(cfg.get("source", 0)), lambda v: self._set(node, "source", int(v) if v.isdigit() else v))))
                self._add(self._field("Width", self._spin(cfg.get("width", 1280), 320, 3840, lambda v: self._set(node, "width", v))))
                self._add(self._field("Height", self._spin(cfg.get("height", 720), 240, 2160, lambda v: self._set(node, "height", v))))

        elif t == "detector":
            detector_engine = cfg.get("engine", "yolo26")
            self._add(self._field("Engine", self._combo([("yolo26", "YOLO26 object detection"), ("rfdetr", "RF-DETR object detection")], detector_engine, lambda v: self._set_engine(node, v))))
            self._add(self._field("Confidence", self._fslider(cfg.get("threshold", 0.55), 0.1, 0.95, 0.05, lambda v: self._set(node, "threshold", v))))
            if detector_engine == "rfdetr":
                self._add(self._field("RF-DETR model", self._combo(RFDETR_DETECTION_MODEL_OPTIONS, cfg.get("rfdetrModel", "rfdetr-nano"), lambda v: self._set(node, "rfdetrModel", v))))
                self._add(self._field("RF-DETR checkpoint path", self._file_row(
                    cfg.get("rfdetrCheckpoint", ""),
                    lambda v: self._set(node, "rfdetrCheckpoint", v),
                    title="Select RF-DETR checkpoint",
                    placeholder="Optional .pth, .pt, or .ckpt checkpoint",
                    file_filter="RF-DETR checkpoints (*.pth *.pt *.ckpt);;All files (*.*)",
                )))
            else:
                self._add(self._field("YOLO26 model", self._combo(DETECTION_MODEL_OPTIONS, cfg.get("yoloModel", "yolo26n.pt"), lambda v: self._set(node, "yoloModel", v))))
            self._add(self._field("Device", self._combo(self.get_devices(), cfg.get("device", "auto"), lambda v: self._set(node, "device", v))))
            if detector_engine == "yolo26":
                self._add(self._field("Image size", self._spin(cfg.get("imgsz", 640), 320, 1280, lambda v: self._set(node, "imgsz", v))))
                self._add(self._field("End-to-end head", self._checkbox(cfg.get("end2end", True), lambda v: self._set(node, "end2end", v))))

        elif t == "segmenter":
            segmenter_engine = cfg.get("engine", "yolo26")
            self._add(self._field("Engine", self._combo([("yolo26", "YOLO26 instance segmentation"), ("rfdetr", "RF-DETR instance segmentation"), ("sam3", "SAM 3 concept segmentation")], segmenter_engine, lambda v: self._set_engine(node, v))))
            self._add(self._field("Confidence", self._fslider(cfg.get("threshold", 0.55), 0.1, 0.95, 0.05, lambda v: self._set(node, "threshold", v))))
            if segmenter_engine == "sam3":
                self._add(self._field("SAM 3 checkpoint path", self._line(cfg.get("samCheckpoint", ""), lambda v: self._set(node, "samCheckpoint", v))))
                self._add(self._field("Concept prompts", self._textarea(cfg.get("concepts", "person"), lambda v: self._set(node, "concepts", v))))
            elif segmenter_engine == "rfdetr":
                self._add(self._field("RF-DETR segmentation model", self._combo(RFDETR_SEGMENTATION_MODEL_OPTIONS, cfg.get("rfdetrModel", "rfdetr-seg-nano"), lambda v: self._set(node, "rfdetrModel", v))))
                self._add(self._field("RF-DETR checkpoint path", self._file_row(
                    cfg.get("rfdetrCheckpoint", ""),
                    lambda v: self._set(node, "rfdetrCheckpoint", v),
                    title="Select RF-DETR checkpoint",
                    placeholder="Optional .pth, .pt, or .ckpt checkpoint",
                    file_filter="RF-DETR checkpoints (*.pth *.pt *.ckpt);;All files (*.*)",
                )))
            else:
                self._add(self._field("YOLO26 segmentation model", self._combo(SEGMENTATION_MODEL_OPTIONS, cfg.get("yoloModel", "yolo26n-seg.pt"), lambda v: self._set(node, "yoloModel", v))))
            self._add(self._field("Device", self._combo(self.get_devices(), cfg.get("device", "auto"), lambda v: self._set(node, "device", v))))
            if segmenter_engine != "rfdetr":
                self._add(self._field("Image size", self._spin(cfg.get("imgsz", 640), 320, 1280, lambda v: self._set(node, "imgsz", v))))
            if segmenter_engine == "yolo26":
                self._add(self._field("End-to-end head", self._checkbox(cfg.get("end2end", True), lambda v: self._set(node, "end2end", v))))

        elif t == "classifier":
            self._add(self._field("Confidence", self._fslider(cfg.get("threshold", 0.25), 0.05, 0.95, 0.05, lambda v: self._set(node, "threshold", v))))
            self._add(self._field("YOLO26 classification model", self._combo(CLASSIFICATION_MODEL_OPTIONS, cfg.get("yoloModel", "yolo26n-cls.pt"), lambda v: self._set(node, "yoloModel", v))))
            self._add(self._field("Device", self._combo(self.get_devices(), cfg.get("device", "auto"), lambda v: self._set(node, "device", v))))
            self._add(self._field("Image size", self._spin(cfg.get("imgsz", 224), 64, 640, lambda v: self._set(node, "imgsz", v))))

        elif t == "filter":
            self._add(self._field("Allowed classes", self._textarea(cfg.get("classes", ""), lambda v: self._set(node, "classes", v))))
            self._add(self._field("Minimum count", self._spin(cfg.get("minCount", 1), 1, 100, lambda v: self._set(node, "minCount", v))))

        elif t == "preview":
            self._add(self._field("Show masks", self._checkbox(cfg.get("showMasks", True), lambda v: self._set(node, "showMasks", v))))
            self._add(self._field("Mask opacity", self._fslider(cfg.get("maskOpacity", 0.35), 0.1, 0.8, 0.05, lambda v: self._set(node, "maskOpacity", v))))
            self._add(self._field("Show boxes", self._checkbox(cfg.get("showBoxes", True), lambda v: self._set(node, "showBoxes", v))))
            self._add(self._field("Show labels", self._checkbox(cfg.get("showLabels", True), lambda v: self._set(node, "showLabels", v))))
            self._add(self._field("Use class filter", self._checkbox(cfg.get("useFilter", False), lambda v: self._set(node, "useFilter", v))))

        elif t == "alert":
            self._add(self._field("Cooldown seconds", self._spin(cfg.get("cooldownSeconds", 5), 1, 120, lambda v: self._set(node, "cooldownSeconds", v))))
            self._add(self._field("Alert message", self._textarea(cfg.get("message", ""), lambda v: self._set(node, "message", v))))

        self._layout.addStretch(1)

    # -- field plumbing ---
    def _add(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def _field(self, label: str, control: QWidget) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        cap = QLabel(label)
        cap.setStyleSheet(f"color: {C_MUTED}; font-size: 11px; font-weight: 600;")
        lay.addWidget(cap)
        lay.addWidget(control)
        return wrap

    def _set(self, node: dict, key: str, value) -> None:
        node["config"][key] = value
        self.editor.persist()

    def _set_enabled(self, node: dict, value: bool) -> None:
        node["enabled"] = value
        self.editor.refresh_statuses()
        self.editor.node_items[node["id"]].update()
        self.editor.persist()

    def _set_source_type(self, node: dict, value: str) -> None:
        node["config"]["sourceType"] = value
        node["subtitle"] = input_subtitle(node)
        self.editor.node_items[node["id"]].update()
        self.editor.persist()
        self.show_node(node["id"])

    def _set_engine(self, node: dict, value: str) -> None:
        node["config"]["engine"] = value
        if value == "rfdetr":
            default_model = "rfdetr-seg-nano" if node["type"] == "segmenter" else "rfdetr-nano"
            node["config"].setdefault("rfdetrModel", default_model)
        self.editor.persist()
        self.show_node(node["id"])

    def _checkbox(self, value: bool, on_change) -> QCheckBox:
        box = QCheckBox()
        box.setChecked(bool(value))
        box.toggled.connect(on_change)
        return box

    def _combo(self, options, value, on_change) -> QComboBox:
        combo = QComboBox()
        idx = 0
        for i, (val, label) in enumerate(options):
            combo.addItem(label, val)
            if val == value:
                idx = i
        combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(lambda i: on_change(combo.itemData(i)))
        return combo

    def _line(self, value: str, on_change) -> QLineEdit:
        line = QLineEdit(str(value))
        line.textChanged.connect(on_change)
        return line

    def _spin(self, value, lo, hi, on_change) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(int(value))
        spin.valueChanged.connect(on_change)
        return spin

    def _fslider(self, value, lo, hi, step, on_change) -> FloatSlider:
        slider = FloatSlider(float(value), lo, hi, step)
        slider.valueChanged.connect(on_change)
        return slider

    def _textarea(self, value: str, on_change) -> QTextEdit:
        area = QTextEdit(str(value))
        area.setFixedHeight(64)
        area.textChanged.connect(lambda: on_change(area.toPlainText()))
        return area

    def _file_row(
        self,
        value: str,
        on_change,
        title: str = "Select input file",
        placeholder: str = "Select an image or video file",
        file_filter: str = "Vision files (*.bmp *.jpg *.jpeg *.png *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv);;All files (*.*)",
    ) -> QWidget:
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        line = QLineEdit(str(value))
        line.setPlaceholderText(placeholder)
        line.textChanged.connect(on_change)
        button = QPushButton("📁")
        button.setFixedWidth(40)

        def browse():
            path, _ = QFileDialog.getOpenFileName(
                self, title, str(ROOT), file_filter,
            )
            if path:
                line.setText(path)
                on_change(path)

        button.clicked.connect(browse)
        lay.addWidget(line, 1)
        lay.addWidget(button)
        return wrap


# ---- runtime device probe (background) ------------------------------------
class DeviceProbe(QThread):
    done = Signal(dict)

    def run(self) -> None:
        try:
            self.done.emit(engine.runtime_devices())
        except Exception:  # noqa: BLE001
            self.done.emit({})


# ---- bridge between worker thread and GUI ---------------------------------
class RunnerBridge(QObject):
    sig_log = Signal(str)
    sig_event = Signal(str, str)
    sig_status = Signal(dict)
    sig_frame = Signal(object, bool)

    def emit_frame(self, frame_bgr, shown: bool) -> None:
        if frame_bgr is None:
            self.sig_frame.emit(None, shown)
            return
        h, w = frame_bgr.shape[:2]
        image = QImage(frame_bgr.data, w, h, frame_bgr.strides[0], QImage.Format.Format_BGR888).copy()
        self.sig_frame.emit(image, shown)


# ---- live preview window --------------------------------------------------
class PreviewWindow(QWidget):
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VisoNode — Live preview")
        self.setObjectName("root")
        self.resize(960, 600)
        self.setStyleSheet(f"background: #05080f;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("Waiting for frames…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {C_MUTED}; font-size: 14px;")
        self._label.setMinimumSize(320, 240)
        layout.addWidget(self._label)
        self._pixmap: QPixmap | None = None

    def set_image(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            return
        self._label.setPixmap(
            self._pixmap.scaled(self._label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


# ---- main window ----------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.workflow = load_workflow()
        self.runtime_devices: dict = {}
        self.setWindowTitle("VisoNode")
        self.resize(1480, 920)

        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_topbar())

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setHandleWidth(10)
        body.addWidget(self._build_palette())
        body.addWidget(self._build_canvas())
        body.addWidget(self._build_inspector_panel())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setStretchFactor(2, 0)
        body.setSizes([220, 900, 340])

        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.setHandleWidth(10)
        main_split.addWidget(body)
        main_split.addWidget(self._build_terminal())
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 0)
        main_split.setSizes([700, 200])
        outer.addWidget(main_split, 1)

        # runner wiring
        self.bridge = RunnerBridge()
        self.bridge.sig_log.connect(self._on_log)
        self.bridge.sig_event.connect(self._on_event)
        self.bridge.sig_status.connect(self._on_status)
        self.bridge.sig_frame.connect(self._on_frame)
        self.runner = core.WorkflowRunner(
            on_log=self.bridge.sig_log.emit,
            on_event=self.bridge.sig_event.emit,
            on_status=self.bridge.sig_status.emit,
            on_frame=self.bridge.emit_frame,
        )
        self.preview = PreviewWindow()
        self.preview.closed.connect(self._on_preview_closed)

        self.editor.nodeSelected.connect(self._on_node_selected)
        self.inspector.show_node(self.editor.selected_id)
        self._set_version_label()
        self._log_line("VisoNode desktop ready. Build a graph and press Run.")

        # probe devices in background
        self._probe = DeviceProbe()
        self._probe.done.connect(self._on_devices)
        self._probe.start()

    # -- top bar ---
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topbar")
        bar.setFixedHeight(58)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 16, 0)
        mark = QLabel("VN")
        mark.setObjectName("brandMark")
        title = QLabel("VisoNode")
        title.setObjectName("brandTitle")
        self.version_badge = QLabel("v…")
        self.version_badge.setObjectName("versionBadge")
        lay.addWidget(mark)
        lay.addWidget(title)
        lay.addWidget(self.version_badge)
        lay.addStretch(1)

        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.clicked.connect(self._reset_workflow)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self._stop_workflow)
        self.stop_btn.setEnabled(False)
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._start_workflow)
        lay.addWidget(self.reset_btn)
        lay.addWidget(self.stop_btn)
        lay.addWidget(self.run_btn)
        return bar

    # -- palette ---
    def _build_palette(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(224)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 16, 14, 14)
        lay.setSpacing(8)
        heading = QLabel("NODES")
        heading.setObjectName("panelHeading")
        lay.addWidget(heading)
        for node_type, bp in NODE_BLUEPRINTS.items():
            btn = PaletteNodeButton(node_type, f"{NODE_ICONS[node_type]}   {bp['title']}")
            btn.clicked.connect(lambda _, t=node_type: self.editor.add_node(t))
            lay.addWidget(btn)
        lay.addSpacing(6)
        hint = QLabel("Drag nodes onto the canvas. Drag a node's right port onto another node's left port to connect them.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        lay.addWidget(hint)
        lay.addStretch(1)
        return panel

    # -- canvas ---
    def _build_canvas(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        toolbar = QFrame()
        toolbar.setObjectName("panel")
        tlay = QHBoxLayout(toolbar)
        tlay.setContentsMargins(16, 9, 16, 9)
        wf = QLabel("AI vision workflow")
        wf.setStyleSheet("font-weight: 700;")
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet(f"color: {C_MUTED};")
        tlay.addWidget(wf)
        editor_badge = QLabel("Editor")
        editor_badge.setObjectName("versionBadge")
        tlay.addWidget(editor_badge)
        tlay.addWidget(self.status_label)
        tlay.addStretch(1)
        self.fps_label = self._metric("fps", "0")
        self.obj_label = self._metric("objects", "0")
        self.device_label = self._metric("device", "CPU")
        for m in (self.fps_label, self.obj_label, self.device_label):
            tlay.addWidget(m)
        lay.addWidget(toolbar)

        self.editor = NodeEditor(self.workflow)
        self.editor.workflowChanged.connect(self._on_workflow_changed)
        lay.addWidget(self.editor, 1)
        return wrap

    def _metric(self, name: str, value: str) -> QWidget:
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(5)
        val = QLabel(value)
        val.setObjectName("metricValue")
        unit = QLabel(name)
        unit.setObjectName("metric")
        lay.addWidget(val)
        lay.addWidget(unit)
        wrap.value_label = val  # type: ignore[attr-defined]
        return wrap

    # -- inspector + events ---
    def _build_inspector_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(316)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 14)
        lay.setSpacing(10)

        head = QHBoxLayout()
        heading = QLabel("INSPECTOR")
        heading.setObjectName("panelHeading")
        self.inspector_node_label = QLabel("")
        self.inspector_node_label.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        head.addWidget(heading)
        head.addStretch(1)
        head.addWidget(self.inspector_node_label)
        lay.addLayout(head)

        self.inspector = Inspector(self.editor, self.device_options)
        lay.addWidget(self.inspector, 3)

        sep = QFrame()
        sep.setObjectName("hsep")
        lay.addWidget(sep)

        ev_heading = QLabel("EVENTS")
        ev_heading.setObjectName("panelHeading")
        lay.addWidget(ev_heading)
        self.event_list = QListWidget()
        self.event_list.setWordWrap(True)
        lay.addWidget(self.event_list, 2)
        return panel

    # -- terminal ---
    def _build_terminal(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("⌧  Python backend")
        title.setStyleSheet("font-weight: 700;")
        self.term_status = QLabel("●  Ready")
        self.term_status.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px;")
        header.addWidget(title)
        header.addWidget(self.term_status)
        header.addStretch(1)
        open_btn = QPushButton("Open terminal")
        open_btn.clicked.connect(self._open_external_terminal)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.terminal.clear())
        self.term_toggle = QPushButton("Hide")
        self.term_toggle.clicked.connect(self._toggle_terminal)
        for b in (open_btn, clear_btn, self.term_toggle):
            header.addWidget(b)
        lay.addLayout(header)

        self.terminal = QPlainTextEdit()
        self.terminal.setObjectName("terminal")
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumBlockCount(800)
        lay.addWidget(self.terminal, 1)
        return panel

    def _toggle_terminal(self) -> None:
        visible = self.terminal.isVisible()
        self.terminal.setVisible(not visible)
        self.term_toggle.setText("Show" if visible else "Hide")

    # -- device options ---
    def device_options(self):
        opts = [("auto", "Auto"), ("cpu", "CPU")]
        for dev in (self.runtime_devices.get("devices") or []):
            gb = f" {round(dev['memoryMb'] / 1024)}GB" if dev.get("memoryMb") else ""
            opts.append((dev["id"], f"{dev['id'].upper()} {dev.get('name', '')}{gb}"))
        return opts

    def _on_devices(self, payload: dict) -> None:
        self.runtime_devices = payload or {}
        node = self.editor.get_node(self.editor.selected_id)
        if node and node["type"] in ("detector", "segmenter", "classifier"):
            self.inspector.show_node(self.editor.selected_id)
        rec = self.runtime_devices.get("recommendation")
        if rec:
            self._log_line(rec)

    # -- node selection ---
    def _on_node_selected(self, node_id: str) -> None:
        if not node_id:
            self.inspector_node_label.setText("")
            self.inspector.show_node("")
            return
        node = self.editor.get_node(node_id)
        self.inspector_node_label.setText(node["title"] if node else "")
        self.inspector.show_node(node_id)

    def _on_workflow_changed(self) -> None:
        node = self.editor.get_node(self.editor.selected_id)
        if node:
            self.inspector_node_label.setText(node["title"])

    # -- run / stop ---
    def _start_workflow(self) -> None:
        if self.runner.is_running():
            return
        save_workflow(self.workflow)
        self.editor.refresh_statuses(override="starting")
        self.status_label.setText("Starting")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        try:
            self.runner.start(copy.deepcopy(self.workflow))
        except Exception as exc:  # noqa: BLE001
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Error")
            self.editor.refresh_statuses(override="error")
            self._log_line(f"ERROR: {exc}")
            QMessageBox.warning(self, "Could not start", str(exc))

    def _stop_workflow(self) -> None:
        self.runner.stop()

    def _on_preview_closed(self) -> None:
        if self.runner.is_running():
            self.runner.stop()

    # -- bridge slots (GUI thread) ---
    def _on_log(self, line: str) -> None:
        self._log_line(line)

    def _log_line(self, line: str) -> None:
        from time import strftime

        stamp = strftime("%H:%M:%S")
        low = line.lower()
        if low.startswith("error") or ": error" in low or "exception" in low:
            color = C_DANGER
        elif "warn" in low:
            color = C_WARN
        elif any(k in low for k in ("loaded", "running", "opened", "workflow")):
            color = C_ACCENT
        else:
            color = C_TEXT
        safe = (line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self.terminal.appendHtml(
            f'<span style="color:{C_MUTED}">{stamp}</span>&nbsp;&nbsp;<span style="color:{color}">{safe}</span>'
        )

    def _on_event(self, title: str, detail: str) -> None:
        from time import strftime

        item = QListWidgetItem(f"{strftime('%H:%M:%S')}  {title}\n{detail}")
        self.event_list.insertItem(0, item)
        while self.event_list.count() > 10:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _on_status(self, status: dict) -> None:
        state = status.get("state", "idle")
        running = bool(status.get("running"))
        self.status_label.setText(state.capitalize())
        self.fps_label.value_label.setText(str(status.get("fps", 0)))
        self.obj_label.value_label.setText(str(status.get("objectCount", 0)))
        self.device_label.value_label.setText(status.get("device", "CPU") or "CPU")
        if state == "error":
            self.editor.refresh_statuses(override="error")
            err = status.get("error")
            if err:
                self.term_status.setText("●  Error")
                self.term_status.setStyleSheet(f"color: {C_DANGER}; font-size: 11px;")
        elif running:
            self.editor.refresh_statuses(running=True)
        else:
            self.editor.refresh_statuses(running=False)

        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if not running:
            self.preview.hide()

    def _on_frame(self, image, shown: bool) -> None:
        if not shown or image is None:
            return
        if not self.preview.isVisible():
            self.preview.show()
        self.preview.set_image(image)

    # -- misc ---
    def _open_external_terminal(self) -> None:
        try:
            message = engine.open_external_terminal()
            self._log_line(f"{message} in {ROOT}")
        except Exception as exc:  # noqa: BLE001
            self._log_line(f"ERROR: {exc}")

    def _reset_workflow(self) -> None:
        if self.runner.is_running():
            self.runner.stop()
        self.workflow = normalize_gui(default_workflow())
        self.editor.workflow = self.workflow
        self.editor.selected_id = INPUT_NODE_ID
        self.editor.rebuild()
        self.editor.select_node(INPUT_NODE_ID)
        save_workflow(self.workflow)
        self._log_line("Workflow reset to the default graph.")

    def _set_version_label(self) -> None:
        info = dict(core.VERSION_INFO)
        info.update(core.git_version_info())
        version = info.get("version", "0.0.0")
        commit = info.get("commit")
        label = f"v{version}"
        if commit:
            label += f"+{commit}" + (".dirty" if info.get("dirty") else "")
        self.version_badge.setText(label)

    def closeEvent(self, event):
        try:
            self.runner.stop()
        except Exception:  # noqa: BLE001
            pass
        # let the background device probe finish so the interpreter can tear
        # down its QThread cleanly instead of aborting.
        if self._probe.isRunning():
            self._probe.wait(5000)
        self.preview.close()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="VisoNode desktop GUI")
    parser.add_argument("--version", action="version", version=f"VisoNode {core.APP_VERSION}")
    parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
