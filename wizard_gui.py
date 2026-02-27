import sys
import os
import logging
from collections import deque


def resource_path(relative_path):
    """Resolve path to bundled data file (PyInstaller-compatible)."""
    base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    return os.path.join(base, relative_path)

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSpinBox, QFileDialog, QWidget, QStackedWidget, QSizePolicy, QButtonGroup,
    QRadioButton, QGroupBox, QSlider,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QSize, Qt, QPoint
from PyQt5.QtGui import QColor, QPixmap, QPainter, QImage, QPen

from wizard_state import WizardState, WizardStep
from map_pipeline import (
    run_coastline, run_height_ocean,
    run_place_cc_random, run_place_cc_manual, undo_last_cc, clear_all_cc,
    run_place_resources_random, run_place_resource_manual, undo_last_resource, clear_all_resources,
    run_finalize, write_tmx,
)
from tileset_renderer import TilesetExtractor, MapRenderer
from procedural_map_generator_functions import _get_mirrored_positions
from generator_gui import FixedGridWidget, ColorChangingButton


# ==========================================================================
# Worker threads
# ==========================================================================

class CoastlineWorker(QThread):
    preview_signal = pyqtSignal(str, object)  # stage, height_map
    finished_signal = pyqtSignal()

    def __init__(self, state):
        super().__init__()
        self.state = state

    def run(self):
        logging.getLogger('procedural_map_generator_functions').setLevel(logging.WARNING)
        try:
            run_coastline(self.state, preview_cb=self.preview_signal.emit)
        finally:
            self.finished_signal.emit()


class HeightOceanWorker(QThread):
    preview_signal = pyqtSignal(str, object)
    finished_signal = pyqtSignal()

    def __init__(self, state, seed=None):
        super().__init__()
        self.state = state
        self.seed = seed

    def run(self):
        logging.getLogger('procedural_map_generator_functions').setLevel(logging.WARNING)
        try:
            run_height_ocean(self.state, seed=self.seed, preview_cb=self.preview_signal.emit)
        finally:
            self.finished_signal.emit()


class FinalizeWorker(QThread):
    preview_signal = pyqtSignal(str, object, object, object, object)  # stage, hmap, id, items, units
    finished_signal = pyqtSignal(str)  # output path

    def __init__(self, state):
        super().__init__()
        self.state = state

    def run(self):
        logging.getLogger('procedural_map_generator_functions').setLevel(logging.WARNING)
        try:
            run_finalize(self.state, preview_cb=self.preview_signal.emit)
            output_file = write_tmx(self.state)
            self.finished_signal.emit(output_file)
        except Exception as e:
            self.finished_signal.emit(f"ERROR: {e}")


# ==========================================================================
# StepBar
# ==========================================================================

class StepBar(QWidget):
    step_clicked = pyqtSignal(int)

    STEP_NAMES = ["Coastline", "Hills", "Height/Ocean", "Command Centers", "Resources", "Finalize"]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        self.buttons = []
        for i, name in enumerate(self.STEP_NAMES):
            btn = QPushButton(f"{i+1}. {name}")
            btn.setEnabled(False)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self.step_clicked.emit(idx))
            layout.addWidget(btn)
            self.buttons.append(btn)
        self.setLayout(layout)
        self._update_style(0, -1)

    def _update_style(self, current, completed):
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == current)
            if i <= completed:
                btn.setEnabled(True)
                btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }"
                                  "QPushButton:checked { background-color: #2196F3; color: white; }")
            elif i == completed + 1:
                btn.setEnabled(True)
                btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }"
                                  "QPushButton:checked { background-color: #2196F3; color: white; }")
            else:
                btn.setEnabled(False)
                btn.setStyleSheet("QPushButton { background-color: #9E9E9E; color: #ddd; }")
            # Current step override
            if i == current:
                btn.setChecked(True)

    def update_state(self, current_step, completed_step):
        self._update_style(current_step, completed_step)


# ==========================================================================
# ClickablePreviewWidget
# ==========================================================================

class ClickablePreviewWidget(QWidget):
    """Preview widget with mouse interaction for drawing and clicking.

    Uses paintEvent to render the map instead of a child QLabel, so all
    mouse events arrive directly at this widget — no propagation issues.
    """
    map_clicked = pyqtSignal(int, int)      # row, col in map coordinates
    map_drawn = pyqtSignal(list, int)       # list of (row, col), draw_value (1=wall, 2=gap)

    DISPLAY_SIZE = 512

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._overlay_pixmap = None
        self._stage_text = "Waiting..."
        self._map_h = 0
        self._map_w = 0
        self._click_mode = False   # Step 4/5: click to place
        self._draw_mode = False    # Step 2: draw walls
        self._drawing = False
        self._draw_points = []
        self._brush_size = 1
        self._draw_value = 1       # 1 = wall, 2 = gap
        self._active_draw_value = 1  # value used during current stroke
        self._wall_overlay_timer = QTimer()
        self._wall_overlay_timer.setInterval(50)
        self._wall_overlay_timer.setSingleShot(True)
        self._wall_overlay_timer.timeout.connect(self._rebuild_wall_overlay)
        self._pending_wall_matrix = None
        # Composed image (base + overlay) cached for painting
        self._display_pixmap = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Fixed-size canvas area that we paint ourselves
        self._canvas_size = self.DISPLAY_SIZE
        self.setMinimumSize(self._canvas_size, self._canvas_size + 30)

        self.stage_label = QLabel("Waiting...")
        self.stage_label.setAlignment(Qt.AlignCenter)
        layout.addStretch()
        layout.addWidget(self.stage_label)

        self.setLayout(layout)
        self.setMouseTracking(True)

    def set_map_size(self, h, w):
        self._map_h = h
        self._map_w = w

    def set_click_mode(self, enabled):
        self._click_mode = enabled
        self._draw_mode = False

    def set_draw_mode(self, enabled):
        self._draw_mode = enabled
        self._click_mode = False

    def set_draw_value(self, value):
        self._draw_value = value

    def set_brush_size(self, size):
        self._brush_size = size

    def update_preview(self, pixmap, stage_text=""):
        self._pixmap = pixmap
        if stage_text:
            self._stage_text = stage_text
            self.stage_label.setText(stage_text)
        self._compose_display()

    def set_wall_overlay(self, wall_matrix):
        """Render wall_matrix as semi-transparent overlay (debounced for performance)."""
        if self._pixmap is None or wall_matrix is None:
            self._overlay_pixmap = None
            self._pending_wall_matrix = None
            self._compose_display()
            return

        self._pending_wall_matrix = wall_matrix
        if not self._wall_overlay_timer.isActive():
            self._wall_overlay_timer.start()

    def _rebuild_wall_overlay(self):
        """Actually rebuild the wall overlay image from numpy array."""
        wall_matrix = self._pending_wall_matrix
        if wall_matrix is None or self._pixmap is None:
            return

        h, w = wall_matrix.shape
        px_w = self._pixmap.width()
        px_h = self._pixmap.height()

        # Detect border: cells that are filled (1 or 2) and have at least one empty (0) neighbour.
        filled = wall_matrix > 0  # 1 (wall) or 2 (gap)
        # Pad with False (out-of-bounds = empty) and check 4-connected neighbours
        padded = np.pad(filled, 1, constant_values=False)
        has_empty_neighbour = (
            ~padded[:-2, 1:-1] |  # top
            ~padded[2:, 1:-1]  |  # bottom
            ~padded[1:-1, :-2] |  # left
            ~padded[1:-1, 2:]     # right
        )
        border = filled & has_empty_neighbour

        # Build RGBA overlay showing only the border
        overlay_img = np.zeros((h, w, 4), dtype=np.uint8)
        # Red = wall border, Blue = gap border
        wall_border = border & (wall_matrix == 1)
        gap_border = border & (wall_matrix == 2)
        overlay_img[wall_border] = [200, 50, 50, 200]     # Red = wall
        overlay_img[gap_border] = [50, 150, 200, 200]      # Blue = gap
        # Show interior as very faint so user can see their painted area
        interior = filled & ~border
        overlay_img[interior] = [150, 150, 150, 40]

        overlay_img = np.ascontiguousarray(overlay_img)
        qimg = QImage(overlay_img.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        qimg = qimg.scaled(px_w, px_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)

        self._overlay_pixmap = QPixmap.fromImage(qimg)
        self._compose_display()

    def clear_overlay(self):
        self._overlay_pixmap = None
        self._pending_wall_matrix = None
        self._compose_display()

    def clear(self):
        self._pixmap = None
        self._overlay_pixmap = None
        self._display_pixmap = None
        self._pending_wall_matrix = None
        self.stage_label.setText("Waiting...")
        self.update()

    def _compose_display(self):
        if self._pixmap is None:
            self._display_pixmap = None
        elif self._overlay_pixmap is not None:
            result = self._pixmap.copy()
            painter = QPainter(result)
            painter.drawPixmap(0, 0, self._overlay_pixmap)
            painter.end()
            self._display_pixmap = result
        else:
            self._display_pixmap = self._pixmap
        self.update()  # schedule repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        # Dark background for canvas area
        painter.fillRect(0, 0, self.width(), self._canvas_size, QColor("#1a1a2e"))
        if self._display_pixmap is not None:
            # Center the pixmap in the canvas area
            px_w = self._display_pixmap.width()
            px_h = self._display_pixmap.height()
            x = (self.width() - px_w) // 2
            y = (self._canvas_size - px_h) // 2
            painter.drawPixmap(x, y, self._display_pixmap)
        painter.end()

    def _pixel_to_map(self, pos):
        """Convert widget pixel position to map row, col."""
        if self._map_h == 0 or self._map_w == 0 or self._display_pixmap is None:
            return None, None

        pm_w = self._display_pixmap.width()
        pm_h = self._display_pixmap.height()

        # Pixmap is centered in the canvas area
        offset_x = (self.width() - pm_w) // 2
        offset_y = (self._canvas_size - pm_h) // 2

        px = pos.x() - offset_x
        py = pos.y() - offset_y

        if px < 0 or py < 0 or px >= pm_w or py >= pm_h:
            return None, None

        col = int(px * self._map_w / pm_w)
        row = int(py * self._map_h / pm_h)
        col = max(0, min(col, self._map_w - 1))
        row = max(0, min(row, self._map_h - 1))
        return row, col

    def mousePressEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return super().mousePressEvent(event)

        row, col = self._pixel_to_map(event.pos())
        if row is None:
            return

        if self._click_mode and event.button() == Qt.LeftButton:
            self.map_clicked.emit(row, col)
        elif self._draw_mode:
            self._drawing = True
            # Right-click always draws gaps; left-click uses panel setting
            self._active_draw_value = 2 if event.button() == Qt.RightButton else self._draw_value
            self._draw_points = self._get_brush_points(row, col)
            self.map_drawn.emit(self._draw_points[:], self._active_draw_value)

    def mouseMoveEvent(self, event):
        if not self._drawing or not self._draw_mode:
            return

        row, col = self._pixel_to_map(event.pos())
        if row is None:
            return

        new_points = self._get_brush_points(row, col)
        self._draw_points.extend(new_points)
        self.map_drawn.emit(new_points, self._active_draw_value)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._drawing = False

    def _get_brush_points(self, row, col):
        """Get all points within brush radius."""
        points = []
        r = self._brush_size // 2
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < self._map_h and 0 <= nc < self._map_w:
                    points.append((nr, nc))
        return points


# ==========================================================================
# Step Panels
# ==========================================================================

class CoastlineStepPanel(QWidget):
    generate_clicked = pyqtSignal()
    reroll_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        # 5x5 grid
        self.button_size = 40
        self.grid_spacing = 5
        self.pattern_colors = {
            "Forest": (QColor("green"), QColor("blue")),
            "Winter": (QColor("white"), QColor("blue")),
            "Volcanic": (QColor("gray"), QColor("red")),
            "Desert": (QColor("yellow"), QColor("brown")),
            "Jungle": (QColor("green"), QColor("blue")),
        }
        self.matrix = [
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
        ]

        self.grid_widget = FixedGridWidget(5, 5, self.button_size, self.grid_spacing)
        self.buttons = []
        default_pattern = "Forest"
        color1, color2 = self.pattern_colors[default_pattern]
        for i in range(5):
            row = []
            for j in range(5):
                color = color1 if self.matrix[i][j] == 1 else color2
                button = ColorChangingButton(color, i, j)
                button.clicked.connect(self._button_clicked)
                self.grid_widget.addButton(button, i, j)
                row.append(button)
            self.buttons.append(row)

        layout.addWidget(self.grid_widget, alignment=Qt.AlignCenter)

        # Dimensions
        dims_layout = QHBoxLayout()
        dims_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(50, 640)
        self.height_spin.setValue(160)
        dims_layout.addWidget(self.height_spin)
        dims_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(50, 640)
        self.width_spin.setValue(160)
        dims_layout.addWidget(self.width_spin)
        layout.addLayout(dims_layout)

        # Mirroring
        mirror_layout = QHBoxLayout()
        mirror_layout.addWidget(QLabel("Mirroring:"))
        self.mirroring_combo = QComboBox()
        self.mirroring_combo.addItems(["none", "horizontal", "vertical", "both", "diagonal1", "diagonal2"])
        self.mirroring_combo.setCurrentText("vertical")
        self.mirroring_combo.currentTextChanged.connect(self._update_matrix)
        mirror_layout.addWidget(self.mirroring_combo)
        layout.addLayout(mirror_layout)

        # Tileset
        tileset_layout = QHBoxLayout()
        tileset_layout.addWidget(QLabel("Tileset:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(list(self.pattern_colors.keys()))
        self.pattern_combo.setCurrentText(default_pattern)
        self.pattern_combo.currentTextChanged.connect(self._update_button_colors)
        tileset_layout.addWidget(self.pattern_combo)
        layout.addLayout(tileset_layout)

        # Generate + Re-roll
        btn_layout = QHBoxLayout()
        self.gen_btn = QPushButton("Generate Coastline")
        self.gen_btn.clicked.connect(self.generate_clicked.emit)
        btn_layout.addWidget(self.gen_btn)
        self.reroll_btn = QPushButton("Re-roll")
        self.reroll_btn.clicked.connect(self.reroll_clicked.emit)
        btn_layout.addWidget(self.reroll_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self._update_matrix()

    def get_matrix(self):
        return [row[:] for row in self.matrix]

    def get_pattern_index(self):
        return list(self.pattern_colors.keys()).index(self.pattern_combo.currentText()) + 1

    def _button_clicked(self):
        sender = self.sender()
        r, c = sender.row, sender.col
        self.matrix[r][c] = 1 - self.matrix[r][c]
        self._update_mirrored(r, c)
        self._update_button_colors()

    def _update_mirrored(self, row, col):
        mirroring = self.mirroring_combo.currentText()
        val = self.matrix[row][col]
        if mirroring == "horizontal":
            self.matrix[4 - row][col] = val
        elif mirroring == "vertical":
            self.matrix[row][4 - col] = val
        elif mirroring == "both":
            self.matrix[4 - row][col] = val
            self.matrix[row][4 - col] = val
            self.matrix[4 - row][4 - col] = val
        elif mirroring == "diagonal1":
            self.matrix[col][row] = val
        elif mirroring == "diagonal2":
            self.matrix[4 - col][4 - row] = val

    def _update_matrix(self):
        for r in range(5):
            for c in range(5):
                self._update_mirrored(r, c)
        self._update_button_colors()

    def _update_button_colors(self):
        pattern = self.pattern_combo.currentText()
        c1, c2 = self.pattern_colors[pattern]
        for i, row in enumerate(self.buttons):
            for j, btn in enumerate(row):
                btn.setColor(c1 if self.matrix[i][j] == 1 else c2)

    def set_generating(self, generating):
        self.gen_btn.setEnabled(not generating)
        self.reroll_btn.setEnabled(not generating)


class HillDrawingStepPanel(QWidget):
    clear_walls_clicked = pyqtSignal()
    brush_size_changed = pyqtSignal(int)
    draw_value_changed = pyqtSignal(int)  # 1 = wall, 2 = gap

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Draw wall lines on the preview. They will be auto-mirrored."))
        layout.addWidget(QLabel("Use Gap brush (or right-click) to mark passages through walls."))

        # Draw mode toggle
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Brush:"))
        self.wall_btn = QPushButton("Wall")
        self.gap_btn = QPushButton("Gap")
        self.wall_btn.setCheckable(True)
        self.gap_btn.setCheckable(True)
        self.wall_btn.setChecked(True)
        self.wall_btn.clicked.connect(lambda: self._set_draw_value(1))
        self.gap_btn.clicked.connect(lambda: self._set_draw_value(2))
        mode_layout.addWidget(self.wall_btn)
        mode_layout.addWidget(self.gap_btn)
        layout.addLayout(mode_layout)

        # Brush size
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Brush size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 9)
        self.brush_slider.setValue(3)
        self.brush_slider.setSingleStep(2)
        self.brush_slider.setTickInterval(2)
        self.brush_slider.valueChanged.connect(lambda v: self.brush_size_changed.emit(v))
        brush_layout.addWidget(self.brush_slider)
        self.brush_label = QLabel("3")
        self.brush_slider.valueChanged.connect(lambda v: self.brush_label.setText(str(v)))
        brush_layout.addWidget(self.brush_label)
        layout.addLayout(brush_layout)

        self.clear_btn = QPushButton("Clear Walls")
        self.clear_btn.clicked.connect(self.clear_walls_clicked.emit)
        layout.addWidget(self.clear_btn)

        layout.addStretch()
        self.setLayout(layout)

    def _set_draw_value(self, value):
        self.wall_btn.setChecked(value == 1)
        self.gap_btn.setChecked(value == 2)
        self.draw_value_changed.emit(value)


class HeightOceanStepPanel(QWidget):
    generate_clicked = pyqtSignal()
    reroll_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Height and ocean levels (visual terrain variation, no gameplay impact)."))

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Height levels (1-7):"))
        self.height_levels_spin = QSpinBox()
        self.height_levels_spin.setRange(1, 7)
        self.height_levels_spin.setValue(7)
        h_layout.addWidget(self.height_levels_spin)
        layout.addLayout(h_layout)

        o_layout = QHBoxLayout()
        o_layout.addWidget(QLabel("Ocean levels (1-3):"))
        self.ocean_levels_spin = QSpinBox()
        self.ocean_levels_spin.setRange(1, 3)
        self.ocean_levels_spin.setValue(3)
        o_layout.addWidget(self.ocean_levels_spin)
        layout.addLayout(o_layout)

        btn_layout = QHBoxLayout()
        self.gen_btn = QPushButton("Generate Height")
        self.gen_btn.clicked.connect(self.generate_clicked.emit)
        btn_layout.addWidget(self.gen_btn)
        self.reroll_btn = QPushButton("Re-roll")
        self.reroll_btn.clicked.connect(self.reroll_clicked.emit)
        btn_layout.addWidget(self.reroll_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.setLayout(layout)

    def set_generating(self, generating):
        self.gen_btn.setEnabled(not generating)
        self.reroll_btn.setEnabled(not generating)


class CommandCenterStepPanel(QWidget):
    place_random_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Place command centers. Click the preview to place manually, or use random."))

        # Mode toggle
        mode_layout = QHBoxLayout()
        self.manual_radio = QRadioButton("Manual (click preview)")
        self.random_radio = QRadioButton("Random")
        self.manual_radio.setChecked(True)
        mode_layout.addWidget(self.manual_radio)
        mode_layout.addWidget(self.random_radio)
        layout.addLayout(mode_layout)

        # Num centers
        n_layout = QHBoxLayout()
        n_layout.addWidget(QLabel("Num players (even, 0-10):"))
        self.num_centers_spin = QSpinBox()
        self.num_centers_spin.setRange(0, 10)
        self.num_centers_spin.setSingleStep(2)
        self.num_centers_spin.setValue(4)
        n_layout.addWidget(self.num_centers_spin)
        layout.addLayout(n_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.place_btn = QPushButton("Place Random")
        self.place_btn.clicked.connect(self.place_random_clicked.emit)
        btn_layout.addWidget(self.place_btn)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_clicked.emit)
        btn_layout.addWidget(self.undo_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        self.status_label = QLabel("No command centers placed.")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def is_manual(self):
        return self.manual_radio.isChecked()


class ResourceStepPanel(QWidget):
    place_random_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Place resource pools. Click the preview to place manually, or use random."))

        mode_layout = QHBoxLayout()
        self.manual_radio = QRadioButton("Manual (click preview)")
        self.random_radio = QRadioButton("Random")
        self.manual_radio.setChecked(True)
        mode_layout.addWidget(self.manual_radio)
        mode_layout.addWidget(self.random_radio)
        layout.addLayout(mode_layout)

        n_layout = QHBoxLayout()
        n_layout.addWidget(QLabel("Num resource pulls (0-50):"))
        self.num_resources_spin = QSpinBox()
        self.num_resources_spin.setRange(0, 50)
        self.num_resources_spin.setValue(12)
        n_layout.addWidget(self.num_resources_spin)
        layout.addLayout(n_layout)

        btn_layout = QHBoxLayout()
        self.place_btn = QPushButton("Place Random")
        self.place_btn.clicked.connect(self.place_random_clicked.emit)
        btn_layout.addWidget(self.place_btn)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_clicked.emit)
        btn_layout.addWidget(self.undo_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        self.status_label = QLabel("No resources placed.")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def is_manual(self):
        return self.manual_radio.isChecked()


class FinalizeStepPanel(QWidget):
    export_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Finalize terrain smoothing, decoration, and export to TMX."))

        # Output path
        self.default_output_path = os.path.dirname(sys.executable)
        self.output_label = QLabel(f"Output: {self.default_output_path}")
        layout.addWidget(self.output_label)

        path_btn = QPushButton("Select output directory")
        path_btn.clicked.connect(self._select_dir)
        layout.addWidget(path_btn)

        self.export_btn = QPushButton("Finalize && Export")
        self.export_btn.clicked.connect(self.export_clicked.emit)
        layout.addWidget(self.export_btn)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def _select_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.default_output_path)
        if d:
            self.default_output_path = d
            self.output_label.setText(f"Output: {d}")

    def set_exporting(self, exporting):
        self.export_btn.setEnabled(not exporting)


# ==========================================================================
# WizardGUI — main orchestrator
# ==========================================================================

class WizardGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = WizardState()
        self._worker = None
        self._extractor_cache = {}
        self._renderer = None

        self._preview_queue = deque()
        self._preview_timer = QTimer()
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._flush_preview)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()

        # Step bar
        self.step_bar = StepBar()
        self.step_bar.step_clicked.connect(self._go_to_step)
        main_layout.addWidget(self.step_bar)

        # Content area: preview on left, controls on right
        content_layout = QHBoxLayout()

        self.preview = ClickablePreviewWidget()
        self.preview.map_clicked.connect(self._on_map_clicked)
        self.preview.map_drawn.connect(self._on_map_drawn)
        content_layout.addWidget(self.preview, stretch=2)

        # Step panels stack
        self.panel_stack = QStackedWidget()

        self.coastline_panel = CoastlineStepPanel()
        self.coastline_panel.generate_clicked.connect(self._run_coastline)
        self.coastline_panel.reroll_clicked.connect(self._run_coastline)
        self.panel_stack.addWidget(self.coastline_panel)

        self.hill_panel = HillDrawingStepPanel()
        self.hill_panel.clear_walls_clicked.connect(self._clear_walls)
        self.hill_panel.brush_size_changed.connect(self.preview.set_brush_size)
        self.hill_panel.draw_value_changed.connect(self.preview.set_draw_value)
        self.panel_stack.addWidget(self.hill_panel)

        self.height_panel = HeightOceanStepPanel()
        self.height_panel.generate_clicked.connect(self._run_height_ocean)
        self.height_panel.reroll_clicked.connect(self._run_height_ocean)
        self.panel_stack.addWidget(self.height_panel)

        self.cc_panel = CommandCenterStepPanel()
        self.cc_panel.place_random_clicked.connect(self._place_cc_random)
        self.cc_panel.undo_clicked.connect(self._undo_cc)
        self.cc_panel.clear_clicked.connect(self._clear_cc)
        self.panel_stack.addWidget(self.cc_panel)

        self.resource_panel = ResourceStepPanel()
        self.resource_panel.place_random_clicked.connect(self._place_resources_random)
        self.resource_panel.undo_clicked.connect(self._undo_resource)
        self.resource_panel.clear_clicked.connect(self._clear_resources)
        self.panel_stack.addWidget(self.resource_panel)

        self.finalize_panel = FinalizeStepPanel()
        self.finalize_panel.export_clicked.connect(self._run_finalize)
        self.panel_stack.addWidget(self.finalize_panel)

        content_layout.addWidget(self.panel_stack, stretch=1)
        main_layout.addLayout(content_layout)

        # Navigation
        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("< Back")
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)
        main_layout.addLayout(nav_layout)

        self.setLayout(main_layout)
        self._go_to_step(0)

    # --- Navigation ---

    def _go_to_step(self, step_idx):
        step = WizardStep(step_idx)
        self.state.current_step = step
        self.panel_stack.setCurrentIndex(step_idx)
        self.step_bar.update_state(step_idx, self.state.completed_step)

        self.back_btn.setEnabled(step_idx > 0)
        # Can advance if current step is completed, or step is optional (hills, CC, resources)
        optional_steps = {WizardStep.HILLS, WizardStep.COMMAND_CENTERS, WizardStep.RESOURCES}
        can_advance = (step_idx <= self.state.completed_step) or (step in optional_steps)
        self.next_btn.setEnabled(can_advance and step_idx < 5)

        # Set interaction mode
        self.preview.set_click_mode(False)
        self.preview.set_draw_mode(False)

        if step == WizardStep.HILLS:
            self.preview.set_draw_mode(True)
            self.preview.set_brush_size(self.hill_panel.brush_slider.value())
            self._refresh_coastline_preview()
        elif step == WizardStep.COMMAND_CENTERS:
            if self.cc_panel.is_manual():
                self.preview.set_click_mode(True)
            self._refresh_height_preview()
        elif step == WizardStep.RESOURCES:
            if self.resource_panel.is_manual():
                self.preview.set_click_mode(True)
            self._refresh_height_preview()
        elif step == WizardStep.COASTLINE:
            if self.state.coastline_height_map is not None:
                self._refresh_coastline_preview()
        elif step == WizardStep.HEIGHT_OCEAN:
            if self.state.height_map is not None:
                self._refresh_height_preview()
        elif step == WizardStep.FINALIZE:
            if self.state.height_map is not None:
                self._refresh_height_preview()

    def _go_back(self):
        cur = int(self.state.current_step)
        if cur > 0:
            self._go_to_step(cur - 1)

    def _go_next(self):
        cur = int(self.state.current_step)
        step = WizardStep(cur)

        # Mark hills as completed if advancing from it (optional step)
        if step == WizardStep.HILLS and self.state.completed_step < int(WizardStep.HILLS):
            if self.state.wall_matrix is None:
                self.state.wall_matrix = np.zeros((self.state.height, self.state.width), dtype=int)
            self.state.completed_step = max(self.state.completed_step, int(WizardStep.HILLS))

        # Mark CC/resources completed when advancing
        if step == WizardStep.COMMAND_CENTERS:
            if self.state.units_matrix is None:
                shape = self.state.height_map.shape if self.state.height_map is not None else (self.state.height, self.state.width)
                self.state.units_matrix = np.zeros(shape, dtype=int)
            self.state.completed_step = max(self.state.completed_step, int(WizardStep.COMMAND_CENTERS))
        if step == WizardStep.RESOURCES:
            if self.state.items_matrix is None:
                shape = self.state.height_map.shape if self.state.height_map is not None else (self.state.height, self.state.width)
                self.state.items_matrix = np.zeros(shape, dtype=int)
            self.state.completed_step = max(self.state.completed_step, int(WizardStep.RESOURCES))

        if cur < 5:
            self._go_to_step(cur + 1)

    # --- Renderer helpers ---

    def _get_renderer(self):
        pattern = self.coastline_panel.get_pattern_index()
        if pattern not in self._extractor_cache:
            tmx_path = resource_path(f"generator_blueprint{pattern}.tmx")
            self._extractor_cache[pattern] = TilesetExtractor(tmx_path)
        extractor = self._extractor_cache[pattern]
        self._renderer = MapRenderer(extractor)
        return self._renderer

    def _render_height_map(self, height_map):
        renderer = self._get_renderer()
        qimg = renderer.render_height_map(height_map)
        return QPixmap.fromImage(qimg)

    def _render_terrain(self, id_matrix, items_matrix, units_matrix):
        renderer = self._get_renderer()
        qimg = renderer.render_terrain(id_matrix, items_matrix, units_matrix)
        return QPixmap.fromImage(qimg)

    def _refresh_coastline_preview(self):
        if self.state.coastline_height_map is not None:
            pixmap = self._render_height_map(self.state.coastline_height_map)
            self.preview.set_map_size(self.state.height, self.state.width)
            self.preview.update_preview(pixmap, "Coastline")
            if self.state.wall_matrix is not None and np.any(self.state.wall_matrix):
                self.preview.set_wall_overlay(self.state.wall_matrix)
            else:
                self.preview.clear_overlay()

    def _refresh_height_preview(self):
        if self.state.height_map is not None:
            pixmap = self._render_height_map(self.state.height_map)
            self.preview.set_map_size(self.state.height, self.state.width)
            self.preview.update_preview(pixmap, "Height Map")
            # Cancel any pending wall-only timer so it doesn't overwrite
            self.preview._wall_overlay_timer.stop()
            self.preview._pending_wall_matrix = None
            # Build combined overlay (walls + entities) in one pass
            self._build_combined_overlay()

    def _build_combined_overlay(self):
        """Build a single overlay with walls + CCs + resources (no timer conflicts)."""
        if self.preview._pixmap is None:
            return

        h, w = self.state.height, self.state.width
        has_walls = self.state.wall_matrix is not None and np.any(self.state.wall_matrix)
        has_entities = bool(self.state.cc_positions or self.state.resource_positions)

        if not has_walls and not has_entities:
            self.preview.clear_overlay()
            return

        px_w = self.preview._pixmap.width()
        px_h = self.preview._pixmap.height()

        overlay_img = np.zeros((h, w, 4), dtype=np.uint8)

        # Walls
        if has_walls:
            mask = self.state.wall_matrix == 1
            overlay_img[mask] = [200, 50, 50, 140]

        # CC markers (3x3 blue blocks)
        if self.state.cc_positions:
            for r, c in self.state.cc_positions:
                r_min, r_max = max(0, r - 1), min(h, r + 2)
                c_min, c_max = max(0, c - 1), min(w, c + 2)
                overlay_img[r_min:r_max, c_min:c_max] = [50, 100, 255, 220]

        # Resource markers (3x3 yellow blocks)
        if self.state.resource_positions:
            for r, c in self.state.resource_positions:
                r_min, r_max = max(0, r - 1), min(h, r + 2)
                c_min, c_max = max(0, c - 1), min(w, c + 2)
                overlay_img[r_min:r_max, c_min:c_max] = [255, 220, 50, 220]

        overlay_img = np.ascontiguousarray(overlay_img)
        qimg = QImage(overlay_img.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        qimg = qimg.scaled(px_w, px_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)

        self.preview._overlay_pixmap = QPixmap.fromImage(qimg)
        self.preview._compose_display()

    # --- Worker management ---

    def _cancel_worker(self):
        """Stop and clean up any running worker thread."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.preview_signal.disconnect()
            self._worker.finished_signal.disconnect()
            self._worker.quit()
            self._worker.wait(3000)
            if self._worker.isRunning():
                self._worker.terminate()
                self._worker.wait(1000)
        self._worker = None
        self._preview_queue.clear()
        self._preview_timer.stop()

    # --- Step 1: Coastline ---

    def _run_coastline(self):
        self._cancel_worker()

        self.state.invalidate_from(WizardStep.COASTLINE)
        panel = self.coastline_panel
        self.state.initial_matrix = panel.get_matrix()
        self.state.height = panel.height_spin.value()
        self.state.width = panel.width_spin.value()
        self.state.mirroring = panel.mirroring_combo.currentText()
        self.state.pattern = panel.get_pattern_index()

        panel.set_generating(True)
        self.preview.clear()
        self.preview.set_map_size(self.state.height, self.state.width)

        self._worker = CoastlineWorker(self.state)
        self._worker.preview_signal.connect(self._on_coastline_preview)
        self._worker.finished_signal.connect(self._on_coastline_done)
        self._worker.start()

    def _on_coastline_preview(self, stage, height_map):
        self._preview_queue.append(("coastline", stage, height_map))
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _on_coastline_done(self):
        self.coastline_panel.set_generating(False)
        self._worker = None
        self._flush_all_previews()
        self._refresh_coastline_preview()
        self.step_bar.update_state(int(self.state.current_step), self.state.completed_step)
        self.next_btn.setEnabled(True)

    # --- Step 2: Hill Drawing ---

    def _on_map_drawn(self, points, draw_value=1):
        """Handle wall/gap drawing on preview. draw_value: 1=wall, 2=gap."""
        if self.state.current_step != WizardStep.HILLS:
            return
        if self.state.wall_matrix is None:
            self.state.wall_matrix = np.zeros((self.state.height, self.state.width), dtype=int)

        mirroring = self.state.mirroring
        h, w = self.state.height, self.state.width

        for r, c in points:
            mirrored = _get_mirrored_positions(r, c, h, w, mirroring)
            for mr, mc in mirrored:
                if 0 <= mr < h and 0 <= mc < w:
                    if draw_value == 2 and self.state.wall_matrix[mr, mc] == 0:
                        continue  # gap brush only affects filled cells
                    self.state.wall_matrix[mr, mc] = draw_value

        self.preview.set_wall_overlay(self.state.wall_matrix)

    def _clear_walls(self):
        self.state.wall_matrix = np.zeros((self.state.height, self.state.width), dtype=int)
        self.preview.clear_overlay()
        self._refresh_coastline_preview()

    # --- Step 3: Height/Ocean ---

    def _run_height_ocean(self):
        self._cancel_worker()

        self.state.invalidate_from(WizardStep.HEIGHT_OCEAN)
        self.state.num_height_levels = self.height_panel.height_levels_spin.value()
        self.state.num_ocean_levels = self.height_panel.ocean_levels_spin.value()

        self.height_panel.set_generating(True)
        self.preview.clear()
        self.preview.set_map_size(self.state.height, self.state.width)

        self._worker = HeightOceanWorker(self.state)
        self._worker.preview_signal.connect(self._on_height_preview)
        self._worker.finished_signal.connect(self._on_height_done)
        self._worker.start()

    def _on_height_preview(self, stage, height_map):
        self._preview_queue.append(("height", stage, height_map))
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _on_height_done(self):
        self.height_panel.set_generating(False)
        self._worker = None
        self._flush_all_previews()
        self._refresh_height_preview()
        self.step_bar.update_state(int(self.state.current_step), self.state.completed_step)
        self.next_btn.setEnabled(True)

    # --- Step 4: Command Centers ---

    def _place_cc_random(self):
        self.state.num_command_centers = self.cc_panel.num_centers_spin.value()
        # Clear existing
        clear_all_cc(self.state)
        try:
            run_place_cc_random(self.state)
            self.cc_panel.status_label.setText(f"Placed {len(self.state.cc_positions)} command centers.")
        except ValueError as e:
            self.cc_panel.status_label.setText(f"Error: {e}")
        self._refresh_height_preview()

    def _on_map_clicked(self, row, col):
        step = self.state.current_step
        if step == WizardStep.COMMAND_CENTERS and self.cc_panel.is_manual():
            placed = run_place_cc_manual(self.state, row, col)
            if placed:
                self.cc_panel.status_label.setText(f"{len(self.state.cc_positions)} CC positions placed.")
            else:
                self.cc_panel.status_label.setText("Invalid position (water, wall, or out of bounds).")
            self._refresh_height_preview()
        elif step == WizardStep.RESOURCES and self.resource_panel.is_manual():
            placed = run_place_resource_manual(self.state, row, col)
            if placed:
                self.resource_panel.status_label.setText(f"{len(self.state.resource_positions)} resource positions placed.")
            else:
                self.resource_panel.status_label.setText("Invalid position.")
            self._refresh_height_preview()

    def _undo_cc(self):
        undo_last_cc(self.state)
        self.cc_panel.status_label.setText(f"{len(self.state.cc_positions)} CC positions remain.")
        self._refresh_height_preview()

    def _clear_cc(self):
        clear_all_cc(self.state)
        self.cc_panel.status_label.setText("Cleared all command centers.")
        self._refresh_height_preview()

    # --- Step 5: Resources ---

    def _place_resources_random(self):
        self.state.num_resource_pulls = self.resource_panel.num_resources_spin.value()
        clear_all_resources(self.state)
        try:
            run_place_resources_random(self.state)
            self.resource_panel.status_label.setText(f"Placed {len(self.state.resource_positions)} resource positions.")
        except Exception as e:
            self.resource_panel.status_label.setText(f"Error: {e}")
        self._refresh_height_preview()

    def _undo_resource(self):
        undo_last_resource(self.state)
        self.resource_panel.status_label.setText(f"{len(self.state.resource_positions)} resource positions remain.")
        self._refresh_height_preview()

    def _clear_resources(self):
        clear_all_resources(self.state)
        self.resource_panel.status_label.setText("Cleared all resources.")
        self._refresh_height_preview()

    # --- Step 6: Finalize ---

    def _run_finalize(self):
        self._cancel_worker()

        self.state.output_path = self.finalize_panel.default_output_path

        self.finalize_panel.set_exporting(True)
        self.finalize_panel.status_label.setText("Finalizing...")
        self.preview.clear()

        self._worker = FinalizeWorker(self.state)
        self._worker.preview_signal.connect(self._on_finalize_preview)
        self._worker.finished_signal.connect(self._on_finalize_done)
        self._worker.start()

    def _on_finalize_preview(self, stage, height_map, id_matrix, items_matrix, units_matrix):
        self._preview_queue.append(("finalize", stage, height_map, id_matrix, items_matrix, units_matrix))
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _on_finalize_done(self, result):
        self.finalize_panel.set_exporting(False)
        self._worker = None
        self._flush_all_previews()
        if result.startswith("ERROR"):
            self.finalize_panel.status_label.setText(result)
        else:
            self.finalize_panel.status_label.setText(f"Exported: {result}")
        self.step_bar.update_state(int(self.state.current_step), self.state.completed_step)

    # --- Preview queue ---

    def _flush_preview(self):
        if not self._preview_queue:
            self._preview_timer.stop()
            return

        entry = self._preview_queue.popleft()
        kind = entry[0]

        if kind in ("coastline", "height"):
            stage, height_map = entry[1], entry[2]
            pixmap = self._render_height_map(height_map)
            self.preview.set_map_size(height_map.shape[0], height_map.shape[1])
            self.preview.update_preview(pixmap, stage.replace("_", " ").title())
        elif kind == "finalize":
            stage = entry[1]
            height_map, id_matrix = entry[2], entry[3]
            items_matrix, units_matrix = entry[4], entry[5]
            if id_matrix is not None:
                pixmap = self._render_terrain(id_matrix, items_matrix, units_matrix)
            else:
                pixmap = self._render_height_map(height_map)
            self.preview.update_preview(pixmap, stage.replace("_", " ").title())

    def _flush_all_previews(self):
        while self._preview_queue:
            self._flush_preview()
        self._preview_timer.stop()


# ==========================================================================
# MainWindow — mode toggle between Wizard and Quick Generate
# ==========================================================================

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rusted Warfare Map Generator")

        layout = QVBoxLayout()

        # Mode toggle
        toggle_layout = QHBoxLayout()
        self.wizard_btn = QPushButton("Wizard Mode")
        self.wizard_btn.setCheckable(True)
        self.wizard_btn.setChecked(True)
        self.wizard_btn.clicked.connect(lambda: self._switch_mode(0))
        toggle_layout.addWidget(self.wizard_btn)

        self.quick_btn = QPushButton("Quick Generate")
        self.quick_btn.setCheckable(True)
        self.quick_btn.clicked.connect(lambda: self._switch_mode(1))
        toggle_layout.addWidget(self.quick_btn)
        layout.addLayout(toggle_layout)

        # Stacked modes
        self.mode_stack = QStackedWidget()

        self.wizard_gui = WizardGUI()
        self.mode_stack.addWidget(self.wizard_gui)

        # Lazy-load quick generate to avoid circular import issues
        from generator_gui import MapGeneratorGUI
        self.quick_gui = MapGeneratorGUI()
        self.mode_stack.addWidget(self.quick_gui)

        layout.addWidget(self.mode_stack)
        self.setLayout(layout)
        self._switch_mode(0)

    def _switch_mode(self, idx):
        self.mode_stack.setCurrentIndex(idx)
        self.wizard_btn.setChecked(idx == 0)
        self.quick_btn.setChecked(idx == 1)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
