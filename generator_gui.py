import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QSpinBox, QDialog, QTextEdit, QFileDialog, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize
from PyQt5.QtGui import QColor
from generator_main import generate_map


from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt

class FixedGridWidget(QWidget):
    def __init__(self, rows, cols, button_size, spacing):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.button_size = button_size
        self.spacing = spacing
        self.buttons = []

    def addButton(self, button, row, col):
        button.setParent(self)
        button.setFixedSize(self.button_size, self.button_size)
        self.buttons.append((button, row, col))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateButtonPositions()

    def updateButtonPositions(self):
        for button, row, col in self.buttons:
            x = col * (self.button_size + self.spacing)
            y = row * (self.button_size + self.spacing)
            button.move(x, y)

    def sizeHint(self):
        width = self.cols * (self.button_size + self.spacing) - self.spacing
        height = self.rows * (self.button_size + self.spacing) - self.spacing
        return QSize(width, height)

    def minimumSizeHint(self):
        return self.sizeHint()


class MapGeneratorWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, generate_map_func, **kwargs):
        super().__init__()
        self.generate_map_func = generate_map_func
        self.kwargs = kwargs

    def run(self):
        # Redirect print statements to progress_signal
        def custom_print(*args, **kwargs):
            self.progress_signal.emit(' '.join(map(str, args)))

        import builtins
        builtins.print = custom_print

        try:
            self.generate_map_func(**self.kwargs)
        finally:
            # Restore original print function
            builtins.print = print
            self.finished_signal.emit()


class ProgressWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map Generation Progress")
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        self.setModal(True)

    def append_text(self, text):
        self.text_edit.append(text)


class ColorChangingButton(QPushButton):
    def __init__(self, color, row, col):
        super().__init__()
        self.setFixedSize(40, 40)
        self.row = row
        self.col = col
        self.setColor(color)

    def setColor(self, color):
        self.color = color
        self.setStyleSheet(f"background-color: {color.name()};")

    def toggleColor(self, color1, color2):
        self.setColor(color1 if self.color == color2 else color2)

class MapGeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.button_size = 40
        self.grid_spacing = 5
        self.pattern_colors = {
            "Forest": (QColor("green"), QColor("blue")),
            "Winter": (QColor("white"), QColor("blue")),
            "Volcanic": (QColor("gray"), QColor("red")),
            "Desert": (QColor("yellow"), QColor("brown")),
            "Jungle": (QColor("green"), QColor("blue"))
        }
        self.matrix = [[1, 0, 1, 0, 1],
                       [1, 1, 0, 1, 1],
                       [1, 0, 1, 0, 1],
                       [1, 1, 0, 1, 1],
                       [1, 0, 1, 0, 1]]
        self.default_output_path = os.path.dirname(os.path.abspath(__file__))
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Создаем FixedGridWidget вместо QGridLayout
        self.grid_widget = FixedGridWidget(5, 5, self.button_size, self.grid_spacing)
        self.buttons = []
        default_pattern = "Forest"
        color1, color2 = self.pattern_colors[default_pattern]
        for i in range(5):
            row = []
            for j in range(5):
                color = color1 if self.matrix[i][j] == 1 else color2
                button = ColorChangingButton(color, i, j)
                button.clicked.connect(self.button_clicked)
                self.grid_widget.addButton(button, i, j)
                row.append(button)
            self.buttons.append(row)

        # Создаем контейнер для центрирования сетки
        grid_container = QWidget()
        grid_container_layout = QVBoxLayout()
        grid_container_layout.addWidget(self.grid_widget, alignment=Qt.AlignCenter)
        grid_container_layout.setContentsMargins(0, 0, 0, 0)
        grid_container.setLayout(grid_container_layout)

        layout.addWidget(grid_container)

        self.output_path_label = QLabel(f"Output path: {self.default_output_path}")
        layout.addWidget(self.output_path_label)

        select_path_button = QPushButton("Select output directory")
        select_path_button.clicked.connect(self.select_output_directory)
        layout.addWidget(select_path_button)

        height = self.create_spinbox(50, 640, 160)
        layout.addLayout(self.create_input("Height (50-640):", "height", height))

        width = self.create_spinbox(50, 640, 160)
        layout.addLayout(self.create_input("Width (50-640):", "width", width))

        self.mirroring_combo = QComboBox()
        self.mirroring_combo.addItems(["none", "horizontal", "vertical", "both", "diagonal1", "diagonal2"])
        self.mirroring_combo.setCurrentText("vertical")
        self.mirroring_combo.currentTextChanged.connect(self.update_matrix)
        layout.addLayout(self.create_input("Mirroring:", "mirroring", self.mirroring_combo))

        num_res_pulls = self.create_spinbox(0, 50, 12)
        layout.addLayout(self.create_input("Num of res_pulls (0-50):", "num_res_pulls", num_res_pulls))

        num_com_centers = self.create_spinbox(0, 10, 4, 2)
        layout.addLayout(self.create_input("Num of players (even, 0-10):", "num_com_centers", num_com_centers))

        num_height_levels = self.create_spinbox(1, 7, 7)
        layout.addLayout(self.create_input("Num of height levels (1-7):", "num_height_levels", num_height_levels))

        num_ocean_levels = self.create_spinbox(1, 3, 3)
        layout.addLayout(self.create_input("Num of ocean levels (1-3):", "num_ocean_levels", num_ocean_levels))

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(list(self.pattern_colors.keys()))
        self.pattern_combo.setCurrentText(default_pattern)
        self.pattern_combo.currentTextChanged.connect(self.update_colors)
        layout.addLayout(self.create_input("Tileset:", "pattern", self.pattern_combo))

        generate_button = QPushButton("Generate map")
        generate_button.clicked.connect(self.generate_map)
        layout.addWidget(generate_button)

        self.setLayout(layout)
        self.setWindowTitle('Map generator')
        self.show()

        # Применяем начальное отражение
        self.update_matrix()

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.default_output_path)
        if directory:
            self.default_output_path = directory
            self.output_path_label.setText(f"Output path: {self.default_output_path}")

    def create_spinbox(self, min_value, max_value, default_value, step=1):
        spinbox = QSpinBox()
        spinbox.setMinimum(min_value)
        spinbox.setMaximum(max_value)
        spinbox.setSingleStep(step)
        spinbox.setValue(default_value)
        return spinbox

    def create_input(self, label, name, widget):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        widget.setObjectName(name)
        layout.addWidget(widget)
        return layout

    def button_clicked(self):
        sender = self.sender()
        row, col = sender.row, sender.col
        self.matrix[row][col] = 1 - self.matrix[row][col]  # Toggle between 0 and 1
        self.update_button_colors()
        self.update_mirrored_buttons(row, col)

    def update_mirrored_buttons(self, row, col):
        mirroring = self.mirroring_combo.currentText()
        if mirroring == "horizontal":
            mirrored_row = 4 - row
            self.matrix[mirrored_row][col] = self.matrix[row][col]
        elif mirroring == "vertical":
            mirrored_col = 4 - col
            self.matrix[row][mirrored_col] = self.matrix[row][col]
        elif mirroring == "both":
            mirrored_row = 4 - row
            mirrored_col = 4 - col
            self.matrix[mirrored_row][col] = self.matrix[row][col]
            self.matrix[row][mirrored_col] = self.matrix[row][col]
            self.matrix[mirrored_row][mirrored_col] = self.matrix[row][col]
        elif mirroring == "diagonal1":
            self.matrix[col][row] = self.matrix[row][col]
        elif mirroring == "diagonal2":
            mirrored_row = 4 - col
            mirrored_col = 4 - row
            self.matrix[mirrored_row][mirrored_col] = self.matrix[row][col]

        self.update_button_colors()

    def update_matrix(self):
        for row in range(5):
            for col in range(5):
                self.update_mirrored_buttons(row, col)

    def update_colors(self):
        self.update_button_colors()
        self.update_matrix()

    def update_button_colors(self):
        current_pattern = self.pattern_combo.currentText()
        color1, color2 = self.pattern_colors[current_pattern]
        for i, row in enumerate(self.buttons):
            for j, button in enumerate(row):
                button.setColor(color1 if self.matrix[i][j] == 1 else color2)

    def get_matrix(self):
        return self.matrix

    def generate_map(self):
        initial_matrix = self.get_matrix()
        height = self.findChild(QSpinBox, "height").value()
        width = self.findChild(QSpinBox, "width").value()
        mirroring = self.mirroring_combo.currentText()
        num_res_pulls = self.findChild(QSpinBox, "num_res_pulls").value()
        num_com_centers = self.findChild(QSpinBox, "num_com_centers").value()
        num_height_levels = self.findChild(QSpinBox, "num_height_levels").value()
        num_ocean_levels = self.findChild(QSpinBox, "num_ocean_levels").value()
        pattern = list(self.pattern_colors.keys()).index(self.pattern_combo.currentText()) + 1

        output_path = self.default_output_path

        # Create and show progress window
        self.progress_window = ProgressWindow(self)
        self.progress_window.show()

        # Create and start worker thread
        self.worker = MapGeneratorWorker(
            generate_map,
            initial_matrix=initial_matrix,
            height=height, width=width,
            mirroring=mirroring,
            num_res_pulls=num_res_pulls,
            num_com_centers=num_com_centers,
            num_height_levels=num_height_levels,
            num_ocean_levels=num_ocean_levels,
            pattern=pattern,
            output_path=output_path
        )
        self.worker.progress_signal.connect(self.progress_window.append_text)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.start()

    def on_generation_finished(self):
        self.progress_window.append_text("Map generation completed. This window will close in 3 seconds.")
        QTimer.singleShot(3000, self.close_progress_window)

    def close_progress_window(self):
        if self.progress_window:
            self.progress_window.close()
            self.progress_window = None
        self.worker = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MapGeneratorGUI()
    sys.exit(app.exec_())
