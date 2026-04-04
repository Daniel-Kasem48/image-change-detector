"""
Desktop GUI for video-to-video comparison (PySide6).

Easy install:
    ./venv/bin/pip install -r requirements-desktop.txt

Run:
    ./venv/bin/python video_compare_desktop.py
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

LOG_FILE = Path.home() / "VideoChangeDetector.log"


def _show_fatal_error(message: str) -> None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "Video Change Detector Error", 0x10)
            return
        except Exception:
            pass
    print(message)


def _resource_path(filename: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / filename
    return Path(__file__).parent / filename


def _resolve_default_config() -> Path:
    # Prefer the desktop-tuned config first for easier out-of-box behavior.
    candidates = [
        _resource_path("config.desktop.yaml"),
        _resource_path("config.yaml"),
        Path("config.desktop.yaml"),
        Path("config.yaml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("config.desktop.yaml")


try:
    from PySide6.QtCore import QObject, QThread, Signal
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from PySide6.QtCore import QUrl
except ModuleNotFoundError:
    _show_fatal_error(
        "PySide6 is not installed.\n\n"
        "Install desktop dependencies with:\n"
        "./venv/bin/pip install -r requirements-desktop.txt"
    )
    raise SystemExit(1)

sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.config import load_config
    from src.video_compare import compare_videos
except Exception as exc:  # noqa: BLE001
    _show_fatal_error(f"Failed to load application modules:\n{exc}")
    raise SystemExit(1)


class CompareWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            cfg = load_config(self.params["config"])
            Path(self.params["output_dir"]).mkdir(parents=True, exist_ok=True)
            result = compare_videos(
                self.params["video_a"],
                self.params["video_b"],
                app_config=cfg,
                method=self.params["mode"],
                alignment_method=self.params["alignment_method"],
                min_alignment_score=float(self.params["min_alignment_score"]),
                gap_penalty=float(self.params["gap_penalty"]),
                min_detector_alignment_conf=float(self.params["min_detector_alignment_conf"]),
                stabilize=bool(self.params["stabilize"]),
                illumination_mode=self.params["illumination_mode"],
                sample_fps=float(self.params["sample_fps"]),
                min_change_percent=float(self.params["min_change_percent"]),
                min_inner_change_percent=self.params["min_inner_change_percent"],
                min_regions=int(self.params["min_regions"]),
                min_persistence=int(self.params["min_persistence"]),
                edge_ignore_px=int(self.params["edge_ignore_px"]),
                max_pairs=self.params["max_pairs"],
                output_dir=self.params["output_dir"],
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Change Detector Desktop")
        self.resize(1000, 760)

        self.worker_thread: QThread | None = None
        self.worker: CompareWorker | None = None

        self.default_cfg = _resolve_default_config()

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self._build_files_group())
        layout.addWidget(self._build_options_group())
        layout.addLayout(self._build_actions())
        self.result_box = QPlainTextEdit()
        self.result_box.setPlaceholderText("Result JSON will appear here.")
        layout.addWidget(self.result_box)
        self.setCentralWidget(root)

    def _build_files_group(self) -> QGroupBox:
        group = QGroupBox("Files")
        form = QFormLayout(group)
        self.video_a = self._line_with_button("Select Video A", self._pick_video_a)
        self.video_b = self._line_with_button("Select Video B", self._pick_video_b)
        self.output_dir = self._line_with_button("Select Output Folder", self._pick_output_dir)
        self.output_dir[0].setText("outputs/video_changes_gui")
        self.config = self._line_with_button("Select Config", self._pick_config)
        self.config[0].setText(str(self.default_cfg))
        form.addRow("Video A", self._pair_widget(*self.video_a))
        form.addRow("Video B", self._pair_widget(*self.video_b))
        form.addRow("Output Dir", self._pair_widget(*self.output_dir))
        form.addRow("Config", self._pair_widget(*self.config))
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Options")
        grid = QGridLayout(group)

        self.mode = QComboBox()
        self.mode.addItems(["pixel", "ssim", "combined", "robust", "drone"])
        self.mode.setCurrentText("robust")

        self.alignment = QComboBox()
        self.alignment.addItems(["feature-dtw", "ratio"])
        self.illumination = QComboBox()
        self.illumination.addItems(["none", "clahe", "match"])
        self.stabilize = QCheckBox("Stabilize Frames")

        self.sample_fps = QLineEdit("1.0")
        self.min_alignment_score = QLineEdit("0.08")
        self.gap_penalty = QLineEdit("-0.05")
        self.min_detector_align_conf = QLineEdit("0.0")
        self.min_change_percent = QLineEdit("1.0")
        self.min_inner_change_percent = QLineEdit("")
        self.min_regions = QLineEdit("1")
        self.min_persistence = QLineEdit("1")
        self.edge_ignore_px = QLineEdit("0")
        self.max_pairs = QLineEdit("")

        fields = [
            ("Mode", self.mode),
            ("Alignment", self.alignment),
            ("Illumination", self.illumination),
            ("Sample FPS", self.sample_fps),
            ("Min Alignment Score", self.min_alignment_score),
            ("Gap Penalty", self.gap_penalty),
            ("Min Detector Align Conf", self.min_detector_align_conf),
            ("Min Change %", self.min_change_percent),
            ("Min Inner Change % (optional)", self.min_inner_change_percent),
            ("Min Regions", self.min_regions),
            ("Min Persistence", self.min_persistence),
            ("Edge Ignore Px", self.edge_ignore_px),
            ("Max Pairs (optional)", self.max_pairs),
        ]
        row = 0
        for label, widget in fields:
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
            row += 1
        grid.addWidget(self.stabilize, row, 0, 1, 2)
        return group

    def _build_actions(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Comparison")
        self.run_btn.clicked.connect(self.run_comparison)
        open_out_btn = QPushButton("Open Output Folder")
        open_out_btn.clicked.connect(self.open_output_folder)
        layout.addWidget(self.run_btn)
        layout.addWidget(open_out_btn)
        layout.addStretch()
        self.status = QLabel("Ready")
        layout.addWidget(self.status)
        return layout

    def _line_with_button(self, button_text: str, callback):
        line = QLineEdit()
        button = QPushButton(button_text)
        button.clicked.connect(callback)
        return line, button

    def _pair_widget(self, line: QLineEdit, button: QPushButton) -> QWidget:
        wrap = QWidget()
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line)
        layout.addWidget(button)
        return wrap

    def _pick_video_a(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Video A", "", "Videos (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self.video_a[0].setText(path)

    def _pick_video_b(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Video B", "", "Videos (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self.video_b[0].setText(path)

    def _pick_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output Folder")
        if path:
            self.output_dir[0].setText(path)

    def _pick_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Config", "", "YAML (*.yaml *.yml)")
        if path:
            self.config[0].setText(path)

    def run_comparison(self) -> None:
        try:
            params = {
                "video_a": self.video_a[0].text().strip(),
                "video_b": self.video_b[0].text().strip(),
                "output_dir": self.output_dir[0].text().strip(),
                "config": self.config[0].text().strip(),
                "mode": self.mode.currentText(),
                "alignment_method": self.alignment.currentText(),
                "illumination_mode": self.illumination.currentText(),
                "stabilize": self.stabilize.isChecked(),
                "sample_fps": float(self.sample_fps.text().strip()),
                "min_alignment_score": float(self.min_alignment_score.text().strip()),
                "gap_penalty": float(self.gap_penalty.text().strip()),
                "min_detector_alignment_conf": float(self.min_detector_align_conf.text().strip()),
                "min_change_percent": float(self.min_change_percent.text().strip()),
                "min_inner_change_percent": (
                    float(self.min_inner_change_percent.text().strip())
                    if self.min_inner_change_percent.text().strip()
                    else None
                ),
                "min_regions": int(self.min_regions.text().strip()),
                "min_persistence": int(self.min_persistence.text().strip()),
                "edge_ignore_px": int(self.edge_ignore_px.text().strip()),
                "max_pairs": int(self.max_pairs.text().strip()) if self.max_pairs.text().strip() else None,
            }
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        if not Path(params["video_a"]).exists() or not Path(params["video_b"]).exists():
            QMessageBox.critical(self, "Missing Files", "Please select valid Video A and Video B files.")
            return
        if not Path(params["config"]).exists():
            QMessageBox.critical(self, "Missing Config", "Please select a valid config file.")
            return

        self.run_btn.setEnabled(False)
        self.status.setText("Running...")
        self.result_box.setPlainText("Running comparison...\n")

        self.worker_thread = QThread()
        self.worker = CompareWorker(params)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _on_done(self, result: dict) -> None:
        self.result_box.setPlainText(json.dumps(result, indent=2))
        self.status.setText("Done")
        self.run_btn.setEnabled(True)
        QMessageBox.information(
            self,
            "Complete",
            f"Comparison complete.\nChanged pairs: {result.get('num_changed_pairs', 0)}",
        )

    def _on_error(self, error_text: str) -> None:
        self.result_box.setPlainText(error_text)
        self.status.setText("Failed")
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_text.splitlines()[0] if error_text else "Unknown error")

    def open_output_folder(self) -> None:
        path = Path(self.output_dir[0].text().strip() or "outputs/video_changes_gui")
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


def main() -> None:
    logging.basicConfig(
        filename=str(LOG_FILE),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Launching desktop app")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        msg = (
            f"The app crashed during startup.\n\n{exc}\n\n"
            f"A log may be available at:\n{LOG_FILE}"
        )
        _show_fatal_error(msg)
        logging.exception("Fatal startup error")
        raise
