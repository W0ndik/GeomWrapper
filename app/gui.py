import os
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv

from app.config import AppParams
from app.pipeline import run_pipeline, PipelineResult


class PipelineWorker(QtCore.QThread):
    finished_ok = QtCore.Signal(object)   # PipelineResult
    finished_err = QtCore.Signal(str)

    def __init__(self, stl_path: str, params: AppParams):
        super().__init__()
        self._path = stl_path
        self._params = params

    def run(self):
        try:
            res = run_pipeline(self._path, self._params)
            self.finished_ok.emit(res)
        except Exception as e:
            self.finished_err.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Wrapper Cubes Prototype")

        self._stl_path = ""
        self._worker = None
        self._res: PipelineResult | None = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)

        self.ctrl = self._build_controls()
        root.addWidget(self.ctrl, 0)

        self.vtk_widget = QtInteractor(central)
        root.addWidget(self.vtk_widget.interactor, 1)

        self.plotter = self.vtk_widget
        self.plotter.set_background("white")

        self._actors = {}
        self._build_default_scene()

        self.resize(1300, 800)

    def _build_default_scene(self):
        self.plotter.add_text("Open STL and run pipeline", font_size=12)

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        file_box = QtWidgets.QGroupBox("Input")
        fl = QtWidgets.QVBoxLayout(file_box)

        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Open STL")
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_save = QtWidgets.QPushButton("Save shrink shell STL")
        self.btn_save.setEnabled(False)

        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_run)

        fl.addWidget(self.path_edit)
        fl.addLayout(btn_row)
        fl.addWidget(self.btn_save)

        lay.addWidget(file_box)

        mode_box = QtWidgets.QGroupBox("Mode")
        ml = QtWidgets.QFormLayout(mode_box)

        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItem("Uniform grid", userData="uniform")
        self.cb_mode.addItem("Octree 2to1", userData="octree")

        self.sp_oct_level = QtWidgets.QSpinBox()
        self.sp_oct_level.setRange(1, 12)
        self.sp_oct_level.setValue(6)

        self.sp_oct_bal_iters = QtWidgets.QSpinBox()
        self.sp_oct_bal_iters.setRange(0, 200)
        self.sp_oct_bal_iters.setValue(20)

        ml.addRow("grid mode", self.cb_mode)
        ml.addRow("octree max level", self.sp_oct_level)
        ml.addRow("balance iters", self.sp_oct_bal_iters)

        lay.addWidget(mode_box)

        grid_box = QtWidgets.QGroupBox("Grid")
        gl = QtWidgets.QFormLayout(grid_box)

        self.sp_pitch = QtWidgets.QDoubleSpinBox()
        self.sp_pitch.setRange(0.001, 1e6)
        self.sp_pitch.setDecimals(6)
        self.sp_pitch.setValue(2.0)

        self.sp_padding_mul = QtWidgets.QDoubleSpinBox()
        self.sp_padding_mul.setRange(0.0, 1000.0)
        self.sp_padding_mul.setDecimals(3)
        self.sp_padding_mul.setValue(3.0)

        self.sp_band_mul = QtWidgets.QDoubleSpinBox()
        self.sp_band_mul.setRange(0.0, 1000.0)
        self.sp_band_mul.setDecimals(3)
        self.sp_band_mul.setValue(0.75)

        self.sp_max_dim = QtWidgets.QSpinBox()
        self.sp_max_dim.setRange(16, 2000)
        self.sp_max_dim.setValue(260)

        gl.addRow("pitch", self.sp_pitch)
        gl.addRow("padding mul", self.sp_padding_mul)
        gl.addRow("band mul", self.sp_band_mul)
        gl.addRow("max dim", self.sp_max_dim)

        lay.addWidget(grid_box)

        sh_box = QtWidgets.QGroupBox("Shrink")
        sl = QtWidgets.QFormLayout(sh_box)

        self.sp_iters = QtWidgets.QSpinBox()
        self.sp_iters.setRange(0, 10000)
        self.sp_iters.setValue(25)

        self.sp_step = QtWidgets.QDoubleSpinBox()
        self.sp_step.setRange(0.0, 1.0)
        self.sp_step.setDecimals(4)
        self.sp_step.setValue(0.35)

        self.sp_constraint = QtWidgets.QDoubleSpinBox()
        self.sp_constraint.setRange(0.0, 10.0)
        self.sp_constraint.setDecimals(4)
        self.sp_constraint.setValue(0.35)

        self.sp_lap_iters = QtWidgets.QSpinBox()
        self.sp_lap_iters.setRange(0, 1000)
        self.sp_lap_iters.setValue(1)

        self.sp_lap_relax = QtWidgets.QDoubleSpinBox()
        self.sp_lap_relax.setRange(0.0, 1.0)
        self.sp_lap_relax.setDecimals(4)
        self.sp_lap_relax.setValue(0.15)

        sl.addRow("iters", self.sp_iters)
        sl.addRow("step", self.sp_step)
        sl.addRow("constraint mul", self.sp_constraint)
        sl.addRow("lap iters per step", self.sp_lap_iters)
        sl.addRow("lap relax", self.sp_lap_relax)

        lay.addWidget(sh_box)

        view_box = QtWidgets.QGroupBox("View")
        vl = QtWidgets.QVBoxLayout(view_box)

        self.cb_show_mesh = QtWidgets.QCheckBox("Show STL")
        self.cb_show_cubes = QtWidgets.QCheckBox("Show front cubes")
        self.cb_show_shell0 = QtWidgets.QCheckBox("Show cube shell")
        self.cb_show_shell1 = QtWidgets.QCheckBox("Show shrink shell")
        self.cb_show_wire = QtWidgets.QCheckBox("Show wireframe shrink shell")

        self.cb_show_mesh.setChecked(True)
        self.cb_show_cubes.setChecked(True)
        self.cb_show_shell0.setChecked(False)
        self.cb_show_shell1.setChecked(True)
        self.cb_show_wire.setChecked(True)

        vl.addWidget(self.cb_show_mesh)
        vl.addWidget(self.cb_show_cubes)
        vl.addWidget(self.cb_show_shell0)
        vl.addWidget(self.cb_show_shell1)
        vl.addWidget(self.cb_show_wire)

        lay.addWidget(view_box)

        self.status = QtWidgets.QLabel("Ready")
        lay.addWidget(self.status)

        lay.addStretch(1)

        self.btn_open.clicked.connect(self._on_open)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_save.clicked.connect(self._on_save)

        self.cb_show_mesh.toggled.connect(self._sync_visibility)
        self.cb_show_cubes.toggled.connect(self._sync_visibility)
        self.cb_show_shell0.toggled.connect(self._sync_visibility)
        self.cb_show_shell1.toggled.connect(self._sync_visibility)
        self.cb_show_wire.toggled.connect(self._sync_visibility)

        return w

    def _collect_params(self) -> AppParams:
        p = AppParams()
        p.grid.pitch = float(self.sp_pitch.value())
        p.grid.padding_mul = float(self.sp_padding_mul.value())
        p.grid.band_mul = float(self.sp_band_mul.value())
        p.grid.max_dim = int(self.sp_max_dim.value())

        mode = self.cb_mode.currentData()
        p.octree.enabled = (mode == "octree")
        p.octree.max_level = int(self.sp_oct_level.value())
        p.octree.balance_max_iters = int(self.sp_oct_bal_iters.value())

        p.shrink.iters = int(self.sp_iters.value())
        p.shrink.step = float(self.sp_step.value())
        p.shrink.constraint_mul = float(self.sp_constraint.value())
        p.shrink.lap_iters_per_step = int(self.sp_lap_iters.value())
        p.shrink.lap_relax = float(self.sp_lap_relax.value())
        return p

    def _set_busy(self, busy: bool, text: str = ""):
        self.btn_open.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and bool(self._stl_path))
        self.btn_save.setEnabled((not busy) and (self._res is not None))
        self.status.setText(text if text else ("Busy" if busy else "Ready"))

    def _on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select STL", "", "STL files (*.stl);;All files (*.*)")
        if not path:
            return
        self._stl_path = path
        self.path_edit.setText(path)
        self.btn_run.setEnabled(True)
        self.status.setText("STL selected")

    def _on_run(self):
        if not self._stl_path or not os.path.isfile(self._stl_path):
            QtWidgets.QMessageBox.critical(self, "Error", "STL path is invalid")
            return

        params = self._collect_params()

        self._set_busy(True, "Running pipeline")
        self._worker = PipelineWorker(self._stl_path, params)
        self._worker.finished_ok.connect(self._on_pipeline_ok)
        self._worker.finished_err.connect(self._on_pipeline_err)
        self._worker.start()

    def _on_pipeline_ok(self, res_obj):
        self._res = res_obj
        self._worker = None
        self._render_result(res_obj)
        self._set_busy(False, f"Done. pitch_used={res_obj.pitch_used:.4g} band={res_obj.band:.4g}")

    def _on_pipeline_err(self, msg: str):
        self._worker = None
        self._set_busy(False, "Error")
        QtWidgets.QMessageBox.critical(self, "Pipeline error", msg)

    def _render_result(self, res: PipelineResult):
        self.plotter.clear()
        self._actors.clear()

        a_mesh = self.plotter.add_mesh(res.mesh, opacity=0.25, show_edges=False)
        a_cubes = self.plotter.add_mesh(res.vox_front, opacity=0.35, show_edges=True)
        a_shell0 = self.plotter.add_mesh(res.shell0, opacity=0.20, show_edges=False)
        a_shell1 = self.plotter.add_mesh(res.shell1, opacity=0.60, show_edges=False)
        a_wire = self.plotter.add_mesh(res.shell1, style="wireframe", line_width=1)

        self._actors["mesh"] = a_mesh
        self._actors["cubes"] = a_cubes
        self._actors["shell0"] = a_shell0
        self._actors["shell1"] = a_shell1
        self._actors["wire"] = a_wire

        self.plotter.reset_camera()
        self._sync_visibility()
        self.btn_save.setEnabled(True)

    def _sync_visibility(self):
        def set_vis(name: str, vis: bool):
            act = self._actors.get(name)
            if act is None:
                return
            act.SetVisibility(1 if vis else 0)

        set_vis("mesh", self.cb_show_mesh.isChecked())
        set_vis("cubes", self.cb_show_cubes.isChecked())
        set_vis("shell0", self.cb_show_shell0.isChecked())
        set_vis("shell1", self.cb_show_shell1.isChecked())
        set_vis("wire", self.cb_show_wire.isChecked())

        self.plotter.render()

    def _on_save(self):
        if self._res is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save STL", "shell_shrink.stl", "STL files (*.stl)")
        if not path:
            return
        try:
            self._res.shell1.save(path)
            self.status.setText(f"Saved: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))


def run_gui():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()