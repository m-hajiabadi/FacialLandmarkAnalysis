__author__ = "Morteza Hajiabadi"
__copyright__ = "Copyright 2026, Morteza Hajiabadi"
__license__ = "Proprietary / All Rights Reserved"

import os
import shutil
import sys
import json
import hashlib
import logging
import math
import tempfile
import subprocess
import vtk, qt, ctk, slicer # type: ignore
import platform
import numpy as np
from slicer.ScriptedLoadableModule import * # type: ignore
from slicer.util import VTKObservationMixin # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Environment Helper Functions (.packages directory)
# ─────────────────────────────────────────────────────────────────────────────

def get_packages_dir():
    """Return the absolute path of .packages folder next to this .py file."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, ".packages")

# Single source of truth for the AI dependency set. Both the installer and the
# manifest hash (used to decide whether a re-install is needed) read from this,
# so bumping a version here is the only thing required to invalidate old installs.
AI_IMPORT_NAMES = ["numpy", "torch", "cv2", "scipy", "skimage", "PIL", "pandas", "yaml", "tqdm", "matplotlib", "timm"]
AI_PIP_PACKAGES = [
    "numpy<2.0.0",
    "scipy",
    "scikit-image",
    "opencv-python-headless",
    "Pillow",
    "pandas",
    "PyYAML",
    "tqdm",
    "timm",
    "matplotlib",
]

_MANIFEST_FILENAME = ".install_manifest.json"

def _manifest_path(packages_dir):
    return os.path.join(packages_dir, _MANIFEST_FILENAME)

def _expected_manifest_fingerprint():
    """
    Fingerprint of "what should be installed" - the pinned package list plus the
    interpreter/platform it was installed for. If any of these change (e.g. the
    module is updated with a new dependency, or Slicer's bundled Python changes),
    the fingerprint changes and a fresh install is correctly triggered again.
    """
    raw = json.dumps({
        "packages": sorted(AI_PIP_PACKAGES),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "torch_index": os.environ.get("FLA_TORCH_CUDA_INDEX", "default"),
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _read_manifest(packages_dir):
    path = _manifest_path(packages_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_manifest(packages_dir):
    path = _manifest_path(packages_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "fingerprint": _expected_manifest_fingerprint(),
                "installed_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2)
    except Exception as e:
        logging.warning(f"Could not write install manifest: {e}")

def _run_ai_import_check(packages_dir):
    """
    Actively tests importing AI packages from .packages in an isolated subprocess.
    This is the expensive path (spawns a process, imports 10+ packages) and should
    only be used to *confirm* a first-time install, or as a one-off fallback when
    the manifest is missing/stale - not on every module load.
    """
    packages_dir_clean = packages_dir.replace('\\', '/')
    test_code = f"""import sys, os
# Filter out Slicer site-packages to avoid binary/version collisions
sys.path = [p for p in sys.path if 'site-packages' not in p.lower()]
sys.path.insert(0, '{packages_dir_clean}')
if sys.platform == 'win32':
    t_lib = os.path.join('{packages_dir_clean}', 'torch', 'lib')
    if os.path.isdir(t_lib):
        os.add_dll_directory(t_lib)
import {', '.join(AI_IMPORT_NAMES)}
print('ALL_OK')
"""
    try:
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{packages_dir}{os.pathsep}{existing_pp}" if existing_pp else packages_dir

        res = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=120,  # generous timeout to allow cold HDD disk read; only paid once
            env=env
        )
        if res.returncode == 0 and "ALL_OK" in res.stdout:
            return True
        logging.warning(
            f"AI environment verification failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )
        return False
    except Exception as e:
        logging.warning(f"AI environment check error: {e}")
        return False

def is_ai_environment_ready(force_full_check=False):
    """
    Cheap, fast check used on every module load / detection run.

    Trusts a persistent manifest file written after a successful install: if the
    manifest exists and its fingerprint matches the current requirement set, the
    environment is considered ready WITHOUT spawning a subprocess or re-importing
    every package. This is what makes installation happen only once - on
    subsequent runs this function is just a file read.

    Falls back to the full (slow) import check only when the manifest is
    missing/mismatched, or when force_full_check=True is explicitly requested
    (e.g. right after an install, to confirm it actually worked before trusting it).
    """
    packages_dir = get_packages_dir()
    if not os.path.isdir(packages_dir):
        return False

    manifest = _read_manifest(packages_dir)
    manifest_ok = bool(manifest) and manifest.get("fingerprint") == _expected_manifest_fingerprint()

    if manifest_ok and not force_full_check:
        return True

    # No trustworthy manifest yet (first run, manifest missing/corrupted, or the
    # pinned requirement set changed) - do the one-off expensive verification.
    ready = _run_ai_import_check(packages_dir)
    if ready:
        _write_manifest(packages_dir)
    return ready


def setup_inference_environment(status_label=None):
    """
    Installs AI dependencies into .packages using pip --target.

    Runs ONLY on a genuine first-time setup: if is_ai_environment_ready() already
    trusts the manifest, this function returns immediately without touching pip
    or the network. Once packages are installed and verified, a manifest file is
    written so all future calls (across Slicer restarts) take the fast path in
    is_ai_environment_ready() instead of re-running this installer.
    """
    packages_dir = get_packages_dir()

    if is_ai_environment_ready():
        return packages_dir

    logging.info(f"Performing first-time AI package setup at: {packages_dir}")
    os.makedirs(packages_dir, exist_ok=True)

    cuda_index = os.environ.get("FLA_TORCH_CUDA_INDEX", "https://download.pytorch.org/whl/cu121")
    system = platform.system()

    # 1. Install PyTorch. No --upgrade: on a first-time install the --target dir is
    #    empty so pip installs fresh; on a repeat call (manifest missing/stale) we
    #    still don't want pip silently jumping to a newer, unverified torch build.
    if status_label:
        status_label.setText("⏳ در حال بررسی و دانلود PyTorch (فقط بار اول)...")
        slicer.app.processEvents()

    if system == "Darwin":
        cmd_torch = [
            sys.executable, "-m", "pip", "install",
            "--no-user", "--target", packages_dir,
            "torch", "torchvision"
        ]
    else:
        cmd_torch = [
            sys.executable, "-m", "pip", "install",
            "--no-user", "--target", packages_dir,
            "torch", "torchvision",
            "--index-url", cuda_index
        ]

    ok1 = run_pip_streaming(cmd_torch, status_label, "در حال بررسی PyTorch")
    if not ok1:
        logging.error("Failed to install PyTorch.")
        return None

    # 2. Install remaining AI libraries (pinned list shared with the manifest fingerprint).
    if status_label:
        status_label.setText("⏳ در حال بررسی پکیج‌های مکمل هوش مصنوعی...")
        slicer.app.processEvents()

    cmd_deps = [
        sys.executable, "-m", "pip", "install",
        "--no-user", "--target", packages_dir
    ] + AI_PIP_PACKAGES

    ok2 = run_pip_streaming(cmd_deps, status_label, "در حال بررسی سایر پکیج‌ها")
    if not ok2:
        logging.error("Failed to install AI dependencies.")
        return None

    # 3. Confirm the install actually works, then persist the manifest so every
    #    future run (this session and after Slicer restarts) skips straight past
    #    both pip and the subprocess import check.
    if not is_ai_environment_ready(force_full_check=True):
        logging.error("AI packages installed but failed post-install verification.")
        return None

    logging.info("✓ Isolated AI dependencies installed and verified (first-time setup complete).")
    return packages_dir

def run_pip_streaming(cmd, status_label=None, status_prefix=""):
    """
    Runs pip install while streaming download output in real-time
    to keep Slicer's UI responsive during PyTorch (~2.5GB) download.
    """
    logging.info(f"Running pip command: {' '.join(cmd)}")
    env = os.environ.copy()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        for line in iter(process.stdout.readline, ''):
            line_str = line.strip()
            if line_str:
                print(f"[pip] {line_str}")
                logging.info(f"[pip] {line_str}")

                if status_label and any(k in line_str for k in ["MB", "%", "Downloading", "Installing", "Collecting"]):
                    status_label.setText(f"⏳ {status_prefix}\n{line_str[:80]}")
                    slicer.app.processEvents()

        process.stdout.close()
        return process.wait() == 0
    except Exception as e:
        logging.error(f"Pip execution error: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# UI / Reporting Dependencies (Installed into Slicer directly)
# ─────────────────────────────────────────────────────────────────────────────

_DEPENDENCIES_CHECKED = False

def ensure_dependencies():
    """
    Installs ONLY pure-Python reporting & UI packages into Slicer.
    NO numpy, scipy, skimage, or torch are touched here.
    """
    global _DEPENDENCIES_CHECKED
    if _DEPENDENCIES_CHECKED:
        return

    safe_ui_packages = [
        ('openpyxl', 'openpyxl'),
        ('Pillow', 'PIL'),
        ('jdatetime', 'jdatetime'),
        ('reportlab', 'reportlab'),
        ('arabic-reshaper', 'arabic_reshaper'),
        ('python-bidi', 'bidi'),
    ]

    for pkg_name, module_name in safe_ui_packages:
        try:
            __import__(module_name)
        except ImportError:
            try:
                logging.info(f"Installing UI package: {pkg_name}")
                slicer.util.pip_install(pkg_name)
            except Exception as e:
                logging.warning(f"Could not install {pkg_name}: {e}")

    _DEPENDENCIES_CHECKED = True
         
def to_persian_digits(text):
    """Convert English digits to Persian digits."""
    en_to_fa = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
    return str(text).translate(en_to_fa)

#
# Module
#

class FacialLandmarkAnalysis(ScriptedLoadableModule): # type: ignore
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent) # type: ignore
        self.parent.title = "Facial Landmark Analysis"
        self.parent.categories = ["Orthodontics"]
        self.parent.dependencies = []
        self.parent.contributors = ["Morteza Hajiabadi"]
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        iconPath = os.path.join(moduleDir, 'Resources', 'Icons', 'FacialLandmarkAnalysis.png')
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        self.parent.helpText = "Automatic facial landmark detection with Persian Excel export."
        self.parent.acknowledgementText = "Developed & Maintained by Morteza Hajiabadi"

#
# Widget
#

class FacialLandmarkAnalysisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin): # type: ignore
    VIEW_KEYS = ['frontal', 'right', 'left', 'smile']
    VIEW_CODES = {'frontal': 'F', 'right': 'L', 'left': 'L', 'smile': 'S'}
    VIEW_LABELS_FA = {
        'frontal': 'نمای روبرو (Frontal)',
        'right':   'نمای نیمرخ راست (Right Profile)',
        'left':    'نمای نیمرخ چپ (Left Profile)',
        'smile':   'نمای لبخند (Smile)',
    }
    
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)  # type: ignore
        VTKObservationMixin.__init__(self)
        
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)  # type: ignore
        
        ensure_dependencies()

        self.logic = FacialLandmarkAnalysisLogic()
        self.imagePaths = {k: None for k in self.VIEW_KEYS}
        self.imageNodes = {k: None for k in self.VIEW_KEYS}
        self.imageSizes = {k: None for k in self.VIEW_KEYS}
        self.markupNodes = {k: None for k in self.VIEW_KEYS}
        self.inferenceResults = {k: None for k in self.VIEW_KEYS}
        self.landmarksDetected = False
        
        self._ckptEdits = {}

        moduleDir = os.path.dirname(os.path.abspath(__file__))
        
        # ── HARDCODED MODEL PATHS (no UI) ──
        self._inferScriptPath = os.path.join(moduleDir, 'models', 'scripts', 'infer.py')
        self._ckptPaths = {
            'f_coarse': os.path.join(moduleDir, 'models', 'outputs', 'f_coarse', 'checkpoints', 'best_val_mre_px.pt'),
            'f_fine':   os.path.join(moduleDir, 'models', 'outputs', 'f_fine',   'checkpoints', 'best_val_mre_px.pt'),
            'l_coarse': os.path.join(moduleDir, 'models', 'outputs', 'l_coarse', 'checkpoints', 'best_val_mre_px.pt'),
            'l_fine':   os.path.join(moduleDir, 'models', 'outputs', 'l_fine',   'checkpoints', 'best_val_mre_px.pt'),
            's_coarse': os.path.join(moduleDir, 'models', 'outputs', 's_coarse', 'checkpoints', 'best_val_mre_px.pt'),
            's_fine':   os.path.join(moduleDir, 'models', 'outputs', 's_fine',   'checkpoints', 'best_val_s_combined.pt'),
        }
        self._presenceThresh = "0.6"
        
        # ── Patient Info ──
        patientCollapsible = ctk.ctkCollapsibleButton()
        patientCollapsible.text = "اطلاعات بیمار"
        self.layout.addWidget(patientCollapsible)
        patientLayout = qt.QFormLayout(patientCollapsible)
        self.patientNameEdit = qt.QLineEdit()
        patientLayout.addRow(":نام بیمار", self.patientNameEdit)
        self.doctorNameEdit = qt.QLineEdit()
        self.doctorNameEdit.setText("دکتر سید علیرضا پرهیز")
        patientLayout.addRow(":نام پزشک", self.doctorNameEdit)

        import jdatetime # type: ignore
        
        self.dateEdit = qt.QLineEdit()
        today_str = jdatetime.date.today().strftime("%Y/%m/%d")
        persian_date = to_persian_digits(today_str)
        self.dateEdit.setText(persian_date)
        patientLayout.addRow("تاریخ:", self.dateEdit)

        # ── Load Images (4 views) ──
        imageCollapsible = ctk.ctkCollapsibleButton()
        imageCollapsible.text = "مرحله ۱ : بارگذاری تصاویر"
        self.layout.addWidget(imageCollapsible)
        imageLayout = qt.QGridLayout(imageCollapsible)

        self._loadBtns = {}
        self._loadLabels = {}
        btn_texts = {
            'frontal': 'Frontal',
            'right': 'Right Profile',
            'left': 'Left Profile',
            'smile': 'Smile'
        }
        for row_idx, vk in enumerate(self.VIEW_KEYS):
            btn = qt.QPushButton(btn_texts[vk])
            btn.connect('clicked()', lambda v=vk: self.onLoadImage(v))
            lbl = qt.QLabel("Not loaded")
            imageLayout.addWidget(btn, row_idx, 0)
            imageLayout.addWidget(lbl, row_idx, 1)
            self._loadBtns[vk] = btn
            self._loadLabels[vk] = lbl

        # ── Detection ──
        detectionCollapsible = ctk.ctkCollapsibleButton()
        detectionCollapsible.text = "مرحله ۲ : تشخیص لندمارک ها"
        self.layout.addWidget(detectionCollapsible)
        detectionLayout = qt.QVBoxLayout(detectionCollapsible)

        self.runDetectionBtn = qt.QPushButton("🔍 اجرا")
        self.runDetectionBtn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-size: 14px; "
            "font-weight: bold; padding: 12px;"
        )
        self.runDetectionBtn.connect('clicked()', self.onRunDetection)
        detectionLayout.addWidget(self.runDetectionBtn)

        self.detectionStatusLabel = qt.QLabel("در انتظار بارگذاری تصاویر و اجرای مدل")
        self.detectionStatusLabel.setStyleSheet("color: orange; font-style: italic;")
        self.detectionStatusLabel.setWordWrap(True)
        detectionLayout.addWidget(self.detectionStatusLabel)

        # Progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 4)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        detectionLayout.addWidget(self.progressBar)

        # ── Review ──
        reviewCollapsible = ctk.ctkCollapsibleButton()
        reviewCollapsible.text = "مرحله ۳: بررسی و اصلاح لندمارک ها "
        self.layout.addWidget(reviewCollapsible)
        reviewLayout = qt.QVBoxLayout(reviewCollapsible)

        viewSelectorLayout = qt.QHBoxLayout()
        viewSelectorLayout.addWidget(qt.QLabel("View:"))
        self.viewComboBox = qt.QComboBox()
        self.viewComboBox.addItems([
            "Frontal View", "Right Profile", "Left Profile", "Smile View"
        ])
        self.viewComboBox.connect(
            'currentIndexChanged(int)', self.onViewChanged)
        viewSelectorLayout.addWidget(self.viewComboBox)
        reviewLayout.addLayout(viewSelectorLayout)

        self.landmarkListWidget = qt.QTextEdit()
        self.landmarkListWidget.setReadOnly(True)
        self.landmarkListWidget.setMaximumHeight(200)
        self.landmarkListWidget.setStyleSheet(
            "font-family: monospace; font-size: 11px;")
        reviewLayout.addWidget(self.landmarkListWidget)

        # ── Manual Landmark Editing ──
        editCollapsible = ctk.ctkCollapsibleButton()
        editCollapsible.text = "ویرایش دستی لندمارک ها"
        self.layout.addWidget(editCollapsible)
        editLayout = qt.QVBoxLayout(editCollapsible)
        
        addRow = qt.QHBoxLayout()
        self.addLandmarkBtn = qt.QPushButton("➕ افزودن لندمارک جدید")
        self.addLandmarkBtn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
        )
        self.addLandmarkBtn.connect('clicked()', self.onStartAddLandmark)
        addRow.addWidget(self.addLandmarkBtn)
        editLayout.addLayout(addRow)
        
        deleteRow = qt.QHBoxLayout()
        deleteRow.addWidget(qt.QLabel("حذف لندمارک شماره:"))
        self.deleteLandmarkCombo = qt.QComboBox()
        self.deleteLandmarkCombo.setMinimumWidth(100)
        deleteRow.addWidget(self.deleteLandmarkCombo)
        self.deleteLandmarkBtn = qt.QPushButton("🗑️ حذف")
        self.deleteLandmarkBtn.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 6px;"
        )
        self.deleteLandmarkBtn.connect('clicked()', self.onDeleteLandmark)
        deleteRow.addWidget(self.deleteLandmarkBtn)
        editLayout.addLayout(deleteRow)
        
        self.editStatusLabel = qt.QLabel("")
        self.editStatusLabel.setStyleSheet("color: blue; font-style: italic;")
        editLayout.addWidget(self.editStatusLabel)
        
        # ── Scale ──
        calibCollapsible = ctk.ctkCollapsibleButton()
        calibCollapsible.text = "کالیبراسیون مقیاس (اختیاری)"
        calibCollapsible.collapsed = True
        self.layout.addWidget(calibCollapsible)
        calibLayout = qt.QFormLayout(calibCollapsible)
        self.pixelSizeEdit = qt.QLineEdit()
        self.pixelSizeEdit.setText("1.0")
        calibLayout.addRow(":پیکسل در هر میلی‌متر", self.pixelSizeEdit)

        # ── Export ──
        exportCollapsible = ctk.ctkCollapsibleButton()
        exportCollapsible.text = "مرحله ۴: خروجی گرفتن از نتایج"
        self.layout.addWidget(exportCollapsible)
        exportLayout = qt.QVBoxLayout(exportCollapsible)

        # Excel button
        self.exportBtn = qt.QPushButton()
        self.exportBtn.setText("  خروجی Excel")
        self.exportBtn.setIcon(qt.QIcon.fromTheme("x-office-spreadsheet"))
        self.exportBtn.setIconSize(qt.QSize(24, 24))
        self.exportBtn.setStyleSheet(
            "QPushButton { background-color: #1B7E3E; color: white; font-size: 13px; "
            "font-weight: bold; padding: 10px; border-radius: 4px; text-align: left; padding-left: 20px; }"
            "QPushButton:hover { background-color: #14652F; }"
            "QPushButton:disabled { background-color: #9E9E9E; }"
        )
        self.exportBtn.connect('clicked()', self.onExportExcel)
        self.exportBtn.enabled = False
        exportLayout.addWidget(self.exportBtn)
        
        # PDF button
        self.exportPdfBtn = qt.QPushButton()
        self.exportPdfBtn.setText("  خروجی PDF")
        self.exportPdfBtn.setIcon(qt.QIcon.fromTheme("application-pdf"))
        self.exportPdfBtn.setIconSize(qt.QSize(24, 24))
        self.exportPdfBtn.setStyleSheet(
            "QPushButton { background-color: #B71C1C; color: white; font-size: 13px; "
            "font-weight: bold; padding: 10px; border-radius: 4px; text-align: left; padding-left: 20px; }"
            "QPushButton:hover { background-color: #8B0000; }"
            "QPushButton:disabled { background-color: #9E9E9E; }"
        )
        self.exportPdfBtn.connect('clicked()', self.onExportPDF)
        self.exportPdfBtn.enabled = False
        exportLayout.addWidget(self.exportPdfBtn)
        
        # Both button
        self.exportBothBtn = qt.QPushButton()
        self.exportBothBtn.setText("  خروجی کامل (Excel + PDF)")
        self.exportBothBtn.setIcon(qt.QIcon.fromTheme("document-save-all"))
        self.exportBothBtn.setIconSize(qt.QSize(24, 24))
        self.exportBothBtn.setStyleSheet(
            "QPushButton { background-color: #4A148C; color: white; font-size: 13px; "
            "font-weight: bold; padding: 10px; border-radius: 4px; text-align: left; padding-left: 20px; }"
            "QPushButton:hover { background-color: #311B92; }"
            "QPushButton:disabled { background-color: #9E9E9E; }"
        )
        self.exportBothBtn.connect('clicked()', self.onExportBoth)
        self.exportBothBtn.enabled = False
        exportLayout.addWidget(self.exportBothBtn)
        
        self.layout.addStretch(1)

    def _buildInferEnv(self):
        env = os.environ.copy()
        packages_dir = get_packages_dir()

        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = packages_dir + os.pathsep + existing_pythonpath
        else:
            env["PYTHONPATH"] = packages_dir

        if platform.system() == "Windows":
            torch_lib = os.path.join(packages_dir, "torch", "lib")
            if os.path.isdir(torch_lib):
                env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")

        return env
            
    # ── Landmark definitions per view ──
    def getFrontalLandmarks(self):
        return [(i, f"L{i}") for i in range(1, 26)]

    def getLateralLandmarks(self):
        return [(i, f"L{i}") for i in range(1, 17)]

    def getSmileLandmarks(self):
        return [(i, f"L{i}") for i in range(1, 9)]

    def getLandmarksForView(self, viewIndex):
        return [
            self.getFrontalLandmarks(),
            self.getLateralLandmarks(),  # right
            self.getLateralLandmarks(),  # left
            self.getSmileLandmarks()
        ][viewIndex]

    def getViewKeyFromIndex(self, index):
        return self.VIEW_KEYS[index]
    
    def onStartAddLandmark(self):
        """Enter interactive placement mode: user clicks on image → prompt for name."""
        currentIdx = self.viewComboBox.currentIndex
        viewKey = self.getViewKeyFromIndex(currentIdx)
        
        if self.markupNodes.get(viewKey) is None:
            slicer.util.warningDisplay("ابتدا مدل را روی این نما اجرا کنید.")
            return
        
        markupNode = self.markupNodes[viewKey]
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(markupNode.GetID())
        
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        interactionNode.SetPlaceModePersistence(0)  # place one point only
        
        self._pointAddedObserver = markupNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            lambda caller, event, vk=viewKey: self._onNewLandmarkPlaced(vk)
        )
        self.editStatusLabel.setText("👆 روی تصویر کلیک کنید تا لندمارک جدید اضافه شود")
    
    def _onNewLandmarkPlaced(self, viewKey):
        markupNode = self.markupNodes[viewKey]
        n = markupNode.GetNumberOfControlPoints()
        newIdx = n - 1  # last added
        
        # Safe, object-oriented dialog creation for Slicer PythonQt
        dialog = qt.QInputDialog(self.parent)
        dialog.setWindowTitle("نام لندمارک")
        dialog.setLabelText("شماره یا نام لندمارک:")
        dialog.setTextValue("")
        
        # Show dialog and check if the user clicked "OK"
        if dialog.exec_() == qt.QDialog.Accepted:
            name = dialog.textValue().strip()
            if name:
                markupNode.SetNthControlPointLabel(newIdx, name)
                markupNode.SetNthControlPointDescription(newIdx, f"Manual_{name}")
            else:
                markupNode.RemoveNthControlPoint(newIdx)
        else:
            # User cancelled, remove the placed point
            markupNode.RemoveNthControlPoint(newIdx)
        
        if hasattr(self, '_pointAddedObserver'):
            markupNode.RemoveObserver(self._pointAddedObserver)
            del self._pointAddedObserver
        
        self.editStatusLabel.setText("")
        self.updateLandmarkList(viewKey)
        self._refreshDeleteCombo(viewKey)
    
    def onDeleteLandmark(self):
        currentIdx = self.viewComboBox.currentIndex
        viewKey = self.getViewKeyFromIndex(currentIdx)
        markupNode = self.markupNodes.get(viewKey)
        if markupNode is None:
            return
        
        target = self.deleteLandmarkCombo.currentText
        if not target:
            return
        
        for i in range(markupNode.GetNumberOfControlPoints()):
            if markupNode.GetNthControlPointLabel(i) == target:
                markupNode.RemoveNthControlPoint(i)
                break
        
        self.updateLandmarkList(viewKey)
        self._refreshDeleteCombo(viewKey)
    
    def _refreshDeleteCombo(self, viewKey):
        self.deleteLandmarkCombo.clear()
        markupNode = self.markupNodes.get(viewKey)
        if markupNode is None:
            return
        for i in range(markupNode.GetNumberOfControlPoints()):
            self.deleteLandmarkCombo.addItem(markupNode.GetNthControlPointLabel(i))
            
    # ── Load image ──  
    def onLoadImage(self, viewKey):
        from PIL import Image
        
        filePath = qt.QFileDialog.getOpenFileName(
            self.parent,
            f"Select {viewKey.replace('_', ' ').title()} Image",
            "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not filePath:
            return

        self.imagePaths[viewKey] = filePath
        img = Image.open(filePath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        H, W = img_array.shape[:2]
        self.imageSizes[viewKey] = (W, H)

        if self.imageNodes[viewKey] is not None:
            slicer.mrmlScene.RemoveNode(self.imageNodes[viewKey])

        volumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLVectorVolumeNode")
        volumeNode.SetName(f"Photo_{viewKey}")
        img_3d = img_array[np.newaxis, :, :, :]
        slicer.util.updateVolumeFromArray(volumeNode, img_3d)
        volumeNode.SetSpacing(1.0, 1.0, 1.0)
        volumeNode.SetOrigin(0.0, 0.0, 0.0)

        # Identity matrix - no flipping
        ijkToRas = vtk.vtkMatrix4x4()
        ijkToRas.Identity()
        volumeNode.SetIJKToRASMatrix(ijkToRas)

        self.imageNodes[viewKey] = volumeNode
        self._loadLabels[viewKey].setText(f"✓ {os.path.basename(filePath)}")
        self._loadLabels[viewKey].setStyleSheet("color: green; font-weight: bold;")
        self.showImage(viewKey)

    def showImage(self, viewKey):
        if self.imageNodes[viewKey] is None:
            return

        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
        slicer.util.setSliceViewerLayers(background=self.imageNodes[viewKey])

        redSliceWidget = slicer.app.layoutManager().sliceWidget('Red')
        redSliceNode = redSliceWidget.mrmlSliceNode()
        redSliceLogic = redSliceWidget.sliceLogic()

        sliceToRAS = vtk.vtkMatrix4x4()
        sliceToRAS.Identity()
        sliceToRAS.SetElement(0, 0, 1.0)
        sliceToRAS.SetElement(1, 1, -1.0)
        sliceToRAS.SetElement(2, 2, 1.0)
        redSliceNode.GetSliceToRAS().DeepCopy(sliceToRAS)
        redSliceNode.UpdateMatrices()

        redSliceLogic.FitSliceToAll()
        redSliceLogic.SnapSliceOffsetToIJK()

        for key, node in self.markupNodes.items():
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(key == viewKey)
                node.GetDisplayNode().SetViewNodeIDs([redSliceNode.GetID()])

    def _flipImageHorizontally(self, imagePath, outPath):
        """Flip an image horizontally and save it."""
        from PIL import Image, ImageOps
        img = Image.open(imagePath)
        flipped = ImageOps.mirror(img)
        flipped.save(outPath)
        return outPath
    
    def _flipLandmarksHorizontally(self, coords, imageWidth):
        """Mirror landmark x-coordinates around image center."""
        return {lm_id: (imageWidth - x, y) for lm_id, (x, y) in coords.items()}
    
    def onRunDetection(self):
        import time

        missing = [k for k, v in self.imageNodes.items() if v is None]
        if missing:
            slicer.util.warningDisplay(
                f"Please load all 4 images first.\nMissing: {', '.join(missing)}")
            return

        # ── Ensure isolated AI packages are installed ──
        if not is_ai_environment_ready():
            self.detectionStatusLabel.setText("⏳ در حال دانلود و نصب مدل‌ها و پکیج‌های هوش مصنوعی (فقط بار اول)...")
            self.detectionStatusLabel.setStyleSheet("color: blue; font-weight: bold;")
            slicer.app.processEvents()

            res = setup_inference_environment(self.detectionStatusLabel)
            if not res or not is_ai_environment_ready():
                self.detectionStatusLabel.setText("❌ خطا در راه‌اندازی وابستگی‌های هوش مصنوعی")
                self.detectionStatusLabel.setStyleSheet("color: red; font-weight: bold;")
                slicer.util.errorDisplay("Could not setup AI packages. Please check the Python console for details.")
                return

        # Validate infer.py and checkpoints
        infer_script = self._inferScriptPath
        if not os.path.isfile(infer_script):
            slicer.util.errorDisplay(f"infer.py not found:\n{infer_script}")
            return

        for ckpt_key, ckpt_path in self._ckptPaths.items():
            if not os.path.isfile(ckpt_path):
                slicer.util.errorDisplay(f"Checkpoint not found for {ckpt_key}:\n{ckpt_path}")
                return

        # Remove old markups
        for key, node in self.markupNodes.items():
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
                self.markupNodes[key] = None

        self._inferTmpDir = tempfile.mkdtemp(prefix="fla_infer_")
        inference_start_time = time.time()

        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.runDetectionBtn.enabled = False
        self.detectionStatusLabel.setText("⏳ در حال اجرای مدل...")
        self.detectionStatusLabel.setStyleSheet("color: blue; font-weight: bold;")
        slicer.app.processEvents()

        python_bin = sys.executable
        success = True
        
        packages_dir = get_packages_dir()
        packages_dir_clean = packages_dir.replace('\\', '/')
        models_dir_clean = os.path.dirname(os.path.dirname(infer_script)).replace('\\', '/')
        models_scripts_dir_clean = os.path.dirname(infer_script).replace('\\', '/')
        infer_script_clean = infer_script.replace('\\', '/')

        for step_idx, viewKey in enumerate(self.VIEW_KEYS):
            viewCode = self.VIEW_CODES[viewKey]
            imagePath = self.imagePaths[viewKey]

            actual_input_path = imagePath
            if viewKey == 'right':
                flipped_path = os.path.join(self._inferTmpDir, f'right_flipped.jpg')
                actual_input_path = self._flipImageHorizontally(imagePath, flipped_path)

            if viewCode == 'F':
                coarse_ckpt, fine_ckpt = self._ckptPaths['f_coarse'], self._ckptPaths['f_fine']
            elif viewCode == 'L':
                coarse_ckpt, fine_ckpt = self._ckptPaths['l_coarse'], self._ckptPaths['l_fine']
            else:
                coarse_ckpt, fine_ckpt = self._ckptPaths['s_coarse'], self._ckptPaths['s_fine']

            # 1. Build the argument list for infer.py
            args_list = [
                infer_script_clean,
                '--image', actual_input_path.replace('\\', '/'),
                '--view', viewCode,
                '--coarse', coarse_ckpt.replace('\\', '/'),
                '--fine', fine_ckpt.replace('\\', '/'),
                '--out_dir', self._inferTmpDir.replace('\\', '/'),
            ]

            if viewCode == 'S' and self._presenceThresh.strip():
                args_list.extend(['--presence-threshold', self._presenceThresh.strip()])

            # 2. Inject isolated packages & models paths before running infer.py
            bootstrap_code = f"""import sys, os, runpy
sys.path = [p for p in sys.path if 'site-packages' not in p.lower()]
sys.path.insert(0, '{packages_dir_clean}')
sys.path.insert(0, '{models_dir_clean}')
sys.path.insert(0, '{models_scripts_dir_clean}')
if sys.platform == 'win32':
    t_lib = os.path.join('{packages_dir_clean}', 'torch', 'lib')
    if os.path.isdir(t_lib):
        os.add_dll_directory(t_lib)
sys.argv = {repr(args_list)}
runpy.run_path('{infer_script_clean}', run_name='__main__')
"""

            cmd = [python_bin, "-c", bootstrap_code]
            logging.info(f"Running inference for {viewKey}")
            self.detectionStatusLabel.setText(f"⏳ در حال پردازش {self.VIEW_LABELS_FA[viewKey]}...")
            slicer.app.processEvents()

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=self._buildInferEnv(),
                    cwd=os.path.dirname(infer_script),
                )
                if result.returncode != 0:
                    logging.error(f"Inference failed for {viewKey}:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                    slicer.util.errorDisplay(f"Inference failed for {viewKey}:\n{result.stderr[:500]}")
                    success = False
                    break
                else:
                    logging.info(f"Inference OK for {viewKey}")
            except Exception as e:
                slicer.util.errorDisplay(f"Error running inference for {viewKey}:\n{e}")
                success = False
                break

            self.progressBar.setValue(step_idx + 1)
            slicer.app.processEvents()
            
        if not success:
            self.runDetectionBtn.enabled = True
            self.progressBar.setVisible(False)
            self.detectionStatusLabel.setText("❌ خطا در اجرای مدل")
            self.detectionStatusLabel.setStyleSheet("color: red; font-weight: bold;")
            return

        # Parse JSON and create markups
        self._parseInferenceResults()
        for viewKey in self.VIEW_KEYS:
            if self.inferenceResults[viewKey] is not None:
                landmarks = self._jsonToLandmarkPositions(viewKey)
                self.createMarkupNode(viewKey, landmarks)

        self.landmarksDetected = True
        self.exportBtn.enabled = True
        self.exportPdfBtn.enabled = True
        self.exportBothBtn.enabled = True
        self.runDetectionBtn.enabled = True
        self.progressBar.setVisible(False)
        self.viewComboBox.setCurrentIndex(0)
        self.showImage('frontal')
        self.updateLandmarkList('frontal')

        elapsed = time.time() - inference_start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_str_fa = to_persian_digits(f"{mins} دقیقه و {secs} ثانیه" if mins > 0 else f"{secs} ثانیه")

        self.detectionStatusLabel.setText(f"✓ لندمارک ها شناسایی شدند در {time_str_fa}.")
        self.detectionStatusLabel.setStyleSheet("color: green; font-weight: bold;")

    def _parseInferenceResults(self):
        """Find and parse JSON files produced by infer.py."""
        if not os.path.isdir(self._inferTmpDir):
            return

        for viewKey in self.VIEW_KEYS:
            viewCode = self.VIEW_CODES[viewKey]
            suffix = f"_{viewCode}.json"

            found = None
            for fname in os.listdir(self._inferTmpDir):
                if fname.endswith(suffix):
                    found = os.path.join(self._inferTmpDir, fname)
                    break

            if found and os.path.isfile(found):
                try:
                    with open(found, 'r') as f:
                        data = json.load(f)
                    self.inferenceResults[viewKey] = data
                    logging.info(f"Parsed {found}: {len(data.get('landmarks', []))} landmarks")
                except Exception as e:
                    logging.error(f"Failed to parse {found}: {e}")
                    self.inferenceResults[viewKey] = None
            else:
                logging.warning(f"No JSON found for {viewKey} (looking for *{suffix})")
                self.inferenceResults[viewKey] = None

    def _jsonToLandmarkPositions(self, viewKey):
        data = self.inferenceResults[viewKey]
        if data is None:
            return {}
        
        result = {}
        for lm in data.get('landmarks', []):
            lm_id = lm['id']
            if lm.get('present', True) and lm.get('x') is not None and lm.get('y') is not None:
                result[lm_id] = (float(lm['x']), float(lm['y']))
        
        if viewKey == 'right' and self.imageSizes.get('right'):
            W = self.imageSizes['right'][0]
            result = {lm_id: (W - x, y) for lm_id, (x, y) in result.items()}
        
        return result

    def createMarkupNode(self, viewKey, landmarkDict):
        """
        landmarkDict: {id: (x_pixel, y_pixel)} — only present landmarks.
        """
        viewIndex = self.VIEW_KEYS.index(viewKey)
        landmark_defs = self.getLandmarksForView(viewIndex)

        markupNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode")
        markupNode.SetName(f"Landmarks_{viewKey}")
        markupNode.CreateDefaultDisplayNodes()

        displayNode = markupNode.GetDisplayNode()
        displayNode.SetSelectedColor(0.0, 1.0, 1.0)
        displayNode.SetColor(0.0, 1.0, 1.0)
        displayNode.SetActiveColor(1.0, 1.0, 0.0)
        displayNode.SetGlyphScale(3.0)
        displayNode.SetTextScale(3.5)
        displayNode.SetGlyphType(slicer.vtkMRMLMarkupsDisplayNode.Cross2D)
        displayNode.SetPointLabelsVisibility(True)
        displayNode.SetPropertiesLabelVisibility(True)
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)
        displayNode.SetVisibility3D(True)
        displayNode.SetSliceProjection(True)
        displayNode.SetSliceProjectionOpacity(1.0)
        displayNode.SetSliceProjectionColor(0.0, 1.0, 1.0)
        displayNode.SetOccludedVisibility(True)

        cp_index = 0
        for num, abbrev in landmark_defs:
            if num not in landmarkDict:
                continue
            x_pixel, y_pixel = landmarkDict[num]

            ras_x = float(x_pixel)
            ras_y = float(y_pixel)

            markupNode.AddControlPoint(vtk.vtkVector3d(ras_x, ras_y, 0.0))
            markupNode.SetNthControlPointLabel(cp_index, f"{num}")
            markupNode.SetNthControlPointDescription(cp_index, abbrev)
            markupNode.SetNthControlPointLocked(cp_index, False)
            cp_index += 1

        markupNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
            lambda caller, event, vk=viewKey: self.onLandmarkModified(vk)
        )
        self.markupNodes[viewKey] = markupNode

    def onLandmarkModified(self, viewKey):
        currentIdx = self.viewComboBox.currentIndex
        if self.getViewKeyFromIndex(currentIdx) == viewKey:
            self.updateLandmarkList(viewKey)

    def onViewChanged(self, index):
        viewKey = self.getViewKeyFromIndex(index)
        if self.imageNodes[viewKey] is None:
            self.landmarkListWidget.setText(
                f"⚠️ {viewKey} image not loaded yet.")
            return
        self.showImage(viewKey)
        self.updateLandmarkList(viewKey)
        self._refreshDeleteCombo(viewKey)

    def updateLandmarkList(self, viewKey):
        if self.markupNodes[viewKey] is None:
            self.landmarkListWidget.setText("No landmarks yet.")
            return
        markupNode = self.markupNodes[viewKey]
        text = f"{'#':<4}{'X (px)':<12}{'Y (px)':<12}\n" + "-" * 30 + "\n"
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            pos = [0.0, 0.0, 0.0]
            markupNode.GetNthControlPointPosition(i, pos)
            text += f"{label:<4}{pos[0]:<12.1f}{pos[1]:<12.1f}\n"
        self.landmarkListWidget.setText(text)

    def extractCoordinates(self, viewKey):
        """Extract landmark coordinates from markup nodes as {id: (x, y)}."""
        coords = {}
        markupNode = self.markupNodes[viewKey]
        if markupNode is None:
            return coords
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            pos = [0.0, 0.0, 0.0]
            markupNode.GetNthControlPointPosition(i, pos)
            try:
                num = int(label)
                coords[num] = (pos[0], pos[1])
            except (ValueError, IndexError):
                pass
        return coords
    
    # ── Export handlers ──
    def onExportExcel(self):
        if not self.landmarksDetected:
            slicer.util.warningDisplay("Please run landmark detection first.")
            return

        coords = {}
        for viewKey in self.VIEW_KEYS:
            coords[viewKey] = self.extractCoordinates(viewKey)

        try:
            ppm = float(self.pixelSizeEdit.text)
        except ValueError:
            ppm = 1.0

        defaultName = f"{self.patientNameEdit.text or 'patient'}_Outcome.xlsx"
        filePath = qt.QFileDialog.getSaveFileName(
            self.parent, "Save Excel File",
            os.path.join(os.path.expanduser("~"), defaultName),
            "Excel Files (*.xlsx)"
        )
        if not filePath:
            return

        self.logic.exportToExcel(
            coords, self.imagePaths, ppm, filePath,
            self.patientNameEdit.text,
            self.doctorNameEdit.text,
            self.dateEdit.text
        )
        slicer.util.infoDisplay(f"✓ فایل اکسل ذخیره شد!\n\n{filePath}")

    def onExportPDF(self):
        """Export a PDF report."""
        if not self.landmarksDetected:
            slicer.util.warningDisplay("Please run landmark detection first.")
            return

        coords = {}
        for viewKey in self.VIEW_KEYS:
            coords[viewKey] = self.extractCoordinates(viewKey)

        try:
            ppm = float(self.pixelSizeEdit.text)
        except ValueError:
            ppm = 1.0

        defaultName = f"{self.patientNameEdit.text or 'patient'}_Report.pdf"
        filePath = qt.QFileDialog.getSaveFileName(
            self.parent, "Save PDF Report",
            os.path.join(os.path.expanduser("~"), defaultName),
            "PDF Files (*.pdf)"
        )
        if not filePath:
            return

        try:
            self.logic.exportToPDF(
                coords, self.imagePaths, ppm, filePath,
                self.patientNameEdit.text,
                self.doctorNameEdit.text,
                self.dateEdit.text
            )
            slicer.util.infoDisplay(f"✓ گزارش PDF ذخیره شد!\n\n{filePath}")
        except Exception as e:
            logging.error(f"PDF export failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            slicer.util.errorDisplay(f"خطا در ذخیره PDF:\n{str(e)}")

    def onExportBoth(self):
        """Export both Excel and PDF."""
        if not self.landmarksDetected:
            slicer.util.warningDisplay("Please run landmark detection first.")
            return

        dirPath = qt.QFileDialog.getExistingDirectory(
            self.parent, "Choose Export Directory",
            os.path.expanduser("~")
        )
        if not dirPath:
            return

        coords = {}
        for viewKey in self.VIEW_KEYS:
            coords[viewKey] = self.extractCoordinates(viewKey)

        try:
            ppm = float(self.pixelSizeEdit.text)
        except ValueError:
            ppm = 1.0

        patient = self.patientNameEdit.text or 'patient'
        xlsx_path = os.path.join(dirPath, f"{patient}_Outcome.xlsx")
        pdf_path = os.path.join(dirPath, f"{patient}_Report.pdf")

        try:
            self.logic.exportToExcel(
                coords, self.imagePaths, ppm, xlsx_path,
                self.patientNameEdit.text,
                self.doctorNameEdit.text,
                self.dateEdit.text
            )
            self.logic.exportToPDF(
                coords, self.imagePaths, ppm, pdf_path,
                self.patientNameEdit.text,
                self.doctorNameEdit.text,
                self.dateEdit.text
            )
            slicer.util.infoDisplay(
                f"✓ هر دو فایل با موفقیت ذخیره شدند!\n\n"
                f"📊 Excel: {xlsx_path}\n\n"
                f"📄 PDF: {pdf_path}"
            )
        except Exception as e:
            logging.error(f"Export failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            slicer.util.errorDisplay(f"خطا در ذخیره:\n{str(e)}")

#
# Logic
#
#
# Logic
#
class FacialLandmarkAnalysisLogic(ScriptedLoadableModuleLogic): # type: ignore
    
    VIEW_KEYS = ['frontal', 'right', 'left', 'smile']
    
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self) # type: ignore

    # ===== Math =====
    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @staticmethod
    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    @staticmethod
    def angle3(p1, vertex, p2):
        v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
        v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        m1 = math.sqrt(v1[0]**2 + v1[1]**2)
        m2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if m1 == 0 or m2 == 0:
            return 0
        cos_a = max(-1, min(1, dot / (m1 * m2)))
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def pt_line_dist(point, line_p1, line_p2):
        dx = line_p2[0] - line_p1[0]
        dy = line_p2[1] - line_p1[1]
        L = math.sqrt(dx*dx + dy*dy)
        if L == 0:
            return 0
        return ((point[0] - line_p1[0]) * dy - (point[1] - line_p1[1]) * dx) / L

    # =============================================
    # Row builder — 3 columns (index, measurement, interpretation)
    # =============================================
    def _row(self, index, measurement, interpretation, is_header=False):
        return {
            'ایندکس': index,
            'اندازه گیری': measurement,
            'تفسیر کلینیکی': interpretation,
            '_is_header': is_header,
        }

    def _section(self, title):
        """Section header row (rendered as merged, highlighted)."""
        return {
            'ایندکس': title,
            'اندازه گیری': '',
            'تفسیر کلینیکی': '',
            '_is_section': True,
        }

    # =============================================
    # FRONTAL rows
    # =============================================
    def buildFrontalRows(self, F, ppm):
        rows = []

        # ========== قرینگی افقی صورت ==========
        if 1 in F and 7 in F:
            mid_x = (F[1][0] + F[7][0]) / 2
            x_bar = abs(F[7][0] - F[1][0]) / 2 / ppm  # x̄ = |X7-X1|:2

            # Header row explaining the reference formula
            rows.append(self._row("قرینگی افقی صورت",
                                  f"x̄ = |X7-X1|:2 = {x_bar:.2f}",
                                  "", is_header=True))

            # Malar (زدگی گونه) — L15, L16
            if 15 in F and 16 in F:
                d15 = abs(F[15][0] - mid_x) / ppm
                d16 = abs(F[16][0] - mid_x) / ppm
                if d15 > d16:
                    interp = "گونه راست بیرون زده تر است (پروجکشن مالار راست)/ گونه چپ فرورفته تر است (دفیشنسی مالار چپ)"
                elif abs(d15 - d16) < 1e-6:
                    interp = "پروجکشن قرینه گونه دو طرف"
                else:
                    interp = "گونه چپ بیرون زده تر است (پروجکشن مالار چپ)/ گونه راست فرورفته تر است (دفیشنسی مالار راست)"
                rows.append(self._row("",
                                      f"|X15-X̄| = {d15:.2f} | |X16-X̄| = {d16:.2f}",
                                      interp))

            # Ala of nose — L17, L18
            if 17 in F and 18 in F:
                d17 = abs(F[17][0] - mid_x) / ppm
                d18 = abs(F[18][0] - mid_x) / ppm
                if d17 > d18:
                    interp = "الا بینی راست پهن تر است/ الا بینی چپ باریک تر است/ احتمال سابقه شکاف لب و کام قبلی"
                elif abs(d17 - d18) < 1e-6:
                    interp = "عرض قرینه الا بینی دو طرف"
                else:
                    interp = "الا بینی چپ پهن تر است/ الا بینی راست باریک تر است/ احتمال سابقه شکاف لب و کام قبلی"
                rows.append(self._row("",
                                      f"|X17-X̄| = {d17:.2f} | |X18-X̄| = {d18:.2f}",
                                      interp))

            # Cheilion (گوشه لب) — L19, L20
            if 19 in F and 20 in F:
                d19 = abs(F[19][0] - mid_x) / ppm
                d20 = abs(F[20][0] - mid_x) / ppm
                if d19 > d20:
                    interp = "عرض دهان در سمت راست بیشتر است/ عرض دهان در سمت چپ کمتر است/ انحراف مندیبل به سمت راست/ احتمال سابقه شکاف لب و کام قبلی"
                elif abs(d19 - d20) < 1e-6:
                    interp = "عرض قرینه کامیشور دهان دو طرف"
                else:
                    interp = "عرض دهان در سمت چپ بیشتر است/ عرض دهان در سمت راست کمتر است/ انحراف مندیبل به سمت چپ/ احتمال سابقه شکاف لب و کام قبلی"
                rows.append(self._row("",
                                      f"|X19-X̄| = {d19:.2f} | |X20-X̄| = {d20:.2f}",
                                      interp))

            # Gonial — L22, L23
            if 22 in F and 23 in F:
                d22 = abs(F[22][0] - mid_x) / ppm
                d23 = abs(F[23][0] - mid_x) / ppm
                if d22 > d23:
                    interp = "انگل راست بیرون زده تر است/ انگل چپ فرورفته تر است/ انحراف مندیبل به سمت راست"
                elif abs(d22 - d23) < 1e-6:
                    interp = "پروجکشن قرینه انگل دو طرف"
                else:
                    interp = "انگل چپ بیرون زده تر است/ انگل راست فرورفته تر است/ انحراف مندیبل به سمت چپ"
                rows.append(self._row("",
                                      f"|X22-X̄| = {d22:.2f} | |X23-X̄| = {d23:.2f}",
                                      interp))

            # Menton — L24
            if 24 in F:
                d24 = F[24][0] - mid_x  # signed
                if d24 > 1e-6:
                    interp = "انحراف چانه یا مندیبل به سمت راست"
                elif abs(d24) < 1e-6:
                    interp = "چانه انحراف ندارد"
                else:
                    interp = "انحراف چانه یا مندیبل به سمت چپ"
                rows.append(self._row("",
                                      f"X24 - X̄ = {d24/ppm:.2f}",
                                      interp))

        # ========== قرینگی عمودی صورت ==========
        if 1 in F and 7 in F:
            y_bar_ref = (F[1][1] + F[7][1]) / 2
            y_bar_val = abs(F[7][1] - F[1][1]) / 2 / ppm  # Ȳ = |Y7-Y1|:2

            rows.append(self._row("قرینگی عمودی صورت",
                                  f"Ȳ = |Y7-Y1|:2 = {y_bar_val:.2f}",
                                  "", is_header=True))

            # L15, L16 — malar height
            if 15 in F and 16 in F:
                dy15 = abs(F[15][1] - y_bar_ref) / ppm
                dy16 = abs(F[16][1] - y_bar_ref) / ppm
                if dy15 < dy16:
                    interp = "گونه سمت راست فوقانی تر از چپ است/ دفیشنسی گونه چپ/ کنت ماگزیلا در سمت راست"
                elif abs(dy15 - dy16) < 1e-6:
                    interp = "ارتفاع قرینه گونه دو سمت"
                else:
                    interp = "گونه سمت چپ فوقانی تر از راست است/ دفیشنسی گونه راست/ کنت ماگزیلا در سمت چپ"
                rows.append(self._row("",
                                      f"|Y15-Ȳ| = {dy15:.2f} | |Y16-Ȳ| = {dy16:.2f}",
                                      interp))

            # L17, L18 — ala height
            if 17 in F and 18 in F:
                dy17 = abs(F[17][1] - y_bar_ref) / ppm
                dy18 = abs(F[18][1] - y_bar_ref) / ppm
                if dy17 < dy18:
                    interp = "الا بینی راست فوقانی تر از چپ است/ کنت ماگزیلا در سمت راست/ احتمال سابقه شکاف لب و کام قبلی"
                elif abs(dy17 - dy18) < 1e-6:
                    interp = "ارتفاع قرینه الا بینی دو سمت"
                else:
                    interp = "الا بینی چپ فوقانی تر از راست است/ کنت ماگزیلا در سمت چپ/ احتمال سابقه شکاف لب و کام قبلی"
                rows.append(self._row("",
                                      f"|Y17-Ȳ| = {dy17:.2f} | |Y18-Ȳ| = {dy18:.2f}",
                                      interp))

            # L19, L20 — commissure height
            if 19 in F and 20 in F:
                dy19 = abs(F[19][1] - y_bar_ref) / ppm
                dy20 = abs(F[20][1] - y_bar_ref) / ppm
                if dy19 < dy20:
                    interp = "کامیشور راست فوقانی تر از چپ است/ کنت ماگزیلا در سمت راست/ احتمال سابقه شکاف لب و کام قبلی"
                elif abs(dy19 - dy20) < 1e-6:
                    interp = "ارتفاع قرینه کامیشور دو سمت/ فقدان کنت ماگزیلا"
                else:
                    interp = "کامیشور چپ فوقانی تر از راست است/ کنت ماگزیلا در سمت چپ/ احتمال سابقه شکاف لب و کام قبلی"
                rows.append(self._row("",
                                      f"|Y19-Ȳ| = {dy19:.2f} | |Y20-Ȳ| = {dy20:.2f}",
                                      interp))

            # L22, L23 — gonial height
            if 22 in F and 23 in F:
                dy22 = abs(F[22][1] - y_bar_ref) / ppm
                dy23 = abs(F[23][1] - y_bar_ref) / ppm
                if dy22 < dy23:
                    interp = "انگل راست فوقانی تر از چپ است/ انحراف مندیبل به سمت راست"
                elif abs(dy22 - dy23) < 1e-6:
                    interp = "ارتفاع قرینه انگل دو سمت"
                else:
                    interp = "انگل چپ فوقانی تر از راست است/ انحراف مندیبل به سمت چپ"
                rows.append(self._row("",
                                      f"|Y22-Ȳ| = {dy22:.2f} | |Y23-Ȳ| = {dy23:.2f}",
                                      interp))

        # ========== نسبت عرض گونه به عرض گونیال ==========
        if all(k in F for k in [15, 16, 22, 23]):
            zy_w = abs(F[16][0] - F[15][0]) / ppm  # زایگوماتیک
            go_w = abs(F[23][0] - F[22][0]) / ppm  # گونیال
            ratio = (go_w / zy_w) * 100 if zy_w != 0 else 0
            if 70 <= ratio <= 75:
                interp = "نسبت نرمال عرض بای گونیال به عرض بای زایگوماتیک"
            elif ratio > 75:
                interp = "نسبت عرض بای گونیال به عرض بای زایگوماتیک بیشتر از نرمال/ فرم صورت مربعی/ دفیشنسی ماگزیلا یا میدفیس/ هایپرتروفی عضله ماستر"
            else:
                interp = "نسبت عرض بای گونیال به عرض بای زایگوماتیک کمتر از نرمال/ فرم صورت لانگ فیس/ دفیشنسی مندیبل"
            rows.append(self._row("نسبت عرض گونه به عرض گونیال",
                                  f"|X23-X22|:|X16-X15| = {ratio:.1f}%",
                                  interp))

        # ========== یک پنجم های عمودی ==========
        if all(k in F for k in [3, 4, 9, 10, 13, 14]):
            s1 = abs(F[4][0] - F[13][0]) / ppm   # right ear→lat canthus (R)
            s2 = abs(F[3][0] - F[4][0]) / ppm    # right eye width
            s3 = abs(F[9][0] - F[3][0]) / ppm    # inter-canthal
            s4 = abs(F[10][0] - F[9][0]) / ppm   # left eye width
            s5 = abs(F[14][0] - F[10][0]) / ppm  # left canthus→ear

            measurement = (f"|X4-X13|={s1:.2f} | |X3-X4|={s2:.2f} | "
                           f"|X9-X3|={s3:.2f} | |X10-X9|={s4:.2f} | |X14-X10|={s5:.2f}")

            tol = 0.05 * max(s1, s2, s3, s4, s5)
            equal_all = all(abs(a - b) <= tol for a, b in
                            [(s1, s2), (s1, s3), (s1, s4), (s1, s5)])
            if equal_all:
                interp = "نرمال"
            elif s1 > max(s2, s3, s4, s5) + tol:
                interp = "پروترورژن گوش راست"
            elif s5 > max(s1, s2, s3, s4) + tol:
                interp = "پروترورژن گوش چپ"
            elif s3 > max(s1, s2, s4, s5) + tol:
                interp = "هایپرتلوریسم"
            elif s3 < min(s1, s2, s4, s5) - tol:
                interp = "هایپوتلوریسم"
            else:
                interp = "عدم تقارن یک پنجم های عمودی"
            rows.append(self._row("یک پنجم های عمودی", measurement, interp))

        # ========== عرض بینی ==========
        if all(k in F for k in [3, 9, 17, 18]):
            nose_w = abs(F[18][0] - F[17][0]) / ppm
            ic_w = abs(F[9][0] - F[3][0]) / ppm
            diff = nose_w - ic_w
            tol = 0.05 * ic_w
            if abs(diff) <= tol:
                interp = "نرمال"
            elif diff > tol:
                interp = "بیس بینی پهن است/ احتمال سابقه شکاف لب و کام قبلی"
            else:
                interp = "بیس بینی باریک است"
            rows.append(self._row("عرض بینی",
                                  f"|X18-X17|={nose_w:.2f} | |X9-X3|={ic_w:.2f}",
                                  interp))

        # ========== عرض دهان ==========
        if all(k in F for k in [2, 8, 19, 20]):
            mouth_w = abs(F[20][0] - F[19][0]) / ppm
            iris_w = abs(F[8][0] - F[2][0]) / ppm
            diff = mouth_w - iris_w
            tol = 0.05 * iris_w
            if abs(diff) <= tol:
                interp = "نرمال"
            elif diff > tol:
                interp = "عرض دهان بیشتر از نرمال است"
            else:
                interp = "عرض دهان کمتر از نرمال است"
            rows.append(self._row("عرض دهان",
                                  f"|X20-X19|={mouth_w:.2f} | |X8-X2|={iris_w:.2f}",
                                  interp))

        # ========== نمایش اسکرا ==========
        if all(k in F for k in [5, 6, 11, 12]):
            y5, y6, y11, y12 = F[5][1], F[6][1], F[11][1], F[12][1]
            cond_right_ok = y6 <= y5
            cond_left_ok = y12 <= y11
            if cond_right_ok and cond_left_ok:
                interp = "نرمال"
            elif not cond_right_ok and cond_left_ok:
                interp = "نمایش اسکرا در سمت راست/ اکتروپیون پلک پایین راست/ دفی شنسی ماگزیلا و میدفیس"
            elif cond_right_ok and not cond_left_ok:
                interp = "نمایش اسکرا در سمت چپ/ اکتروپیون پلک پایین چپ/ دفی شنسی ماگزیلا و میدفیس"
            else:
                interp = "نمایش اسکرا در هر دو سمت/ اکتروپیون دو طرفه/ دفی شنسی ماگزیلا و میدفیس"
            rows.append(self._row("نمایش اسکرا",
                                  f"Y5={y5:.1f} | Y6={y6:.1f} | Y11={y11:.1f} | Y12={y12:.1f}",
                                  interp))

        # ========== کنت ==========
        if all(k in F for k in [1, 7, 19, 20]):
            y_bar_val = abs(F[7][1] - F[1][1]) / 2 / ppm
            y_bar_ref = (F[1][1] + F[7][1]) / 2
            d19 = abs(F[19][1] - y_bar_ref) / ppm
            d20 = abs(F[20][1] - y_bar_ref) / ppm

            rows.append(self._row("کنت",
                                  f"Ȳ = |Y7-Y1|:2 = {y_bar_val:.2f}",
                                  "", is_header=True))

            if abs(d19 - d20) < 1e-6:
                interp = "ماگزیلا کنت ندارد"
            elif d19 < d20:
                interp = "کنت ماگزیلا در سمت راست"
            else:
                interp = "کنت ماگزیلا در سمت چپ"
            rows.append(self._row("",
                                  f"|Y19-Ȳ| = {d19:.2f} | |Y20-Ȳ| = {d20:.2f}",
                                  interp))

        return rows

    # =============================================
    # SMILE rows
    # =============================================
    def buildSmileRows(self, S, ppm):
        rows = []

        # ========== میدلاین دندانی ماگزیلا به صورت ==========
        if all(k in S for k in [1, 2, 6]):
            mid_face = (S[1][0] + S[2][0]) / 2  # |X2-X1|:2 reference
            x6 = S[6][0]
            tol = 0.5  # px tolerance
            if abs(x6 - mid_face) <= tol:
                interp = "میدلاین دندانی ماگزیلا نسبت به میدلاین صورت on است"
            elif x6 > mid_face:
                interp = "انحراف میدلاین دندانی ماگزیلا نسبت به صورت به سمت راست"
            else:
                interp = "انحراف میدلاین دندانی ماگزیلا نسبت به صورت به سمت چپ"
            rows.append(self._row("میدلاین دندانی ماگزیلا به صورت",
                                  f"X6={x6/ppm:.2f} | (X1+X2)/2={mid_face/ppm:.2f}",
                                  interp))

        # ========== میدلاین دندانی مندیبل به چانه ==========
        if all(k in S for k in [4, 7]):
            x4, x7 = S[4][0], S[7][0]
            tol = 0.5
            if abs(x4 - x7) <= tol:
                interp = "میدلاین دندانی مندیبل نسبت به چانه on است"
            elif x4 < x7:
                interp = "انحراف میدلاین دندانی مندیبل نسبت به چانه به سمت راست/ انحراف چانه به سمت چپ"
            else:
                interp = "انحراف میدلاین دندانی مندیبل نسبت به چانه به سمت چپ/ انحراف چانه به سمت راست"
            rows.append(self._row("میدلاین دندانی مندیبل به چانه",
                                  f"X4={x4/ppm:.2f} | X7={x7/ppm:.2f}",
                                  interp))

        # ========== میدلاین دندانی ماگزیلا به مندیبل ==========
        if all(k in S for k in [6, 7]):
            x6, x7 = S[6][0], S[7][0]
            tol = 0.5
            if abs(x6 - x7) <= tol:
                interp = "میدلاین دندانی ماگزیلا و مندیبل نسبت به هم on است"
            elif x6 < x7:
                interp = "انحراف میدلاین دندانی ماگزیلا و مندیبل نسبت به هم (میدلاین دندانی ماگزیلا به سمت چپ/ میدلاین دندانی مندیبل به سمت راست)"
            else:
                interp = "انحراف میدلاین دندانی ماگزیلا و مندیبل نسبت به هم (میدلاین دندانی ماگزیلا به سمت راست/ میدلاین دندانی مندیبل به سمت چپ)"
            rows.append(self._row("میدلاین دندانی ماگزیلا به مندیبل",
                                  f"X6={x6/ppm:.2f} | X7={x7/ppm:.2f}",
                                  interp))

        # ========== نمایش دندان ==========
        if 8 in S and 6 in S:
            val = abs(S[8][1] - S[6][1]) / ppm
            rows.append(self._row("نمایش دندان",
                                  f"|Y8-Y6| = {val:.2f}",
                                  "نمایش کامل تاج دندان در لبخند (نشانه vertical maxillary excess/ طول لب کوتاه)"))
        elif 6 in S and 3 in S:
            val = abs(S[3][1] - S[6][1]) / ppm
            rows.append(self._row("نمایش دندان",
                                  f"|Y3-Y6| = {val:.2f}",
                                  "مقادیر بیشتر نشانه نمایش بیشتر دندان در لبخند است"))
        else:
            rows.append(self._row("نمایش دندان",
                                  "0",
                                  "عدم نمایش دندان در لبخند/ دفیشنسی عمودی ماگزیلا/ دفیشنسی قدامی خلفی ماگزیلا/ طول لب بلند"))

        # ========== نمایش لثه ==========
        if 8 in S and 3 in S:
            val = abs(S[3][1] - S[8][1]) / ppm
            rows.append(self._row("نمایش لثه",
                                  f"|Y3-Y8| = {val:.2f}",
                                  "نمایش لثه در لبخند (نشانه vertical maxillary excess/ کمبود طول تاج کلینیکی/ رشد بیش از حد لثه/ طول لب کوتاه)"))
        else:
            rows.append(self._row("نمایش لثه",
                                  "0",
                                  "عدم نمایش لثه در لبخند"))

        return rows

    # =============================================
    # PROFILE rows (lateral view — right or left)
    # =============================================
    def buildProfileRows(self, L, ppm, gender='male'):
        rows = []

        # ========== یک سوم های افقی ==========
        if all(k in L for k in [1, 6, 11, 15]):
            d_upper = abs(L[15][1] - L[1][1]) / ppm  # Y15-Y1
            d_mid   = abs(L[1][1]  - L[6][1]) / ppm  # Y1-Y6
            d_lower = abs(L[6][1]  - L[11][1]) / ppm # Y6-Y11
            tol = 0.05 * max(d_upper, d_mid, d_lower)

            if abs(d_upper - d_mid) <= tol and abs(d_mid - d_lower) <= tol:
                interp = "نرمال"
            elif d_upper > d_mid + tol and d_upper > d_lower + tol:
                interp = "ارتفاع یک سوم فوقانی صورت بیشتر از یک سوم میانی و تحتانی (دفیشنسی عمودی ماگزیلا و مندیبل/ الگوی رشد short face/ خط رویش موی عقب رفته/ ارتفاع بلند پیشانی)"
            elif d_mid > d_upper + tol and d_mid > d_lower + tol:
                interp = "ارتفاع یک سوم میانی صورت بیشتر از یک سوم فوقانی و تحتانی (بلند بودن میدفیس/ بلند بودن طول بینی)"
            elif d_lower > d_upper + tol and d_lower > d_mid + tol:
                interp = "ارتفاع یک سوم تحتانی صورت بیشتر از یک سوم فوقانی و میانی (دفیشنسی عمودی میدفیس/ الگوی رشد عمودی long face و هایپردایورجنت)"
            else:
                interp = "عدم تقارن قابل توجه یک‌سوم‌های افقی"
            rows.append(self._row("یک سوم های افقی",
                                  f"|Y15-Y1|={d_upper:.2f} | |Y1-Y6|={d_mid:.2f} | |Y6-Y11|={d_lower:.2f}",
                                  interp))

        # ========== یک سوم تحتانی (نسبت لب بالا/تحتانی) ==========
        if all(k in L for k in [6, 11, 16]):
            upper_lip = abs(L[6][1] - L[16][1]) / ppm   # |Y6-Y16|
            lower_face = abs(L[6][1] - L[11][1]) / ppm  # |Y6-Y11|
            lower_lip = abs(L[16][1] - L[11][1]) / ppm  # |Y16-Y11|
            r1 = upper_lip / lower_face if lower_face else 0
            r2 = lower_lip / lower_face if lower_face else 0

            # نسبت طبیعی 1:3 و 2:3
            if abs(r1 - 1/3) < 0.05 and abs(r2 - 2/3) < 0.05:
                interp = "نرمال"
            elif r1 < 1/3 - 0.05:
                interp = "طول لب بالا کوتاه نسبت به یک سوم تحتانی صورت/ ارتفاع تحتانی افزایش یافته قدام صورت/ mandibular excess"
            elif r1 > 1/3 + 0.05:
                interp = "طول لب بالا بلند نسبت به یک سوم تحتانی صورت/ کاهش ارتفاع یک سوم تحتانی صورت/ دفیشنسی مندیبل"
            elif r2 < 2/3 - 0.05:
                interp = "طول لب پایین کوتاه نسبت به یک سوم تحتانی صورت/ افزایش ارتفاع یک سوم تحتانی صورت/ چانه بلند"
            else:
                interp = "طول لب پایین بلند نسبت به یک سوم تحتانی صورت"
            rows.append(self._row("یک سوم تحتانی",
                                  f"|Y6-Y16|:|Y6-Y11|={r1:.3f} | |Y16-Y11|:|Y6-Y11|={r2:.3f}",
                                  interp))

        # ========== زاویه نازوفرونتال ==========
        if all(k in L for k in [1, 2, 3]):
            angle = self.angle3(L[1], L[2], L[3])
            if 125 <= angle <= 135:
                interp = "نرمال"
            elif angle < 125:
                interp = "برجستگی بیشتر گلابلا/ پروجکشن بیشتر بینی/ low radix"
            else:
                interp = "برجستگی کمتر گلابلا/ پروجکشن کمتر بینی"
            rows.append(self._row("زاویه نازوفرونتال",
                                  f"{angle:.2f}°",
                                  interp))

        # ========== طول بینی ==========
        if all(k in L for k in [1, 2, 3, 4]):
            nose_len = self.dist(L[3], L[2]) / ppm
            mid_face = self.dist(L[4], L[1]) / ppm
            ratio = (nose_len / mid_face * 100) if mid_face else 0
            if abs(ratio - 67) <= 3:
                interp = "نرمال"
            elif ratio < 64:
                interp = "طول بینی کوتاه تر نسبت به یک سوم میانی صورت"
            else:
                interp = "طول بینی بلند تر نسبت به یک سوم میانی صورت"
            rows.append(self._row("طول بینی",
                                  f"√((X3-X2)²+(Y3-Y2)²) : √((X4-X1)²+(Y4-Y1)²) = {ratio:.1f}%",
                                  interp))

        # ========== پروجکشن بینی ==========
        if all(k in L for k in [3, 4, 6]):
            num = abs(L[3][0] - L[6][0]) / ppm
            den = abs(L[6][0] - L[4][0]) / ppm
            ratio = num / den if den else 0
            if abs(ratio - 2) <= 0.15:
                interp = "نرمال"
            elif ratio < 2:
                interp = "پروجکشن کم بینی/ دفیشنسی میدفیس"
            else:
                interp = "پروجکشن زیاد بینی"
            rows.append(self._row("پروجکشن بینی",
                                  f"|X3-X6|:|X6-X4| = {ratio:.3f}",
                                  interp))

        # ========== زاویه نازولیبیال ==========
        if all(k in L for k in [5, 6, 7]):
            angle = self.angle3(L[5], L[6], L[7])
            if gender == 'female':
                lo, hi = 90, 110
            else:
                lo, hi = 90, 95
            if lo <= angle <= hi:
                interp = "نرمال"
            elif angle > hi:
                interp = "ساپورت کم لب بالا/ رتروژن دندان های قدامی ماگزیلا/ کاهش بعد قدامی خلفی ماگزیلا/ روتیشن نوک بینی"
            else:
                interp = "ساپورت زیاد لب بالا/ پروتروژن دندان های قدامی ماگزیلا/ افزایش بعد قدامی خلفی ماگزیلا/ افتادگی نوک بینی"
            rows.append(self._row("زاویه نازولیبیال",
                                  f"{angle:.2f}°",
                                  interp))

        # ========== پروجکشن لب بالا به لب پایین ==========
        if all(k in L for k in [7, 8]):
            diff = (L[7][0] - L[8][0]) / ppm
            # NOTE: sign convention depends on right vs left profile
            if diff > 0:
                interp = "لب بالا جلوتر از لب پایین است/ تمایل به رابطه اسکلتال کلاس یک یا دو"
            else:
                interp = "لب پایین جلوتر از لب بالا است/ تمایل به رابطه اسکلتال کلاس سه"
            rows.append(self._row("پروجکشن لب بالا به لب پایین",
                                  f"X7-X8 = {diff:.2f}",
                                  interp))

        # ========== پروجکشن لب بالا و پایین نسبت به صورت (E-line) ==========
        if all(k in L for k in [5, 7, 8, 10]):
            d7 = self.pt_line_dist(L[7], L[5], L[10]) / ppm
            d8 = self.pt_line_dist(L[8], L[5], L[10]) / ppm

            if d7 > 0.5:
                interp7 = "لب بالا بیرون زده تر از حد نرمال/ تمایل به رابطه اسکلتال کلاس دو"
            elif d7 < -0.5:
                interp7 = "لب بالا عقب تر از حد نرمال/ تمایل به رابطه اسکلتال کلاس سه"
            else:
                interp7 = "لب بالا در حد نرمال نسبت به E-line"

            if d8 > 0.5:
                interp8 = "لب پایین بیرون زده تر از حد نرمال/ تمایل به رابطه اسکلتال کلاس سه"
            elif d8 < -0.5:
                interp8 = "لب پایین عقب تر از حد نرمال/ تمایل به رابطه اسکلتال کلاس دو"
            else:
                interp8 = "لب پایین در حد نرمال نسبت به E-line"

            rows.append(self._row("پروجکشن لب بالا و پایین نسبت به صورت",
                                  f"فاصله X7 از خط 5-10 = {d7:.2f}",
                                  interp7))
            rows.append(self._row("",
                                  f"فاصله X8 از خط 5-10 = {d8:.2f}",
                                  interp8))

        # ========== زاویه منتولیبیال ==========
        if all(k in L for k in [8, 9, 10]):
            angle = self.angle3(L[8], L[9], L[10])
            if 110 <= angle <= 130:
                interp = "نرمال"
            elif angle < 110:
                interp = "نازولیبیال فولد عمیق و حاده/ پروتروژن و eversion لب/ پروتروژن چانه"
            else:
                interp = "نازولیبیال فولد کم عمق و منفرجه/ تمایل به رابطه اسکلتال کلاس دو/ دفیشنسی چانه"
            rows.append(self._row("زاویه منتولیبیال",
                                  f"{angle:.2f}°",
                                  interp))

        # ========== پروجکشن چانه ==========
        if all(k in L for k in [1, 6, 10]):
            raw_angle = self.angle3(L[1], L[6], L[10])
            angle_val = 180 - raw_angle  # 180 - (زاویه بین 1 و 6 و 10)
            if 8 <= angle_val <= 16:
                interp = "نرمال"
            elif angle_val > 16:
                interp = "دفیشنسی چانه/ تمایل به رابطه اسکلتال کلاس دو"
            elif angle_val < 8:
                interp = "پروتروژن چانه/ تمایل به رابطه اسکلتال کلاس سه"
            else:
                interp = "پروتروژن شدید چانه (منفی شود)"
            rows.append(self._row("پروجکشن چانه",
                                  f"180 - زاویه(1,6,10) = {angle_val:.2f}°",
                                  interp))

        # ========== زاویه چانه-گردن ==========
        if all(k in L for k in [11, 12, 13]):
            angle = self.angle3(L[11], L[12], L[13])
            if 90 <= angle <= 110:
                interp = "نرمال/ definition خوب گردن"
            elif angle < 90:
                interp = "زاویه حاده گردن-چانه/ تمایل به رابطه اسکلتال کلاس سه/ شیب زیاد پلن مندیبل"
            else:
                interp = "زاویه منفرجه گردن-چانه/ definition ضعیف گردن/ تمایل به رابطه اسکلتال کلاس دو/ حضور چربی ساب منتال"
            rows.append(self._row("زاویه چانه-گردن",
                                  f"{angle:.2f}°",
                                  interp))

        # ========== زاویه پروفایل صورت ==========
        if all(k in L for k in [1, 6, 10]):
            raw_angle = self.angle3(L[1], L[6], L[10])
            profile_angle = 180 - raw_angle
            # For males normal: -15° to -7°; for females: -17° to -9°
            if gender == 'female':
                low, high = -17, -9
            else:
                low, high = -15, -7

            if profile_angle < low:
                interp = "مقادیر منفی تر به معنی پروفایل صورتی محدب است/ تمایل به رابطه اسکلتال کلاس دو"
            elif profile_angle > high:
                interp = "مقادیر مثبت تر به معنی پروفایل صورتی مقعر است/ تمایل به رابطه اسکلتال کلاس سه"
            else:
                interp = "پروفایل صورتی نرمال و مستقیم است"
            rows.append(self._row("زاویه پروفایل صورت",
                                  f"180 - زاویه(1,6,10) = {profile_angle:.2f}°",
                                  interp))

        return rows

    # =============================================
    # ANNOTATED IMAGE GENERATION
    # =============================================
    def createAnnotatedImage(self, viewKey, imagePath, coords, outPath):
        from PIL import Image, ImageDraw, ImageFont
        
        try:
            img = Image.open(imagePath).convert('RGB')
        except Exception as e:
            logging.error(f"Cannot load image {imagePath}: {e}")
            return False

        draw = ImageDraw.Draw(img)
        W, H = img.size

        try:
            font_size = max(16, int(W / 80))
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()

        LANDMARK_COLOR = (0, 255, 255)
        LINE_COLOR = (255, 255, 0)
        MIDLINE_COLOR = (255, 100, 100)
        AUX_COLOR = (100, 255, 100)

        def draw_vline(x, color=MIDLINE_COLOR, width=2):
            draw.line([(x, 0), (x, H)], fill=color, width=width)

        def draw_hline(y, color=MIDLINE_COLOR, width=2):
            draw.line([(0, y), (W, y)], fill=color, width=width)

        def draw_segment(p1, p2, color=LINE_COLOR, width=2):
            draw.line([p1, p2], fill=color, width=width)

        if viewKey == 'frontal':
            if 1 in coords and 7 in coords:
                mid_x = (coords[1][0] + coords[7][0]) / 2
                draw_vline(mid_x, MIDLINE_COLOR, 4)
                y_line = (coords[1][1] + coords[7][1]) / 2
                draw_hline(y_line, AUX_COLOR, 4)
                draw_segment(coords[1], coords[7], LINE_COLOR, 2)

        elif viewKey == 'smile':
            if 1 in coords and 2 in coords:
                draw_segment(coords[1], coords[2], LINE_COLOR, 3)
                mx = (coords[1][0] + coords[2][0]) / 2
                my = (coords[1][1] + coords[2][1]) / 2
                dx = coords[2][0] - coords[1][0]
                dy = coords[2][1] - coords[1][1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    nx = -dy / length
                    ny = dx / length
                    extent = max(W, H)
                    p1 = (mx + nx * extent, my + ny * extent)
                    p2 = (mx - nx * extent, my - ny * extent)
                    draw_segment(p1, p2, MIDLINE_COLOR, 3)
                    
        elif viewKey in ['right', 'left', 'lateral']:
            if 5 in coords and 10 in coords:
                draw_segment(coords[5], coords[10], LINE_COLOR, 2)
            if 1 in coords and 2 in coords and 3 in coords:
                draw_segment(coords[1], coords[2], AUX_COLOR, 1)
                draw_segment(coords[2], coords[3], AUX_COLOR, 1)
            if 5 in coords and 6 in coords and 7 in coords:
                draw_segment(coords[5], coords[6], AUX_COLOR, 1)
                draw_segment(coords[6], coords[7], AUX_COLOR, 1)
            if 8 in coords and 9 in coords and 10 in coords:
                draw_segment(coords[8], coords[9], AUX_COLOR, 1)
                draw_segment(coords[9], coords[10], AUX_COLOR, 1)
            if 11 in coords and 12 in coords and 13 in coords:
                draw_segment(coords[11], coords[12], AUX_COLOR, 1)
                draw_segment(coords[12], coords[13], AUX_COLOR, 1)
            if 1 in coords and 6 in coords and 10 in coords:
                draw_segment(coords[1], coords[6], MIDLINE_COLOR, 1)
                draw_segment(coords[6], coords[10], MIDLINE_COLOR, 1)

        r = max(4, int(W / 200))
        for num, (x, y) in coords.items():
            draw.line([(x - r*2, y), (x + r*2, y)], fill=LANDMARK_COLOR, width=2)
            draw.line([(x, y - r*2), (x, y + r*2)], fill=LANDMARK_COLOR, width=2)
            draw.text((x + r*2 + 2, y + 2), str(num), fill=LANDMARK_COLOR, font=font)

        img.save(outPath, 'PNG', optimize=True)
        return True

    # =============================================
    # EXCEL EXPORT — 3 columns
    # =============================================
    def exportToExcel(self, coords, imagePaths, ppm, filePath, patientName, doctorName, date):
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()

        HEADER_FONT = Font(name='B Nazanin', bold=True, size=12, color="000000")
        CELL_FONT = Font(name='B Nazanin', size=11)
        INDEX_FONT = Font(name='B Nazanin', bold=True, size=11, color="1F4E78")
        TITLE_FONT = Font(name='B Nazanin', bold=True, size=16, color="2F5496")
        SECTION_FONT = Font(name='B Nazanin', bold=True, size=13, color="C00000")
        SUBHEADER_FONT = Font(name='B Nazanin', bold=True, size=11, color="7F6000")

        HEADER_FILL = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        INDEX_FILL = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
        SECTION_FILL = PatternFill(start_color="FFE699", end_color="FFE699", fill_type="solid")
        SUBHEADER_FILL = PatternFill(start_color="FCE4B5", end_color="FCE4B5", fill_type="solid")

        CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True, readingOrder=2)
        RIGHT = Alignment(horizontal='right', vertical='center', wrap_text=True, readingOrder=2)
        BORDER = Border(
            left=Side(style='thin', color='808080'), right=Side(style='thin', color='808080'),
            top=Side(style='thin', color='808080'), bottom=Side(style='thin', color='808080')
        )

        # Right-to-left column order visually (ایندکس is right-most in RTL sheet)
        COLUMNS = ['ایندکس', 'اندازه گیری', 'تفسیر کلینیکی']

        def buildAnalysisSheet(sheetName, rows, viewKey):
            from openpyxl.drawing.image import Image as XLImage
            from PIL import Image as PILImage
            
            ws = wb.create_sheet(sheetName)
            ws.sheet_view.rightToLeft = True
            
            current_row = 1
            
            # Embed annotated image at top
            if imagePaths.get(viewKey) is not None:
                temp_dir = tempfile.mkdtemp(prefix="fla_xlsx_")
                out_path = os.path.join(temp_dir, f"{viewKey}_annot.png")
                if self.createAnnotatedImage(viewKey, imagePaths[viewKey], coords.get(viewKey, {}), out_path):
                    try:
                        img = PILImage.open(out_path)
                        ow, oh = img.size
                        max_w, max_h = 500, 700
                        scale = min(max_w / ow, max_h / oh)
                        img.thumbnail((int(ow * scale), int(oh * scale)), PILImage.LANCZOS)
                        resized = os.path.join(temp_dir, f"{viewKey}_resized.png")
                        img.save(resized, 'PNG')
                        xl_img = XLImage(resized)
                        xl_img.anchor = f"A{current_row}"
                        ws.add_image(xl_img)
                        rows_for_image = max(28, int(img.size[1] / 20))
                        current_row += rows_for_image + 2
                    except Exception as e:
                        logging.error(f"Image embed failed for {viewKey}: {e}")
            
            # Header row
            for col_idx, col_name in enumerate(COLUMNS, start=1):
                c = ws.cell(row=current_row, column=col_idx, value=col_name)
                c.font = HEADER_FONT
                c.fill = HEADER_FILL
                c.alignment = CENTER
                c.border = BORDER
            header_row = current_row
            current_row += 1
            
            # Data rows
            for row in rows:
                is_section = row.get('_is_section', False)
                is_header = row.get('_is_header', False)
                
                for col_idx, col_name in enumerate(COLUMNS, start=1):
                    value = row.get(col_name, "")
                    c = ws.cell(row=current_row, column=col_idx, value=value)
                    c.alignment = CENTER if col_idx != 3 else RIGHT
                    c.border = BORDER
                    
                    if is_section:
                        c.font = SECTION_FONT
                        c.fill = SECTION_FILL
                        c.alignment = CENTER
                    elif is_header:
                        # Sub-header row: highlight the "measurement" cell (reference formula)
                        if col_idx == 1:
                            c.font = INDEX_FONT
                            c.fill = INDEX_FILL
                        elif col_idx == 2:
                            c.font = SUBHEADER_FONT
                            c.fill = SUBHEADER_FILL
                            c.alignment = CENTER
                        else:
                            c.font = CELL_FONT
                    elif col_idx == 1 and value:
                        c.font = INDEX_FONT
                        c.fill = INDEX_FILL
                    else:
                        c.font = CELL_FONT
                
                if is_section:
                    ws.merge_cells(start_row=current_row, start_column=1,
                                   end_row=current_row, end_column=len(COLUMNS))
                current_row += 1
            
            widths = {1: 32, 2: 42, 3: 60}
            for col, w in widths.items():
                ws.column_dimensions[get_column_letter(col)].width = w
            ws.row_dimensions[header_row].height = 32
            
        default_sheet = wb.active
        wb.remove(default_sheet)
        
        buildAnalysisSheet("Frontal",       self.buildFrontalRows(coords.get('frontal', {}), ppm),        'frontal')
        buildAnalysisSheet("Right Profile", self.buildProfileRows(coords.get('right', {}),   ppm),        'right')
        buildAnalysisSheet("Left Profile",  self.buildProfileRows(coords.get('left', {}),    ppm),        'left')
        buildAnalysisSheet("Smile",         self.buildSmileRows(  coords.get('smile', {}),   ppm),        'smile')
        
        self._buildInformationSheet(wb, patientName, doctorName, date, coords, TITLE_FONT, CELL_FONT, RIGHT)

        wb.save(filePath)
        logging.info(f"Excel saved: {filePath}")

    def _buildInformationSheet(self, wb, patientName, doctorName, date, coords, title_font, cell_font, right):
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        ws = wb.create_sheet("Information")
        ws.sheet_view.rightToLeft = True

        ws['A1'] = "اطلاعات گزارش"
        ws['A1'].font = title_font
        ws['A1'].alignment = right
        ws.merge_cells('A1:D1')
        ws.row_dimensions[1].height = 30

        ws['A3'] = "اطلاعات بیمار"
        ws['A3'].font = Font(name='B Nazanin', bold=True, size=14, color="2F5496")
        ws['A3'].alignment = right
        ws.merge_cells('A3:D3')

        info_data = [
            ("نام بیمار:", patientName),
            ("نام پزشک:", doctorName),
            ("تاریخ:", date),
            ("نرم افزار:", "Facial Landmark Analysis Extension"),
        ]
        for i, (label, value) in enumerate(info_data, start=5):
            c1 = ws.cell(row=i, column=1, value=label)
            c1.font = Font(name='B Nazanin', bold=True, size=12)
            c1.alignment = right
            c2 = ws.cell(row=i, column=2, value=value)
            c2.font = cell_font
            c2.alignment = right

        current_row = len(info_data) + 7
        ws.cell(row=current_row, column=1, value="مختصات لندمارک ها (پیکسل)").font = \
            Font(name='B Nazanin', bold=True, size=14, color="2F5496")
        ws.cell(row=current_row, column=1).alignment = right
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=4)
        current_row += 2

        view_names_fa = {
            'frontal': "نمای روبرو",
            'right': "نمای نیمرخ راست",
            'left': "نمای نیمرخ چپ",
            'smile': "نمای لبخند"
        }

        header_font_small = Font(name='B Nazanin', bold=True, size=11, color="FFFFFF")
        header_fill_small = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        center_align = Alignment(horizontal='center', vertical='center', readingOrder=2)
        border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )

        for viewKey in self.VIEW_KEYS:
            viewCoords = coords.get(viewKey, {})
            if not viewCoords:
                continue

            c = ws.cell(row=current_row, column=1, value=view_names_fa.get(viewKey, viewKey))
            c.font = Font(name='B Nazanin', bold=True, size=12, color="C00000")
            c.alignment = center_align
            c.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
            ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=3)
            current_row += 1

            for col_idx, header in enumerate(["شماره لندمارک", "X (پیکسل)", "Y (پیکسل)"], start=1):
                c = ws.cell(row=current_row, column=col_idx, value=header)
                c.font = header_font_small
                c.fill = header_fill_small
                c.alignment = center_align
                c.border = border
            current_row += 1

            for num in sorted(viewCoords.keys()):
                x, y = viewCoords[num]
                c1 = ws.cell(row=current_row, column=1, value=num)
                c2 = ws.cell(row=current_row, column=2, value=round(x, 2))
                c3 = ws.cell(row=current_row, column=3, value=round(y, 2))
                for c in [c1, c2, c3]:
                    c.font = cell_font
                    c.alignment = center_align
                    c.border = border
                current_row += 1

            current_row += 2

        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 20
        ws.column_dimensions['D'].width = 20

    def _rtl(self, text):
        import arabic_reshaper # type: ignore
        from bidi.algorithm import get_display # type: ignore
        if not text:
            return ""
        try:
            reshaped = arabic_reshaper.reshape(str(text))
            return get_display(reshaped)
        except:
            return str(text)

    def _registerPersianFont(self):
        from reportlab.pdfbase import pdfmetrics # type: ignore
        from reportlab.pdfbase.ttfonts import TTFont # type: ignore
        font_name = 'PersianFont'
        
        font_candidates = [
            r"C:\Windows\Fonts\tahoma.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\BNazanin.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        try:
            font_path = None
            for candidate in font_candidates:
                if os.path.exists(candidate):
                    font_path = candidate
                    break

            if font_path is None:
                return 'Helvetica'

            pdfmetrics.registerFont(TTFont(font_name, font_path))
            return font_name
        except Exception as e:
            logging.error(f"Font registration failed: {e}")
            return 'Helvetica'

    # =============================================
    # PDF EXPORT — 3 columns
    # =============================================
    def exportToPDF(self, coords, imagePaths, ppm, filePath, patientName, doctorName, date):
        from reportlab.lib.pagesizes import A4, landscape # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore
        from reportlab.lib.units import cm # type: ignore
        from reportlab.lib import colors # type: ignore
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT # type: ignore
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak) # type: ignore
        from PIL import Image

        font_name = self._registerPersianFont()
        bold_font = font_name

        # Use landscape orientation to fit 3 columns comfortably
        page_size = landscape(A4)

        doc = SimpleDocTemplate(
            filePath, pagesize=page_size,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=1.5*cm, bottomMargin=1.5*cm,
            title=f"Facial Analysis Report - {patientName}", author=doctorName
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Title'], fontName=bold_font, fontSize=22,
            textColor=colors.HexColor('#1976D2'),
            alignment=TA_CENTER, spaceAfter=20, leading=28
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle', parent=styles['Heading1'], fontName=bold_font, fontSize=16,
            textColor=colors.HexColor('#2F5496'),
            alignment=TA_CENTER, spaceAfter=15, leading=22
        )
        section_style = ParagraphStyle(
            'SectionHeading', parent=styles['Heading2'], fontName=bold_font, fontSize=14,
            textColor=colors.HexColor('#C00000'),
            alignment=TA_RIGHT, spaceAfter=10, spaceBefore=15, leading=20,
            backColor=colors.HexColor('#FFF2CC'), borderPadding=6
        )
        body_style = ParagraphStyle(
            'CustomBody', parent=styles['Normal'], fontName=font_name, fontSize=11,
            textColor=colors.black,
            alignment=TA_RIGHT, leading=16
        )

        story = []
        # --- Cover page ---
        story.append(Spacer(1, 3*cm))
        story.append(Paragraph(self._rtl("گزارش تحلیل لندمارک های صورت"), title_style))
        story.append(Paragraph("Facial Landmark Analysis Report", subtitle_style))
        story.append(Spacer(1, 2*cm))

        info_data = [
            [self._rtl(patientName or "-"), self._rtl(": نام بیمار")],
            [self._rtl(doctorName or "-"), self._rtl(": نام پزشک")],
            [self._rtl(date or "-"), self._rtl(": تاریخ")],
        ]
        info_table = Table(info_data, colWidths=[10*cm, 6*cm])
        info_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), font_name, 12),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#DDEBF7')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#B4C7E7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#B4C7E7')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 4*cm))
        story.append(Paragraph(
            self._rtl("Generated by Facial Landmark Analysis | Engine Developed by Morteza Hajiabadi"),
            body_style
        ))
        story.append(PageBreak())

        # --- Analysis pages ---
        view_order = [
            ('frontal', "نمای روبرو (Frontal)", self.buildFrontalRows, coords.get('frontal', {})),
            ('right',   "نمای نیمرخ راست",       self.buildProfileRows, coords.get('right', {})),
            ('left',    "نمای نیمرخ چپ",         self.buildProfileRows, coords.get('left', {})),
            ('smile',   "نمای لبخند (Smile)",    self.buildSmileRows,   coords.get('smile', {})),
        ]

        temp_dir = tempfile.mkdtemp(prefix="fla_pdf_")

        for viewKey, view_title, rowBuilder, viewCoords in view_order:
            if imagePaths.get(viewKey) is None:
                continue

            story.append(Paragraph(self._rtl(view_title), section_style))
            story.append(Spacer(1, 0.3*cm))

            # Image (smaller since we're in landscape and need room for the wide table)
            out_path = os.path.join(temp_dir, f"{viewKey}_pdf.png")
            if self.createAnnotatedImage(viewKey, imagePaths[viewKey], viewCoords, out_path):
                try:
                    pil_img = Image.open(out_path)
                    ow, oh = pil_img.size
                    max_w, max_h = 10 * cm, 12 * cm
                    aspect = ow / oh
                    if aspect > (max_w / max_h):
                        dw, dh = max_w, max_w / aspect
                    else:
                        dh, dw = max_h, max_h * aspect
                    rl_img = RLImage(out_path, width=dw, height=dh)
                    rl_img.hAlign = 'CENTER'
                    story.append(rl_img)
                    story.append(Spacer(1, 0.4*cm))
                except Exception as e:
                    logging.error(f"PDF image {viewKey} failed: {e}")

            rows = rowBuilder(viewCoords, ppm)
            if rows:
                self._addAnalysisTable(story, rows, font_name, bold_font)

            story.append(PageBreak())

        doc.build(story, onFirstPage=self._pdfFooter, onLaterPages=self._pdfFooter)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

        logging.info(f"PDF saved: {filePath}")

    def _addAnalysisTable(self, story, rows, font_name, bold_font):
        from reportlab.lib.units import cm # type: ignore
        from reportlab.lib import colors # type: ignore
        from reportlab.platypus import (Spacer, Table, TableStyle, Paragraph) # type: ignore
        from reportlab.lib.styles import ParagraphStyle # type: ignore
        from reportlab.lib.enums import TA_RIGHT, TA_CENTER # type: ignore

        # Wrapping style for long clinical interpretations
        wrap_style = ParagraphStyle(
            'Wrap', fontName=font_name, fontSize=8,
            alignment=TA_RIGHT, leading=11, textColor=colors.black,
        )
        wrap_center_style = ParagraphStyle(
            'WrapCenter', fontName=font_name, fontSize=8,
            alignment=TA_CENTER, leading=11, textColor=colors.black,
        )

        def P(text, center=False):
            style = wrap_center_style if center else wrap_style
            return Paragraph(self._rtl(text) if text else "", style)

        # Column order for RTL reading (rightmost first as seen): ایندکس | اندازه گیری | تفسیر کلینیکی
        # In reportlab, arrays are LTR, so first column is left-most on page.
        # For an RTL page we invert: [interpretation, measurement, index]
        headers = [P("تفسیر کلینیکی", center=True),
                   P("اندازه گیری", center=True),
                   P("ایندکس", center=True)]
        table_data = [headers]
        section_rows = []
        header_rows = []

        for i, row in enumerate(rows):
            if row.get('_is_section'):
                clean_title = row.get('ایندکس', '')
                table_data.append(['', '', P(clean_title, center=True)])
                section_rows.append(len(table_data) - 1)
            else:
                is_header = row.get('_is_header', False)
                table_data.append([
                    P(row.get('تفسیر کلینیکی', '')),
                    P(row.get('اندازه گیری', ''), center=True),
                    P(row.get('ایندکس', ''), center=True),
                ])
                if is_header:
                    header_rows.append(len(table_data) - 1)

        # Column widths (landscape A4 usable width ~26cm)
        col_widths = [12*cm, 8*cm, 6*cm]
        table = Table(table_data, colWidths=col_widths, repeatRows=1)

        style_cmds = [
            ('FONT', (0, 0), (-1, -1), font_name, 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFFF00')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]

        for row_idx in section_rows:
            style_cmds.extend([
                ('SPAN', (0, row_idx), (-1, row_idx)),
                ('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#FFE699')),
                ('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.HexColor('#C00000')),
                ('ALIGN', (0, row_idx), (-1, row_idx), 'CENTER'),
            ])

        for row_idx in header_rows:
            # Highlight the "measurement" column (index 1) as the reference formula
            style_cmds.append(('BACKGROUND', (1, row_idx), (1, row_idx), colors.HexColor('#FCE4B5')))

        for i in range(1, len(table_data)):
            if i not in section_rows and i not in header_rows and i % 2 == 0:
                style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F5F5F5')))

        table.setStyle(TableStyle(style_cmds))
        story.append(table)
        story.append(Spacer(1, 0.5*cm))

    def _pdfFooter(self, canvas, doc):
        from reportlab.lib.pagesizes import A4, landscape # type: ignore
        from reportlab.lib.units import cm # type: ignore
        from reportlab.lib import colors # type: ignore
        pw, ph = landscape(A4)
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.setStrokeColor(colors.HexColor('#CCCCCC'))
        canvas.line(1.5*cm, 1*cm, pw - 1.5*cm, 1*cm)
        page_text = f"Page {doc.page}"
        canvas.drawRightString(pw - 1.5*cm, 0.6*cm, page_text)
        canvas.drawString(1.5*cm, 0.6*cm, "Facial Landmark Analysis")
        canvas.restoreState()
        
class FacialLandmarkAnalysisTest(ScriptedLoadableModuleTest): # type: ignore
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        logic = FacialLandmarkAnalysisLogic()
        assert logic.dist((0, 0), (3, 4)) == 5.0
        print("Tests passed!")