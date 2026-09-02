import os
import sys
import json
import logging
import math
import tempfile
import subprocess
import vtk, qt, ctk, slicer # type: ignore
import numpy as np
from slicer.ScriptedLoadableModule import * # type: ignore
from slicer.util import VTKObservationMixin # type: ignore

# Safely install dependencies inside setup() rather than top-level import
def ensure_dependencies():
    """Install required pip packages safely when needed, without crashing Slicer module discovery."""
    packages = [
        ('openpyxl', 'openpyxl'),
        ('Pillow', 'PIL'),
        ('jdatetime', 'jdatetime'),
        ('reportlab', 'reportlab'),
        ('arabic-reshaper', 'arabic_reshaper'),
        ('python-bidi', 'bidi'),
        ('numpy', 'numpy')
    ]
    for pkg_name, module_name in packages:
        try:
            __import__(module_name)
        except ImportError:
            try:
                logging.info(f"Installing missing package: {pkg_name}")
                slicer.util.pip_install(pkg_name)
            except Exception as e:
                logging.warning(f"Could not install {pkg_name}: {e}")
                


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
        # self.parent.icon = qt.QIcon(
        #     os.path.join(os.path.dirname(__file__),
        #                  'Resources', 'Icons', 'FacialLandmarkAnalysis.png')
        # )
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        iconPath = os.path.join(moduleDir, 'Resources', 'Icons', 'FacialLandmarkAnalysis.png')
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        self.parent.helpText = "Automatic facial landmark detection with Persian Excel export."
        self.parent.acknowledgementText = "Developed for Farinroshan."


#
# Widget
#
# type: ignore
class FacialLandmarkAnalysisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin): # type: ignore
    VIEW_KEYS = ['frontal', 'lateral', 'smile']
    VIEW_CODES = {'frontal': 'F', 'lateral': 'L', 'smile': 'S'}
    VIEW_LABELS_FA = {
        'frontal': 'نمای روبرو (Frontal)',
        'lateral': 'نمای نیمرخ (Lateral)',
        'smile': 'نمای لبخند (Smile)',
    }
    
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)  # type: ignore
        VTKObservationMixin.__init__(self)
        

    def _browsePythonExec(self):
        p = qt.QFileDialog.getOpenFileName(
            self.parent, "Select Python (torch env)", "", "Executables (*)"
        )
        if p:
            self.pythonEdit.setText(p)
            qt.QSettings().setValue("FacialLandmarkAnalysis/pythonPath", p)

    def _savePythonPath(self):
        p = self.pythonEdit.text.strip()
        if p:
            qt.QSettings().setValue("FacialLandmarkAnalysis/pythonPath", p)
    
    def _defaultPythonPath(self, moduleDir):
        """Prefer saved path, then venv next to extension, then python3."""
        # 1) Last value saved for this module (survives Slicer restarts; update if machine changes)
        settings = qt.QSettings()
        saved = settings.value("FacialLandmarkAnalysis/pythonPath", "")
        if saved and os.path.isfile(saved):
            return saved

        # 2) Portable: venv shipped/copied next to the extension package
        #    e.g. FacialLandmarkAnalysis/venv/bin/python
        candidates = [
            os.path.join(moduleDir, "venv", "bin", "python"),
            os.path.join(moduleDir, "venv", "bin", "python3"),
            os.path.join(moduleDir, ".venv", "bin", "python"),
            os.path.join(moduleDir, ".venv", "bin", "python3"),
            os.path.join(os.path.dirname(moduleDir), "venv", "bin", "python"),  # parent folder
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c

        # 3) Fallback
        import shutil
        return shutil.which("python3") or shutil.which("python") or sys.executable

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
        
        # ── Model paths ──
        modelCollapsible = ctk.ctkCollapsibleButton()
        modelCollapsible.text = "تنظیمات مدل"
        modelCollapsible.collapsed = True
        self.layout.addWidget(modelCollapsible)
        modelLayout = qt.QFormLayout(modelCollapsible)

        # infer.py path
        self.inferScriptEdit = qt.QLineEdit()
        defaultInfer = os.path.join(moduleDir,
                                     'models', 'scripts', 'infer.py')
        self.inferScriptEdit.setText(defaultInfer)
        inferBrowseBtn = qt.QPushButton("...")
        inferBrowseBtn.setMaximumWidth(30)
        inferBrowseBtn.connect('clicked()', self._browseInferScript)
        inferRow = qt.QHBoxLayout()
        inferRow.addWidget(self.inferScriptEdit)
        inferRow.addWidget(inferBrowseBtn)
        modelLayout.addRow("infer.py:", inferRow)

        # Python interpreter (the one that has torch etc.)
        self.pythonEdit = qt.QLineEdit()
        self.pythonEdit.setText(self._defaultPythonPath(moduleDir))
        
        pythonBrowseBtn = qt.QPushButton("...")
        pythonBrowseBtn.setMaximumWidth(30)
        pythonBrowseBtn.connect('clicked()', self._browsePythonExec)
        
        pythonRow = qt.QHBoxLayout()
        pythonRow.addWidget(self.pythonEdit)
        pythonRow.addWidget(pythonBrowseBtn)
        modelLayout.addRow("Python:", pythonRow)


        # Checkpoint paths  — 6 files total
        # self._ckptEdits = {}
        ckpt_labels = {
            'f_coarse': "Frontal Coarse:",
            'f_fine': "Frontal Fine:",
            'l_coarse': "Lateral Coarse:",
            'l_fine': "Lateral Fine:",
            's_coarse': "Smile Coarse:",
            's_fine': "Smile Fine:",
        }
        defaultCkptDir = os.path.join(moduleDir, 'models', 'outputs')
        ckpt_defaults = {
            'f_coarse': os.path.join(defaultCkptDir, 'f_coarse', 'checkpoints', 'best_val_mre_px.pt'),
            'f_fine':   os.path.join(defaultCkptDir, 'f_fine',   'checkpoints', 'best_val_mre_px.pt'),
            'l_coarse': os.path.join(defaultCkptDir, 'l_coarse', 'checkpoints', 'best_val_mre_px.pt'),
            'l_fine':   os.path.join(defaultCkptDir, 'l_fine',   'checkpoints', 'best_val_mre_px.pt'),
            's_coarse': os.path.join(defaultCkptDir, 's_coarse', 'checkpoints', 'best_val_mre_px.pt'),
            's_fine':   os.path.join(defaultCkptDir, 's_fine',   'checkpoints', 'best_val_s_combined.pt'),
        }
        for key, label in ckpt_labels.items():
            edit = qt.QLineEdit()
            edit.setText(ckpt_defaults.get(key, ''))
            btn = qt.QPushButton("...")
            btn.setMaximumWidth(30)
            btn.connect('clicked()', lambda e=edit: self._browseCkpt(e))
            row = qt.QHBoxLayout()
            row.addWidget(edit)
            row.addWidget(btn)
            modelLayout.addRow(label, row)
            self._ckptEdits[key] = edit

        # Smile presence threshold
        self.presenceThreshEdit = qt.QLineEdit()
        self.presenceThreshEdit.setText("0.6")
        modelLayout.addRow("Smile presence threshold:", self.presenceThreshEdit)
        
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

        # ── Load Images (3 views) ──
        imageCollapsible = ctk.ctkCollapsibleButton()
        imageCollapsible.text = "مرحله ۱ : بارگذاری تصاویر"
        self.layout.addWidget(imageCollapsible)
        imageLayout = qt.QGridLayout(imageCollapsible)

        self._loadBtns = {}
        self._loadLabels = {}
        btn_texts = {'frontal': 'Frontal', 'lateral': 'Lateral (Right or Left)',
                     'smile': 'Smile'}
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

        # self.detectionStatusLabel = qt.QLabel(
        #     "⚠️ AI models not yet available. Placeholder landmarks will be generated."
        # )
        # self.detectionStatusLabel.setStyleSheet(
        #     "color: orange; font-style: italic;")
        # self.detectionStatusLabel.setWordWrap(True)
        # detectionLayout.addWidget(self.detectionStatusLabel)
        
        self.detectionStatusLabel = qt.QLabel("در انتظار بارگذاری تصاویر و اجرای مدل")
        self.detectionStatusLabel.setStyleSheet("color: orange; font-style: italic;")
        self.detectionStatusLabel.setWordWrap(True)
        detectionLayout.addWidget(self.detectionStatusLabel)

        # Progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 3)
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
            "Frontal View", "Lateral View", "Smile View"
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

        self.exportBtn = qt.QPushButton("📊 استخراج خروجی Excel")
        self.exportBtn.setStyleSheet(
            "background-color: #2196F3; color: white; font-size: 14px; "
            "font-weight: bold; padding: 12px;"
        )
        self.exportBtn.connect('clicked()', self.onExportExcel)
        self.exportBtn.enabled = False
        exportLayout.addWidget(self.exportBtn)

        self.exportPdfBtn = qt.QPushButton("📄 استخراج خروجی PDF")
        self.exportPdfBtn.setStyleSheet(
            "background-color: #E91E63; color: white; font-size: 14px; "
            "font-weight: bold; padding: 12px;"
        )
        self.exportPdfBtn.connect('clicked()', self.onExportPDF)
        self.exportPdfBtn.enabled = False
        exportLayout.addWidget(self.exportPdfBtn)

        # Export both button - NEW
        self.exportBothBtn = qt.QPushButton("📦 استخراج خروجی (Excel + PDF)")
        self.exportBothBtn.setStyleSheet(
            "background-color: #9C27B0; color: white; font-size: 14px; "
            "font-weight: bold; padding: 12px;"
        )
        self.exportBothBtn.connect('clicked()', self.onExportBoth)
        self.exportBothBtn.enabled = False
        exportLayout.addWidget(self.exportBothBtn)

        self.layout.addStretch(1)
    
    def _buildInferEnv(self):
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE"):
            env.pop(key, None)
        ld = env.get("LD_LIBRARY_PATH", "")
        if ld:
            parts = [p for p in ld.split(":") if p and "Slicer" not in p and "slicer" not in p]
            if parts:
                env["LD_LIBRARY_PATH"] = ":".join(parts)
            else:
                env.pop("LD_LIBRARY_PATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        return env

    # ── Browse helpers ──
    def _browseInferScript(self):
        p = qt.QFileDialog.getOpenFileName(self.parent, "Select infer.py", "", "Python (*.py)")
        if p:
            self.inferScriptEdit.setText(p)

    def _browseCkpt(self, edit):
        p = qt.QFileDialog.getOpenFileName(self.parent, "Select checkpoint", "", "PyTorch (*.pt *.pth)")
        if p:
            edit.setText(p)
            
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
            self.getLateralLandmarks(),
            self.getSmileLandmarks()
        ][viewIndex]

    def getViewKeyFromIndex(self, index):
        return self.VIEW_KEYS[index]
    
    # ── Load image ──  
    def onLoadImage(self, viewKey):
        from PIL import Image, ImageDraw, ImageFont
        
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

        # This makes the image display right-side up
        sliceToRAS = vtk.vtkMatrix4x4()
        sliceToRAS.Identity()
        sliceToRAS.SetElement(0, 0, 1.0)   # X axis (right)
        # Y axis FLIPPED (down in view = up in image)
        sliceToRAS.SetElement(1, 1, -1.0)
        sliceToRAS.SetElement(2, 2, 1.0)   # Z axis
        redSliceNode.GetSliceToRAS().DeepCopy(sliceToRAS)
        redSliceNode.UpdateMatrices()

        # Fit to view
        redSliceLogic.FitSliceToAll()
        redSliceLogic.SnapSliceOffsetToIJK()

        # Show current markup, hide others
        for key, node in self.markupNodes.items():
            if node is not None and node.GetDisplayNode() is not None:
                node.GetDisplayNode().SetVisibility(key == viewKey)
                node.GetDisplayNode().SetViewNodeIDs([redSliceNode.GetID()])

    # ── Run inference ──
    def onRunDetection(self):
        missing = [k for k, v in self.imageNodes.items() if v is None]
        if missing:
            slicer.util.warningDisplay(
                f"Please load all 3 images first.\nMissing: {', '.join(missing)}")
            return
        
        # Validate model files exist
        infer_script = self.inferScriptEdit.text
        if not os.path.isfile(infer_script):
            slicer.util.errorDisplay(f"infer.py not found:\n{infer_script}")
            return

        for ckpt_key, edit in self._ckptEdits.items():
            if not os.path.isfile(edit.text):
                slicer.util.errorDisplay(
                    f"Checkpoint not found for {ckpt_key}:\n{edit.text}")
                return

        # Remove old markups
        for key, node in self.markupNodes.items():
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
                self.markupNodes[key] = None

        # Create temp dir for inference output
        self._inferTmpDir = tempfile.mkdtemp(prefix="fla_infer_")

        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.runDetectionBtn.enabled = False
        self.detectionStatusLabel.setText("⏳ در حال اجرای مدل...")
        self.detectionStatusLabel.setStyleSheet("color: blue; font-weight: bold;")
        slicer.app.processEvents()

        python_bin = self.pythonEdit.text
        success = True

        for step_idx, viewKey in enumerate(self.VIEW_KEYS):
            viewCode = self.VIEW_CODES[viewKey]
            imagePath = self.imagePaths[viewKey]

            # Build command
            if viewCode == 'F':
                coarse_ckpt = self._ckptEdits['f_coarse'].text
                fine_ckpt = self._ckptEdits['f_fine'].text
            elif viewCode == 'L':
                coarse_ckpt = self._ckptEdits['l_coarse'].text
                fine_ckpt = self._ckptEdits['l_fine'].text
            else:  # S
                coarse_ckpt = self._ckptEdits['s_coarse'].text
                fine_ckpt = self._ckptEdits['s_fine'].text

            cmd = [
                python_bin, infer_script,
                '--image', imagePath,
                '--view', viewCode,
                '--coarse', coarse_ckpt,
                '--fine', fine_ckpt,
                '--out_dir', self._inferTmpDir,
            ]

            # Add presence threshold for smile
            if viewCode == 'S':
                thresh = self.presenceThreshEdit.text.strip()
                if thresh:
                    cmd.extend(['--presence-threshold', thresh])

            logging.info(f"Running inference for {viewKey}: {' '.join(cmd)}")
            self.detectionStatusLabel.setText(
                f"⏳ در حال پردازش {self.VIEW_LABELS_FA[viewKey]}...")
            slicer.app.processEvents()

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=self._buildInferEnv(),
                    cwd=os.path.dirname(infer_script),  # often helps imports inside your repo
                )
                if result.returncode != 0:
                    logging.error(f"Inference failed for {viewKey}:\n"
                                  f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                    slicer.util.errorDisplay(
                        f"Inference failed for {viewKey}:\n{result.stderr[:500]}")
                    success = False
                    break
                else:
                    logging.info(f"Inference OK for {viewKey}")
                    if result.stdout.strip():
                        logging.info(f"STDOUT: {result.stdout[:200]}")
            except subprocess.TimeoutExpired:
                slicer.util.errorDisplay(
                    f"Inference timed out for {viewKey} (300s limit)")
                success = False
                break
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

        # Parse JSON results
        self._parseInferenceResults()

        # Create markups from results
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
        self.detectionStatusLabel.setText(
            "✓ لندمارک ها شناسایی شدند. برای اصلاح، نقاط را جابجا کنید.")
        self.detectionStatusLabel.setStyleSheet("color: green; font-weight: bold;")
        self.viewComboBox.setCurrentIndex(0)
        self.showImage('frontal')
        self.updateLandmarkList('frontal')
        slicer.util.infoDisplay(
            "تشخیص لندمارک ها کامل شد!\nبرای اصلاح، نقاط را جابجا کنید.")

    # def detectLandmarksForView(self, viewKey):
    #     """
    #     PLACEHOLDER - returns fixed positions.
    #     TODO: Replace with AI model inference when models available.
    #     """
    #     if viewKey == 'frontal':
    #         return self._placeholderFrontal()
    #     elif viewKey == 'right':
    #         return self._placeholderRightLateral()
    #     elif viewKey == 'left':
    #         return self._placeholderLeftLateral()
    #     elif viewKey == 'smile':
    #         return self._placeholderSmile()
    #     return []

    # def _placeholderFrontal(self):
    #     W, H = self.imageSizes['frontal']
    #     cx = W / 2
    #     return [
    #         (cx - 0.10*W, 0.42*H), (cx - 0.06*W, 0.42*H), (cx - 0.04*W, 0.43*H),
    #         (cx - 0.14*W, 0.42*H), (cx - 0.10*W, 0.45*H), (cx - 0.10*W, 0.46*H),
    #         (cx + 0.10*W, 0.42*H), (cx + 0.06*W, 0.42*H), (cx + 0.04*W, 0.43*H),
    #         (cx + 0.14*W, 0.42*H), (cx + 0.10*W, 0.45*H), (cx + 0.10*W, 0.46*H),
    #         (cx - 0.25*W, 0.44*H), (cx + 0.25*W, 0.44*H),
    #         (cx - 0.22*W, 0.48*H), (cx + 0.22*W, 0.48*H),
    #         (cx - 0.05*W, 0.60*H), (cx + 0.05*W, 0.60*H),
    #         (cx - 0.07*W, 0.71*H), (cx + 0.07*W, 0.71*H),
    #         (cx, 0.72*H),
    #         (cx - 0.19*W, 0.72*H), (cx + 0.19*W, 0.72*H),
    #         (cx, 0.85*H), (cx, 0.90*H),
    #     ]

    # def _placeholderRightLateral(self):
    #     W, H = self.imageSizes['right']
    #     return [
    #         (0.28*W, 0.15*H), (0.20*W, 0.32*H), (0.14*W, 0.40*H),
    #         (0.10*W, 0.50*H), (0.05*W, 0.58*H), (0.10*W, 0.60*H),
    #         (0.13*W, 0.63*H), (0.13*W, 0.68*H), (0.14*W, 0.75*H),
    #         (0.16*W, 0.80*H), (0.15*W, 0.85*H), (0.20*W, 0.90*H),
    #         (0.24*W, 0.94*H), (0.65*W, 0.55*H), (0.30*W, 0.10*H),
    #         (0.14*W, 0.71*H),
    #     ]

    # def _placeholderLeftLateral(self):
    #     W, H = self.imageSizes['left']
    #     return [
    #         (0.72*W, 0.15*H), (0.80*W, 0.32*H), (0.86*W, 0.40*H),
    #         (0.90*W, 0.50*H), (0.95*W, 0.58*H), (0.90*W, 0.60*H),
    #         (0.87*W, 0.63*H), (0.87*W, 0.68*H), (0.86*W, 0.75*H),
    #         (0.84*W, 0.80*H), (0.85*W, 0.85*H), (0.80*W, 0.90*H),
    #         (0.76*W, 0.94*H), (0.35*W, 0.55*H), (0.70*W, 0.10*H),
    #         (0.86*W, 0.71*H),
    #     ]

    # def _placeholderSmile(self):
    #     W, H = self.imageSizes['smile']
    #     cx = W / 2
    #     return [
    #         (cx - 0.10*W, 0.42*H), (cx + 0.10*W, 0.42*H),
    #         (cx, 0.70*H), (cx, 0.85*H), (cx, 0.92*H),
    #         (cx, 0.76*H), (cx, 0.72*H), (cx, 0.74*H),
    #     ]

    def _parseInferenceResults(self):
        """Find and parse JSON files produced by infer.py."""
        if not os.path.isdir(self._inferTmpDir):
            return

        # infer.py names files like: <stem>_<view>.json
        # e.g. 246F_F.json, 246R_L.json, 246S_S.json
        # We match by the _<viewCode>.json suffix
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
        """
        Convert parsed JSON landmarks to list of (x, y) tuples.
        Non-present landmarks get None so we can skip them.
        Returns dict {id: (x, y)} for present landmarks only.
        """
        data = self.inferenceResults[viewKey]
        if data is None:
            return {}

        result = {}
        for lm in data.get('landmarks', []):
            lm_id = lm['id']
            if lm.get('present', True) and lm.get('x') is not None and lm.get('y') is not None:
                result[lm_id] = (float(lm['x']), float(lm['y']))
        return result


    def createMarkupNode(self, viewKey, landmarkDict):
        """
        landmarkDict: {id: (x_pixel, y_pixel)} — only present landmarks.
        """
        viewIndex = self.VIEW_KEYS.index(viewKey)
        landmark_defs = self.getLandmarksForView(viewIndex)
        H = self.imageSizes[viewKey][1]

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

        # Choose directory
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
            # Export Excel
            self.logic.exportToExcel(
                coords, self.imagePaths, ppm, xlsx_path,
                self.patientNameEdit.text,
                self.doctorNameEdit.text,
                self.dateEdit.text
            )
            # Export PDF
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
class FacialLandmarkAnalysisLogic(ScriptedLoadableModuleLogic): # type: ignore
    
    VIEW_KEYS = ['frontal', 'lateral', 'smile']
    
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
    # Row builder
    # =============================================
    # def _row(self, index, report, normal, interpretation):
    def _row(self, index, report):
        """Build a user-facing row (4 columns only)."""
        return {
            'ایندکس': index,
            'گزارش': report,
            # 'حالت نرمال': normal,
            # 'تفسیر نتیجه': interpretation
        }

    def _interp_equal(self, val1, val2, tol=2.0):
        diff = abs(val1 - val2)
        if diff <= tol:
            return f"برابر (اختلاف = {diff:.2f}) - نرمال"
        return f"نابرابر (اختلاف = {diff:.2f}) - عدم تقارن"

    def _interp_ratio(self, ratio, target_min, target_max, more_msg, less_msg):
        if target_min <= ratio <= target_max:
            return f"نرمال (نسبت = {ratio:.3f})"
        elif ratio > target_max:
            return f"{more_msg} (نسبت = {ratio:.3f})"
        else:
            return f"{less_msg} (نسبت = {ratio:.3f})"

    def _interp_range(self, val, lo, hi, unit, more_msg, less_msg):
        if lo <= val <= hi:
            return f"نرمال ({val:.2f} {unit})"
        elif val > hi:
            return f"{more_msg} ({val:.2f} {unit})"
        else:
            return f"{less_msg} ({val:.2f} {unit})"

    # =============================================
    # FRONTAL rows
    # =============================================
    def buildFrontalRows(self, F, ppm):
        rows = []
        # ==== قرینگی افقی صورت (Slide 19) ====
        if 1 in F and 7 in F:
            mid = self.midpoint(F[1], F[7])
            # rows.append(self._row("قرینگی افقی صورت", "", "", ""))
            rows.append(self._row("قرینگی افقی صورت", ""))

            if 15 in F and 16 in F:
                d15 = abs(F[15][0] - mid[0]) / ppm
                d16 = abs(F[16][0] - mid[0]) / ppm
                rows.append(self._row("",
                                      f"X15 = {d15:.2f} | X16 = {d16:.2f} | اختلاف = {abs(d15-d16):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(d15, d16) + " - فواصل نابرابر نشانه عدم تقارن افقی می باشد"))
            if 17 in F and 18 in F:
                d17 = abs(F[17][0] - mid[0]) / ppm
                d18 = abs(F[18][0] - mid[0]) / ppm
                rows.append(self._row("",
                                      f"X17 = {d17:.2f} | X18 = {d18:.2f} | اختلاف = {abs(d17-d18):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(d17, d18)))
            if 19 in F and 20 in F:
                d19 = abs(F[19][0] - mid[0]) / ppm
                d20 = abs(F[20][0] - mid[0]) / ppm
                rows.append(self._row("",
                                      f"X19 = {d19:.2f} | X20 = {d20:.2f} | اختلاف = {abs(d19-d20):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(d19, d20)))
            if 22 in F and 23 in F:
                d22 = abs(F[22][0] - mid[0]) / ppm
                d23 = abs(F[23][0] - mid[0]) / ppm
                rows.append(self._row("",
                                      f"X22 = {d22:.2f} | X23 = {d23:.2f} | اختلاف = {abs(d22-d23):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(d22, d23)))
            if 24 in F:
                d24 = abs(F[24][0] - mid[0]) / ppm
                if d24 < 1:
                    interp24 = "نرمال - چانه در وسط"
                else:
                    side = "راست" if F[24][0] > mid[0] else "چپ"
                    interp24 = f"چانه انحراف دارد به سمت {side} ({d24:.2f} میلی متر)"
                rows.append(self._row("",
                                      f"فاصله = {d24:.2f}",))
                # "این فاصله باید صفر باشد (نقطه روی این عمود منصف باشد)",
                # interp24))

        # ==== قرینگی عمودی صورت (Slide 20) ====
        if 1 in F and 7 in F:
            y_line = (F[1][1] + F[7][1]) / 2
            # rows.append(self._row("قرینگی عمودی صورت", "", "", ""))
            rows.append(self._row("قرینگی عمودی صورت", ""))

            if 15 in F and 16 in F:
                dy15 = abs(F[15][1] - y_line) / ppm
                dy16 = abs(F[16][1] - y_line) / ppm
                rows.append(self._row("",
                                      f"Y15 = {dy15:.2f} | Y16 = {dy16:.2f} | اختلاف = {abs(dy15-dy16):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(dy15, dy16) + " - فواصل نابرابر نشانه عدم تقارن عمودی می باشد"))
            if 17 in F and 18 in F:
                dy17 = abs(F[17][1] - y_line) / ppm
                dy18 = abs(F[18][1] - y_line) / ppm
                rows.append(self._row("",
                                      f"Y17 = {dy17:.2f} | Y18 = {dy18:.2f} | اختلاف = {abs(dy17-dy18):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(dy17, dy18)))
            if 19 in F and 20 in F:
                dy19 = abs(F[19][1] - y_line) / ppm
                dy20 = abs(F[20][1] - y_line) / ppm
                rows.append(self._row("",
                                      f"Y19 = {dy19:.2f} | Y20 = {dy20:.2f} | اختلاف = {abs(dy19-dy20):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(dy19, dy20)))
            if 22 in F and 23 in F:
                dy22 = abs(F[22][1] - y_line) / ppm
                dy23 = abs(F[23][1] - y_line) / ppm
                rows.append(self._row("",
                                      f"Y22 = {dy22:.2f} | Y23 = {dy23:.2f} | اختلاف = {abs(dy22-dy23):.2f}",))
                # "هر دو فاصله باید برابر باشد",
                # self._interp_equal(dy22, dy23)))

        # ==== نسبت عرض گونه به عرض گونیال (Slide 21) ====
        if all(k in F for k in [15, 16, 22, 23]):
            zy_w = abs(F[16][0] - F[15][0]) / ppm
            go_w = abs(F[23][0] - F[22][0]) / ppm
            ratio = go_w / zy_w if zy_w != 0 else 0
            interp = self._interp_ratio(ratio, 0.70, 0.75,
                                        "عریض تر بودن عرض گونیال به عرض گونه",
                                        "بیشتر بودن عرض گونه به عرض گونیال")
            rows.append(self._row("نسبت عرض گونه به عرض گونیال",
                                  f"{ratio:.3f} ({ratio*100:.1f}%)",))
            # "این نسبت بایستی 70 تا 75 درصد باشد",
            # interp))

        # ==== یک پنجم های عمودی (Slide 22) ====
        if all(k in F for k in [3, 4, 9, 10, 13, 14]):
            s1 = abs(F[4][0] - F[13][0]) / ppm
            s2 = abs(F[3][0] - F[4][0]) / ppm
            s3 = abs(F[9][0] - F[3][0]) / ppm
            s4 = abs(F[10][0] - F[9][0]) / ppm
            s5 = abs(F[14][0] - F[10][0]) / ppm
            max_diff = max(s1, s2, s3, s4, s5) - min(s1, s2, s3, s4, s5)
            interp = "همه فواصل تقریباً برابر - نرمال" if max_diff < 3 else \
                     "فواصل نابرابر - احتمال هایپوتلوریسم یا هایپرتلوریسم یا موقعیت بیرون زده گوش ها"
            rows.append(self._row("یک پنجم های عمودی",
                                  f"X4-X13={s1:.2f} | X3-X4={s2:.2f} | X9-X3={s3:.2f} | X10-X9={s4:.2f} | X14-X10={s5:.2f}",))
            # "همه فواصل باید با هم برابر باشند",
            # interp))

        # ==== عرض بینی (Slide 23) ====
        if all(k in F for k in [3, 9, 17, 18]):
            nose_w = abs(F[18][0] - F[17][0]) / ppm
            ic_w = abs(F[9][0] - F[3][0]) / ppm
            ratio = nose_w / ic_w if ic_w != 0 else 0
            interp = self._interp_ratio(
                ratio, 0.9, 1.1, "عرض بینی پهن", "عرض بینی باریک")
            rows.append(self._row("عرض بینی", f"{ratio:.3f}",))
            # "نسبت باید 1 به 1 باشد", interp))

        # ==== عرض دهان (Slide 24) ====
        if all(k in F for k in [2, 8, 19, 20]):
            mouth_w = abs(F[20][0] - F[19][0]) / ppm
            iris_w = abs(F[8][0] - F[2][0]) / ppm
            ratio = mouth_w / iris_w if iris_w != 0 else 0
            interp = self._interp_ratio(
                ratio, 0.9, 1.1, "عرض دهان زیاد", "عرض دهان کم")
            rows.append(self._row("عرض دهان", f"{ratio:.3f}",))
            # "نسبت باید 1 به 1 باشد", interp))

        # ==== نمایش اسکرا (Slide 25) ====
        if 5 in F and 6 in F:
            ss_r = abs(F[6][1] - F[5][1]) / ppm
            interp = "نرمال (فاصله ≈ صفر)" if ss_r < 1 else \
                     f"دیده شدن صلبیه ({ss_r:.2f} میلی متر) - احتمال دفی شنسی میدفیس"
            rows.append(self._row("نمایش اسکرا (چشم راست)", f"{ss_r:.2f}",))
            # "اختلاف باید صفر باشد", interp))
        if 11 in F and 12 in F:
            ss_l = abs(F[12][1] - F[11][1]) / ppm
            interp = "نرمال (فاصله ≈ صفر)" if ss_l < 1 else \
                     f"دیده شدن صلبیه ({ss_l:.2f} میلی متر) - احتمال دفی شنسی میدفیس"
            rows.append(self._row("نمایش اسکرا (چشم چپ)", f"{ss_l:.2f}",))
            # "اختلاف باید صفر باشد", interp))

        # ==== کنت (Slide 26) ====
        if all(k in F for k in [1, 7, 19, 20]):
            num = abs(F[1][1] - F[19][1]) / ppm
            den = abs(F[7][1] - F[20][1]) / ppm
            ratio = num / den if den != 0 else 0
            interp = "نرمال (نسبت ≈ 1)" if 0.9 <= ratio <= 1.1 else \
                     f"حضور کنت اکلوزال (نسبت = {ratio:.3f})"
            rows.append(self._row("کنت", f"{ratio:.3f}",))
            # "نسبت باید 1 به 1 باشد", interp))

        return rows

    # =============================================
    # SMILE rows
    # =============================================
    def buildSmileRows(self, S, ppm):
        rows = []

        if all(k in S for k in [1, 2, 6]):
            mid = self.midpoint(S[1], S[2])
            dev = abs(S[6][0] - mid[0]) / ppm
            interp = "نرمال (اختلاف ≈ صفر)" if dev < 1 else \
                     f"انحراف میدلاین دندانی فک بالا از میدلاین صورت ({dev:.2f} میلی متر)"
            rows.append(
                self._row("میدلاین دندانی ماگزیلا به صورت", f"{dev:.2f}",))
            # "اختلاف بایستی صفر باشد", interp))

        if all(k in S for k in [4, 7]):
            dev = abs(S[4][0] - S[7][0]) / ppm
            interp = "نرمال (اختلاف ≈ صفر)" if dev < 1 else \
                     f"انحراف میدلاین دندانی فک پایین از میدلاین چانه ({dev:.2f} میلی متر)"
            rows.append(
                self._row("میدلاین دندانی مندیبل به چانه", f"{dev:.2f}",))
            # "اختلاف بایستی صفر باشد", interp))

        if all(k in S for k in [6, 7]):
            dev = abs(S[6][0] - S[7][0]) / ppm
            interp = "نرمال (اختلاف ≈ صفر)" if dev < 1 else \
                     f"عدم هماهنگی میدلاین دندانی فک بالا و پایین ({dev:.2f} میلی متر)"
            rows.append(
                self._row("میدلاین دندانی ماگزیلا به مندیبل", f"{dev:.2f}",))
            # "اختلاف بایستی صفر باشد", interp))

        if 3 in S and 6 in S:
            if 8 in S:
                val = abs(S[8][1] - S[6][1]) / ppm
                formula = "Y8-Y6"
            else:
                val = abs(S[3][1] - S[6][1]) / ppm
                formula = "Y3-Y6"
            rows.append(self._row("نمایش دندان", f"{val:.2f} ({formula})",))
            # "-",
            # f"مقدار نمایش دندان = {val:.2f} میلی متر"))

        if 3 in S:
            if 8 in S:
                val = abs(S[3][1] - S[8][1]) / ppm
                interp = f"نمایش لثه در لبخند = {val:.2f} میلی متر" if val > 0 else \
                    "نرمال (بدون نمایش لثه)"
            else:
                val = 0
                interp = "بدون نمایش لثه"
            rows.append(self._row("نمایش لثه", f"{val:.2f}",))
            # "-", interp))

        return rows

    # =============================================
    # PROFILE rows  (single lateral, no left/right distinction)
    # =============================================
    def buildProfileRows(self, L, ppm):
        """Build profile analysis from a single lateral view."""
        rows = []

        # Section header row for this side
        # rows.append(self._row(f"═══ {sideLabel} ═══", "", "", ""))

        if all(k in L for k in [1, 6, 11, 15]):
            d1 = abs(L[15][1] - L[1][1]) / ppm
            d2 = abs(L[1][1] - L[6][1]) / ppm
            d3 = abs(L[6][1] - L[11][1]) / ppm
            max_val = max(d1, d2, d3)
            if abs(d1 - d2) < 5 and abs(d2 - d3) < 5:
                interp = "همه یک سوم ها تقریباً برابر - نرمال"
            else:
                which = "فوقانی" if max_val == d1 else (
                    "میانی" if max_val == d2 else "تحتانی")
                interp = f"یک سوم {which} رشد عمودی بیشتری دارد"
            rows.append(self._row("یک سوم های افقی",
                                  f"Y15-Y1={d1:.2f} | Y1-Y6={d2:.2f} | Y6-Y11={d3:.2f}",))
            # "همه فواصل باید با هم برابر باشند", interp))

        if all(k in L for k in [6, 11, 16]):
            num1 = abs(L[6][1] - L[16][1]) / ppm
            den1 = abs(L[6][1] - L[11][1]) / ppm
            ratio1 = num1 / den1 if den1 != 0 else 0
            if 0.28 <= ratio1 <= 0.38:
                interp1 = "نرمال (نسبت 1 به 3)"
            elif ratio1 > 0.38:
                interp1 = "طول لب بالا بیشتر از نرمال نسبت به یک سوم تحتانی صورت"
            else:
                interp1 = "طول لب بالا کمتر از نرمال نسبت به یک سوم تحتانی صورت"
            rows.append(self._row("یک سوم تحتانی (1 به 3)", f"{ratio1:.3f}",))
            # "نسبت بایستی 1 به 3 باشد", interp1))

            num2 = abs(L[16][1] - L[11][1]) / ppm
            ratio2 = num2 / den1 if den1 != 0 else 0
            if 0.60 <= ratio2 <= 0.72:
                interp2 = "نرمال (نسبت 2 به 3)"
            elif ratio2 > 0.72:
                interp2 = "بیشتر از نرمال - طول لب بالا نسبت به یک سوم تحتانی صورت زیاد"
            else:
                interp2 = "کمتر از نرمال - طول لب بالا نسبت به یک سوم تحتانی صورت کم"
            rows.append(self._row("یک سوم تحتانی (2 به 3)", f"{ratio2:.3f}",))
            # "نسبت بایستی 2 به 3 باشد", interp2))

        if all(k in L for k in [1, 2, 3]):
            angle = self.angle3(L[1], L[2], L[3])
            interp = self._interp_range(angle, 125, 135, "درجه",
                                        "زاویه منفرج - نازیون کم عمق",
                                        "زاویه حاد - نازیون عمیق")
            rows.append(self._row("زاویه نازوفرونتال", f"{angle:.2f} درجه",))
            # "باید 125 تا 135 درجه باشد", interp))

        if all(k in L for k in [1, 2, 3, 4]):
            d32 = self.dist(L[3], L[2]) / ppm
            d41 = self.dist(L[4], L[1]) / ppm
            ratio = d32 / d41 if d41 != 0 else 0
            if 0.62 <= ratio <= 0.72:
                interp = "نرمال (~ 67 درصد)"
            elif ratio > 0.72:
                interp = f"طول بینی بلند تر از نرمال ({ratio*100:.1f}%)"
            else:
                interp = f"طول بینی کوتاه تر از نرمال ({ratio*100:.1f}%)"
            rows.append(
                self._row("طول بینی", f"{ratio:.3f} ({ratio*100:.1f}%)",))
            # "نسبت بایستی 67 درصد باشد", interp))

        if all(k in L for k in [3, 4, 6]):
            num = abs(L[3][0] - L[6][0]) / ppm
            den = abs(L[6][0] - L[4][0]) / ppm
            ratio = num / den if den != 0 else 0
            if 1.8 <= ratio <= 2.2:
                interp = "نرمال (نسبت 2 به 1)"
            elif ratio < 1.8:
                interp = f"نزدیک به 1 به 1 - نشانه دفی شنسی میدفیس (نسبت = {ratio:.3f})"
            else:
                interp = f"بیشتر از نرمال ({ratio:.3f})"
            rows.append(self._row("پروجکشن بینی", f"{ratio:.3f}",))
            # "نسبت بایستی 2 به 1 باشد", interp))

        if all(k in L for k in [5, 6, 7]):
            angle = self.angle3(L[5], L[6], L[7])
            if 90 <= angle <= 110:
                interp = "نرمال"
            elif angle > 110:
                interp = f"زاویه بیشتر از نرمال - ساپورت کم لب بالا ({angle:.2f} درجه)"
            else:
                interp = f"زاویه کمتر از نرمال - ساپورت زیاد لب بالا ({angle:.2f} درجه)"
            rows.append(self._row("زاویه نازولیبیال", f"{angle:.2f} درجه",))
            # "در مردان 90-95 درجه و در زنان 90-110 درجه باشد", interp))

        if all(k in L for k in [7, 8]):
            diff = (L[7][0] - L[8][0]) / ppm
            interp = "نرمال (مثبت)" if diff > 0 else \
                     f"عدد منفی ({diff:.2f}) - پروفایل صورت به سمت دیسکرپانسی اسکلتال کلاس 3"
            rows.append(
                self._row("پروجکشن لب بالا به لب پایین", f"{diff:.2f}",))
            # "این مقدار باید مثبت باشد", interp))

        if all(k in L for k in [5, 7, 8, 10]):
            d7 = self.pt_line_dist(L[7], L[5], L[10]) / ppm
            d8 = self.pt_line_dist(L[8], L[5], L[10]) / ppm
            sign7 = "مثبت (پروتروژن)" if d7 > 0 else (
                "منفی (رتروژن)" if d7 < 0 else "صفر")
            sign8 = "مثبت (پروتروژن)" if d8 > 0 else (
                "منفی (رتروژن)" if d8 < 0 else "صفر")
            rows.append(self._row("پروجکشن لب بالا و پایین نسبت به صورت",
                                  f"X7={d7:.2f} ({sign7}) | X8={d8:.2f} ({sign8})",))
            # "این فاصله بایستی صفر باشد",
            # "مثبت = جلوتر بودن نقاط از خط و پروتروژن لب ها | منفی = عقب تر بودن و رتروژن لب ها"))

        if all(k in L for k in [8, 9, 10]):
            angle = self.angle3(L[8], L[9], L[10])
            if 110 <= angle <= 130:
                interp = "نرمال"
            elif angle < 110:
                interp = f"زاویه کمتر - عمیق و حاده بودن فولد ({angle:.2f} درجه)"
            else:
                interp = f"زاویه بیشتر - کم عمق و منفرجه بودن فولد ({angle:.2f} درجه)"
            rows.append(self._row("زاویه منتولیبیال", f"{angle:.2f} درجه",))
            # "باید 110 تا 130 درجه باشد", interp))

        if all(k in L for k in [1, 6, 10]):
            raw_angle = self.angle3(L[1], L[6], L[10])
            angle_val = 180 - raw_angle
            if 8 <= angle_val <= 16:
                interp = "نرمال (4 ± 12 درجه)"
            elif angle_val > 16:
                interp = f"زاویه بیشتر - دفی شنسی چانه ({angle_val:.2f} درجه)"
            else:
                interp = f"زاویه کمتر یا منفی - زیاد بودن بعد قدامی-خلفی چانه ({angle_val:.2f} درجه)"
            rows.append(self._row("پروجکشن چانه", f"{angle_val:.2f} درجه",))
            # "باید 4 ± 12 درجه باشد", interp))

        if all(k in L for k in [11, 12, 13]):
            angle = self.angle3(L[11], L[12], L[13])
            if 90 <= angle <= 110:
                interp = "نرمال"
            elif angle > 110:
                interp = f"زاویه بیشتر - منفرجه بودن و دفی شنسی کم گردن ({angle:.2f} درجه)"
            else:
                interp = f"زاویه کمتر - حاده بودن و دفی شنسی خوب گردن ({angle:.2f} درجه)"
            rows.append(self._row("زاویه چانه-گردن", f"{angle:.2f} درجه",))
            # "باید 90 تا 110 درجه باشد", interp))

        if all(k in L for k in [1, 6, 10]):
            raw_angle = self.angle3(L[1], L[6], L[10])
            profile_angle = 180 - abs(raw_angle)
            if -17 <= profile_angle <= -7:
                interp = "نرمال"
            elif profile_angle < -17:
                interp = f"پروفایل صورتی محدب ({profile_angle:.2f} درجه)"
            else:
                interp = f"پروفایل صورتی مقعر ({profile_angle:.2f} درجه)"
            rows.append(self._row("زاویه پروفایل صورت",
                        f"{profile_angle:.2f} درجه",))
            # "در مردان -15 تا -7 درجه و در زنان -17 تا -9 درجه باشد", interp))

        return rows

    # =============================================
    # ANNOTATED IMAGE GENERATION
    # =============================================
    def createAnnotatedImage(self, viewKey, imagePath, coords, outPath):
        from PIL import Image, ImageDraw, ImageFont
        
        """
        Draw landmarks + reference lines on the image.
        Returns True if successful.
        """
        try:
            img = Image.open(imagePath).convert('RGB')
        except Exception as e:
            logging.error(f"Cannot load image {imagePath}: {e}")
            return False

        draw = ImageDraw.Draw(img)
        W, H = img.size

        # Try to load a font
        try:
            font_size = max(16, int(W / 80))
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()

        # Colors
        LANDMARK_COLOR = (0, 255, 255)      # Cyan
        LINE_COLOR = (255, 255, 0)          # Yellow
        MIDLINE_COLOR = (255, 100, 100)     # Red
        AUX_COLOR = (100, 255, 100)         # Green

        # Helper: draw a full-height vertical line through (x, y_top-y_bottom)
        def draw_vline(x, color=MIDLINE_COLOR, width=2):
            draw.line([(x, 0), (x, H)], fill=color, width=width)

        def draw_hline(y, color=MIDLINE_COLOR, width=2):
            draw.line([(0, y), (W, y)], fill=color, width=width)

        def draw_segment(p1, p2, color=LINE_COLOR, width=2):
            draw.line([p1, p2], fill=color, width=width)

        # ==== View-specific reference lines ====
        if viewKey == 'frontal':
            # Perpendicular midline through midpoint of L1 and L7 (Slide 19)
            if 1 in coords and 7 in coords:
                mid_x = (coords[1][0] + coords[7][0]) / 2
                draw_vline(mid_x, MIDLINE_COLOR, 4)

            # Horizontal line through L1 and L7 (Slide 20)
            if 1 in coords and 7 in coords:
                y_line = (coords[1][1] + coords[7][1]) / 2
                draw_hline(y_line, AUX_COLOR, 4)

            # Segments for measurement pairs
            if 1 in coords and 7 in coords:
                draw_segment(coords[1], coords[7], LINE_COLOR, 2)

        elif viewKey == 'smile':
            # Perpendicular midline through midpoint of L1 and L2
            if 1 in coords and 2 in coords:
                mid_x = (coords[1][0] + coords[2][0]) / 2
                draw_vline(mid_x, MIDLINE_COLOR, 4)
                draw_segment(coords[1], coords[2], LINE_COLOR, 2)

        elif viewKey == 'lateral':
            # E-line (from L5 to L10) - Slide 68
            if 5 in coords and 10 in coords:
                draw_segment(coords[5], coords[10], LINE_COLOR, 2)

            # Nasofrontal angle sides (L1-L2, L2-L3) - Slide 63
            if 1 in coords and 2 in coords and 3 in coords:
                draw_segment(coords[1], coords[2], AUX_COLOR, 1)
                draw_segment(coords[2], coords[3], AUX_COLOR, 1)

            # Nasolabial angle sides (L5-L6, L6-L7) - Slide 66
            if 5 in coords and 6 in coords and 7 in coords:
                draw_segment(coords[5], coords[6], AUX_COLOR, 1)
                draw_segment(coords[6], coords[7], AUX_COLOR, 1)

            # Mentolabial angle sides (L8-L9, L9-L10) - Slide 69
            if 8 in coords and 9 in coords and 10 in coords:
                draw_segment(coords[8], coords[9], AUX_COLOR, 1)
                draw_segment(coords[9], coords[10], AUX_COLOR, 1)

            # Cervicomental angle (L11-L12, L12-L13) - Slide 72
            if 11 in coords and 12 in coords and 13 in coords:
                draw_segment(coords[11], coords[12], AUX_COLOR, 1)
                draw_segment(coords[12], coords[13], AUX_COLOR, 1)

            # Chin/profile angle (L1-L6, L6-L10) - Slide 70,73
            if 1 in coords and 6 in coords and 10 in coords:
                draw_segment(coords[1], coords[6], MIDLINE_COLOR, 1)
                draw_segment(coords[6], coords[10], MIDLINE_COLOR, 1)

        # ==== Draw landmarks on top ====
        r = max(4, int(W / 200))
        for num, (x, y) in coords.items():
            # Draw cross
            draw.line([(x - r*2, y), (x + r*2, y)],
                      fill=LANDMARK_COLOR, width=2)
            draw.line([(x, y - r*2), (x, y + r*2)],
                      fill=LANDMARK_COLOR, width=2)
            # Draw number label
            draw.text((x + r*2 + 2, y + 2), str(num),
                      fill=LANDMARK_COLOR, font=font)

        # Save
        img.save(outPath, 'PNG', optimize=True)
        return True

    # =============================================
    # EXCEL EXPORT
    # =============================================
    def exportToExcel(self, coords, imagePaths, ppm, filePath, patientName, doctorName, date):
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        
        """Export to Excel: Frontal, Smile, Profile, Landmarks, Information sheets."""
        wb = openpyxl.Workbook()

        # ---- Styles ----
        HEADER_FONT = Font(name='B Nazanin', bold=True,
                           size=12, color="000000")
        CELL_FONT = Font(name='B Nazanin', size=11)
        INDEX_FONT = Font(name='B Nazanin', bold=True, size=11, color="1F4E78")
        TITLE_FONT = Font(name='B Nazanin', bold=True, size=16, color="2F5496")
        SECTION_FONT = Font(name='B Nazanin', bold=True,
                            size=13, color="C00000")
        HEADER_FILL = PatternFill(
            start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        INDEX_FILL = PatternFill(start_color="DDEBF7",
                                 end_color="DDEBF7", fill_type="solid")
        SECTION_FILL = PatternFill(
            start_color="FFE699", end_color="FFE699", fill_type="solid")
        CENTER = Alignment(horizontal='center',
                           vertical='center', wrap_text=True, readingOrder=2)
        RIGHT = Alignment(horizontal='right', vertical='center',
                          wrap_text=True, readingOrder=2)
        BORDER = Border(
            left=Side(style='thin', color='808080'),
            right=Side(style='thin', color='808080'),
            top=Side(style='thin', color='808080'),
            bottom=Side(style='thin', color='808080')
        )

        # User-facing columns only
        # COLUMNS = ['ایندکس', 'گزارش', 'حالت نرمال', 'تفسیر نتیجه']
        COLUMNS = ['ایندکس', 'گزارش']

        # ---- Helper: build an analysis sheet ----
        def buildAnalysisSheet(sheetName, rows):
            ws = wb.create_sheet(sheetName)
            ws.sheet_view.leftToRight = True

            # Headers
            for col_idx, col_name in enumerate(COLUMNS, start=1):
                c = ws.cell(row=1, column=col_idx, value=col_name)
                c.font = HEADER_FONT
                c.fill = HEADER_FILL
                c.alignment = CENTER
                c.border = BORDER

            # Data rows
            for row_idx, row in enumerate(rows, start=2):
                index_val = row.get('ایندکس', '')

                # Section header row (contains ═══ markers)
                is_section = index_val.startswith('═══')

                for col_idx, col_name in enumerate(COLUMNS, start=1):
                    value = row.get(col_name, "")
                    c = ws.cell(row=row_idx, column=col_idx, value=value)
                    c.alignment = RIGHT if col_idx != 1 else CENTER
                    c.border = BORDER

                    if is_section:
                        c.font = SECTION_FONT
                        c.fill = SECTION_FILL
                        c.alignment = CENTER
                    elif col_idx == 1 and value:
                        c.font = INDEX_FONT
                        c.fill = INDEX_FILL
                    else:
                        c.font = CELL_FONT

                # Merge section headers across all columns
                if is_section:
                    ws.merge_cells(start_row=row_idx, start_column=1,
                                   end_row=row_idx, end_column=len(COLUMNS))

            # Vertical merge for grouped index rows
            self._mergeGroupedRows(ws, rows, col=1)

            # Column widths
            # widths = {1: 40, 2: 50, 3: 40, 4: 60}
            widths = {1: 40, 2: 50}
            for col, w in widths.items():
                ws.column_dimensions[get_column_letter(col)].width = w

            ws.freeze_panes = 'A2'
            ws.row_dimensions[1].height = 32
            for i in range(2, len(rows) + 2):
                ws.row_dimensions[i].height = 40

        # ---- Build all analysis sheets ----
        # Delete default sheet, we'll build our own order
        default_sheet = wb.active
        wb.remove(default_sheet)

        # 1. Frontal sheet
        frontal_rows = self.buildFrontalRows(coords['frontal'], ppm)
        buildAnalysisSheet("Frontal", frontal_rows)

        # 2. Smile sheet
        smile_rows = self.buildSmileRows(coords.get('smile', {}), ppm)
        buildAnalysisSheet("Smile", smile_rows)

        # 3. Profile sheet (single lateral)
        profile_rows = self.buildProfileRows(coords.get('lateral', {}), ppm)
        buildAnalysisSheet("Profile", profile_rows)

        # 4. Landmarks sheet (annotated images embedded)
        self._buildLandmarksSheet(wb, coords, imagePaths,
                                  TITLE_FONT, HEADER_FONT, HEADER_FILL, CENTER, BORDER)

        # 5. Information sheet
        self._buildInformationSheet(wb, patientName, doctorName, date, coords,
                                    TITLE_FONT, CELL_FONT, RIGHT)

        wb.save(filePath)
        logging.info(f"Excel saved: {filePath}")

    def _mergeGroupedRows(self, ws, rows, col=1):
        """Merge consecutive empty index cells with the previous non-empty one."""
        start = None
        for i, row in enumerate(rows):
            index_val = row.get('ایندکس', '')
            # Don't merge section headers
            if index_val.startswith('═══'):
                if start is not None and (i - 1) > start:
                    try:
                        ws.merge_cells(start_row=start + 2, start_column=col,
                                       end_row=i + 1, end_column=col)
                    except:
                        pass
                start = None
                continue

            has_value = bool(index_val.strip())
            if has_value:
                if start is not None and (i - 1) > start:
                    try:
                        ws.merge_cells(start_row=start + 2, start_column=col,
                                       end_row=i + 1, end_column=col)
                    except:
                        pass
                start = i
        # Final group
        if start is not None and (len(rows) - 1) > start:
            try:
                ws.merge_cells(start_row=start + 2, start_column=col,
                               end_row=len(rows) + 1, end_column=col)
            except:
                pass

    def _buildLandmarksSheet(self, wb, coords, imagePaths,
                             title_font, header_font, header_fill, center, border):
        from openpyxl.styles import Font, PatternFill
        from openpyxl.drawing.image import Image as XLImage
        from PIL import Image, ImageDraw, ImageFont
        
        """Build sheet with annotated images."""
        ws = wb.create_sheet("Landmarks")
        ws.sheet_view.leftToRight = True

        ws['A1'] = "تصاویر با لندمارک ها و خطوط راهنما"
        ws['A1'].font = title_font
        ws['A1'].alignment = center
        ws.merge_cells('A1:D1')
        ws.row_dimensions[1].height = 30

        view_names_fa = {
            'frontal': "نمای روبرو (Frontal)",
            'lateral': "نمای نیمرخ (Lateral Profile)",
            'smile': "نمای لبخند (Smile)"
        }


        temp_dir = tempfile.mkdtemp(prefix="fla_export_")
        current_row = 3

        for viewKey in self.VIEW_KEYS:
            if imagePaths.get(viewKey) is None:
                continue

            # Title row
            title_cell = ws.cell(row=current_row, column=1,
                                 value=view_names_fa[viewKey])
            title_cell.font = Font(
                name='B Nazanin', bold=True, size=14, color="C00000")
            title_cell.alignment = center
            title_cell.fill = PatternFill(start_color="FFF2CC",
                                          end_color="FFF2CC", fill_type="solid")
            ws.merge_cells(start_row=current_row, start_column=1,
                           end_row=current_row, end_column=4)
            ws.row_dimensions[current_row].height = 28
            current_row += 2

            # Create annotated image
            out_path = os.path.join(temp_dir, f"{viewKey}_annotated.png")
            success = self.createAnnotatedImage(
                viewKey, imagePaths[viewKey], coords[viewKey], out_path
            )

            if success and os.path.exists(out_path):
                try:
                    # Resize for embedding
                    img = Image.open(out_path)
                    orig_w, orig_h = img.size
                    max_w = 600
                    max_h = 800
                    scale = min(max_w / orig_w, max_h / orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    img.thumbnail((new_w, new_h), Image.LANCZOS)
                    resized_path = os.path.join(
                        temp_dir, f"{viewKey}_resized.png")
                    img.save(resized_path, 'PNG')

                    xl_img = XLImage(resized_path)
                    xl_img.anchor = f"A{current_row}"
                    ws.add_image(xl_img)

                    # Reserve rows for image
                    rows_needed = max(30, int(new_h / 20))
                    current_row += rows_needed + 2
                except Exception as e:
                    logging.error(f"Failed to embed image {viewKey}: {e}")
                    err_cell = ws.cell(row=current_row, column=1,
                                       value=f"⚠️ خطا در نمایش تصویر: {str(e)}")
                    err_cell.font = Font(
                        name='B Nazanin', size=11, color="FF0000")
                    current_row += 2
            else:
                err_cell = ws.cell(row=current_row, column=1,
                                   value="⚠️ تصویر در دسترس نیست")
                err_cell.font = Font(name='B Nazanin', size=11, color="FF0000")
                current_row += 2

        # Column widths
        for col_letter in ['A', 'B', 'C', 'D']:
            ws.column_dimensions[col_letter].width = 25

    def _buildInformationSheet(self, wb, patientName, doctorName, date, coords,
                               title_font, cell_font, right):
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        
        """Build information sheet with patient info and landmark coordinates."""
        ws = wb.create_sheet("Information")
        ws.sheet_view.leftToRight = True

        # Title
        ws['A1'] = "اطلاعات گزارش"
        ws['A1'].font = title_font
        ws['A1'].alignment = right
        ws.merge_cells('A1:D1')
        ws.row_dimensions[1].height = 30

        # Patient info section
        ws['A3'] = "اطلاعات بیمار"
        ws['A3'].font = Font(name='B Nazanin', bold=True,
                             size=14, color="2F5496")
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

        # Coordinates section
        current_row = len(info_data) + 7
        ws.cell(row=current_row, column=1, value="مختصات لندمارک ها (پیکسل)").font = \
            Font(name='B Nazanin', bold=True, size=14, color="2F5496")
        ws.cell(row=current_row, column=1).alignment = right
        ws.merge_cells(start_row=current_row, start_column=1,
                       end_row=current_row, end_column=4)
        current_row += 2

        view_names_fa = {
            'frontal': "نمای روبرو",
            'lateral': "نمای نیمرخ",
            'smile': "نمای لبخند"
        }

        header_font_small = Font(
            name='B Nazanin', bold=True, size=11, color="FFFFFF")
        header_fill_small = PatternFill(
            start_color="4472C4", end_color="4472C4", fill_type="solid")
        center_align = Alignment(
            horizontal='center', vertical='center', readingOrder=2)
        border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )

        for viewKey in self.VIEW_KEYS:
            viewCoords = coords.get(viewKey, {})
            if not viewCoords:
                continue

            # View section header
            c = ws.cell(row=current_row, column=1,
                        value=view_names_fa[viewKey])
            c.font = Font(name='B Nazanin', bold=True, size=12, color="C00000")
            c.alignment = center_align
            c.fill = PatternFill(start_color="FFF2CC",
                                 end_color="FFF2CC", fill_type="solid")
            ws.merge_cells(start_row=current_row, start_column=1,
                           end_row=current_row, end_column=3)
            current_row += 1

            # Column headers
            for col_idx, header in enumerate(["شماره لندمارک", "X (پیکسل)", "Y (پیکسل)"], start=1):
                c = ws.cell(row=current_row, column=col_idx, value=header)
                c.font = header_font_small
                c.fill = header_fill_small
                c.alignment = center_align
                c.border = border
            current_row += 1

            # Data
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

        # Column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 20
        ws.column_dimensions['D'].width = 20

    def _rtl(self, text):
        import arabic_reshaper # type: ignore
        from bidi.algorithm import get_display # type: ignore
        """Convert Persian text for proper RTL display in PDF."""
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
        """Register a Persian-supporting font for PDF. Returns font name."""
        font_name = 'PersianFont'

        # Try to find a suitable font on the system
        font_candidates = [
            # Windows
            r"C:\Windows\Fonts\tahoma.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\BNazanin.ttf",
            # macOS
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]

        font_bold_candidates = [
            r"C:\Windows\Fonts\tahomabd.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

        try:
            font_path = None
            for candidate in font_candidates:
                if os.path.exists(candidate):
                    font_path = candidate
                    break

            if font_path is None:
                logging.warning(
                    "No Persian font found, using default (may not display Persian correctly)")
                return 'Helvetica'

            pdfmetrics.registerFont(TTFont(font_name, font_path))

            # Try to register bold version
            bold_path = None
            for candidate in font_bold_candidates:
                if os.path.exists(candidate):
                    bold_path = candidate
                    break

            if bold_path:
                pdfmetrics.registerFont(TTFont(font_name + '-Bold', bold_path))

            return font_name
        except Exception as e:
            logging.error(f"Font registration failed: {e}")
            return 'Helvetica'

    def exportToPDF(self, coords, imagePaths, ppm, filePath,
                    patientName, doctorName, date):
        from reportlab.lib.pagesizes import A4 # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore
        from reportlab.lib.units import cm # type: ignore
        from reportlab.lib import colors # type: ignore
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT # type: ignore
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, # type: ignore
                                    Table, TableStyle, PageBreak, KeepTogether)
        from PIL import Image, ImageDraw, ImageFont
        """Export a comprehensive PDF report."""

        # Register Persian font
        font_name = self._registerPersianFont()
        bold_font = font_name + '-Bold' if font_name != 'Helvetica' else 'Helvetica-Bold'

        # Create document
        doc = SimpleDocTemplate(
            filePath,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
            title=f"Facial Analysis Report - {patientName}",
            author=doctorName,
        )

        # ===== Styles =====
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Title'],
            fontName=bold_font, fontSize=22, textColor=colors.HexColor('#1976D2'),
            alignment=TA_CENTER, spaceAfter=20, leading=28
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle', parent=styles['Heading1'],
            fontName=bold_font, fontSize=16, textColor=colors.HexColor('#2F5496'),
            alignment=TA_CENTER, spaceAfter=15, leading=22
        )

        section_style = ParagraphStyle(
            'SectionHeading', parent=styles['Heading2'],
            fontName=bold_font, fontSize=14, textColor=colors.HexColor('#C00000'),
            alignment=TA_RIGHT, spaceAfter=10, spaceBefore=15, leading=20,
            backColor=colors.HexColor('#FFF2CC'), borderPadding=6
        )

        subsection_style = ParagraphStyle(
            'SubHeading', parent=styles['Heading3'],
            fontName=bold_font, fontSize=12, textColor=colors.HexColor('#1F4E78'),
            alignment=TA_RIGHT, spaceAfter=6, leading=16
        )

        body_style = ParagraphStyle(
            'CustomBody', parent=styles['Normal'],
            fontName=font_name, fontSize=11, textColor=colors.black,
            alignment=TA_RIGHT, leading=16
        )

        info_style = ParagraphStyle(
            'InfoStyle', parent=styles['Normal'],
            fontName=font_name, fontSize=12, textColor=colors.HexColor('#333333'),
            alignment=TA_RIGHT, leading=18
        )

        # ===== Build story =====
        story = []

        # ===== COVER PAGE =====
        story.append(Spacer(1, 3*cm))
        story.append(
            Paragraph(self._rtl("گزارش تحلیل لندمارک های صورت"), title_style))
        story.append(
            Paragraph("Facial Landmark Analysis Report", subtitle_style))
        story.append(Spacer(1, 2*cm))

        # Patient info table
        info_data = [
            [self._rtl(patientName or "-"), self._rtl(": نام بیمار")],
            [self._rtl(doctorName or "-"), self._rtl(": نام پزشک")],
            [self._rtl(date or "-"), self._rtl(": تاریخ")],
        ]
        info_table = Table(info_data, colWidths=[8*cm, 6*cm])
        info_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), font_name, 12),
            ('FONT', (1, 0), (1, -1), bold_font, 12),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#DDEBF7')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#B4C7E7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#B4C7E7')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 4*cm))

        # Footer info
        footer_text = self._rtl(
            "این گزارش با استفاده از افزونه Facial Landmark Analysis تهیه شده است.")
        story.append(Paragraph(footer_text, body_style))
        story.append(PageBreak())

        # ===== ANNOTATED IMAGES SECTION =====
        story.append(
            Paragraph(self._rtl("تصاویر با لندمارک ها و خطوط راهنما"), section_style))
        story.append(Spacer(1, 0.5*cm))

        view_names = {
            'frontal': "نمای روبرو (Frontal)",
            'lateral': "نمای نیمرخ (Lateral Profile)",
            'smile': "نمای لبخند (Smile)"
        }

        temp_dir = tempfile.mkdtemp(prefix="fla_pdf_")

        for viewKey in self.VIEW_KEYS:
            if imagePaths.get(viewKey) is None:
                continue

            # Create annotated image
            out_path = os.path.join(temp_dir, f"{viewKey}_pdf.png")
            success = self.createAnnotatedImage(
                viewKey, imagePaths[viewKey], coords.get(viewKey, {}), out_path
            )

            if not success or not os.path.exists(out_path):
                continue

            # View title
            story.append(
                Paragraph(self._rtl(view_names[viewKey]), subsection_style))
            story.append(Spacer(1, 0.3*cm))

            # Compute image size to fit page
            try:
                pil_img = Image.open(out_path)
                orig_w, orig_h = pil_img.size
                max_width = 14 * cm
                max_height = 20 * cm
                aspect = orig_w / orig_h

                if aspect > (max_width / max_height):
                    display_w = max_width
                    display_h = max_width / aspect
                else:
                    display_h = max_height
                    display_w = max_height * aspect

                # Create ReportLab image
                rl_img = RLImage(out_path, width=display_w, height=display_h)
                rl_img.hAlign = 'CENTER'

                # Use KeepTogether so image and title stay on same page
                story.append(KeepTogether([rl_img]))
                story.append(Spacer(1, 0.5*cm))

            except Exception as e:
                logging.error(f"Failed to add image {viewKey}: {e}")
                continue

            story.append(PageBreak())

        # ===== ANALYSIS RESULTS SECTIONS =====

        # Frontal Analysis
        frontal_rows = self.buildFrontalRows(coords['frontal'], ppm)
        if frontal_rows:
            story.append(
                Paragraph(self._rtl("تحلیل نمای روبرو"), section_style))
            story.append(Spacer(1, 0.3*cm))
            self._addAnalysisTable(story, frontal_rows, font_name, bold_font)
            story.append(PageBreak())

        # Smile Analysis
        smile_rows = self.buildSmileRows(coords['smile'], ppm)
        if smile_rows:
            story.append(
                Paragraph(self._rtl("تحلیل نمای لبخند"), section_style))
            story.append(Spacer(1, 0.3*cm))
            self._addAnalysisTable(story, smile_rows, font_name, bold_font)
            story.append(PageBreak())

        # Profile Analysis
        profile_rows = self.buildProfileRows(coords.get('lateral', {}), ppm)
        if profile_rows:
            story.append(Paragraph(self._rtl("تحلیل نمای نیمرخ"), section_style))
            story.append(Spacer(1, 0.3*cm))
            self._addAnalysisTable(story, profile_rows, font_name, bold_font)
            story.append(PageBreak())

        # Landmark coordinates
        story.append(
            Paragraph(self._rtl("مختصات لندمارک ها (پیکسل)"), section_style))
        story.append(Spacer(1, 0.3*cm))

        for viewKey in self.VIEW_KEYS:
            viewCoords = coords.get(viewKey, {})
            if not viewCoords:
                continue

            story.append(
                Paragraph(self._rtl(view_names[viewKey]), subsection_style))
            story.append(Spacer(1, 0.2*cm))

            # Coordinates table
            table_data = [[
                self._rtl("Y (پیکسل)"),
                self._rtl("X (پیکسل)"),
                self._rtl("شماره لندمارک")
            ]]
            for num in sorted(viewCoords.keys()):
                x, y = viewCoords[num]
                table_data.append([
                    f"{y:.1f}", f"{x:.1f}", str(num)
                ])

            coord_table = Table(table_data, colWidths=[4*cm, 4*cm, 4*cm])
            coord_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, -1), font_name, 10),
                ('FONT', (0, 0), (-1, 0), bold_font, 11),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.HexColor('#F2F2F2')]),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(coord_table)
            story.append(Spacer(1, 0.5*cm))

        # Build PDF
        doc.build(story, onFirstPage=self._pdfFooter,
                  onLaterPages=self._pdfFooter)

        # Cleanup temp files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

        logging.info(f"PDF saved: {filePath}")

    def _addAnalysisTable(self, story, rows, font_name, bold_font):
        from reportlab.lib.units import cm # type: ignore
        from reportlab.lib import colors # type: ignore
        from reportlab.platypus import (Spacer, Table, TableStyle) # type: ignore
        """Add an analysis table to the PDF story."""
        # Table headers
        # headers = [
        #     self._rtl("تفسیر نتیجه"),
        #     self._rtl("حالت نرمال"),
        #     self._rtl("گزارش"),
        #     self._rtl("ایندکس")
        # ]
        
        headers = [
            self._rtl("گزارش"),
            self._rtl("ایندکس")
        ]

        table_data = [headers]
        section_rows = []  # Track which rows are section headers

        for i, row in enumerate(rows):
            index_val = row.get('ایندکس', '')
            is_section = index_val.startswith('═══')

            if is_section:
                # Section header - single wide cell
                clean_title = index_val.replace('═══', '').strip()
                table_data.append([
                    '',
                    self._rtl(clean_title)
                ])
                section_rows.append(len(table_data) - 1)
            else:
                # table_data.append([
                #     self._rtl(row.get('تفسیر نتیجه', '')),
                #     self._rtl(row.get('حالت نرمال', '')),
                #     self._rtl(row.get('گزارش', '')),
                #     self._rtl(index_val)
                # ])
                table_data.append([
                    self._rtl(row.get('گزارش', '')),
                    self._rtl(index_val)
                ])

        # Column widths (page width ~17cm usable)
        # col_widths = [5.5*cm, 4*cm, 4*cm, 3.5*cm]
        col_widths = [11*cm, 6*cm]

        table = Table(table_data, colWidths=col_widths, repeatRows=1)

        # Base style
        style_cmds = [
            ('FONT', (0, 0), (-1, -1), font_name, 9),
            ('FONT', (0, 0), (-1, 0), bold_font, 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFFF00')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]

        # Style section header rows differently
        for row_idx in section_rows:
            style_cmds.extend([
                ('SPAN', (0, row_idx), (-1, row_idx)),
                ('BACKGROUND', (0, row_idx), (-1, row_idx),
                 colors.HexColor('#FFE699')),
                ('FONT', (0, row_idx), (-1, row_idx), bold_font, 11),
                ('TEXTCOLOR', (0, row_idx), (-1, row_idx),
                 colors.HexColor('#C00000')),
                ('ALIGN', (0, row_idx), (-1, row_idx), 'CENTER'),
            ])

        # Alternate row colors (skip section rows and header)
        for i in range(1, len(table_data)):
            if i not in section_rows and i % 2 == 0:
                style_cmds.append(
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F5F5F5'))
                )

        table.setStyle(TableStyle(style_cmds))
        story.append(table)
        story.append(Spacer(1, 0.5*cm))

    def _pdfFooter(self, canvas, doc):
        from reportlab.lib.pagesizes import A4 # type: ignore
        from reportlab.lib.units import cm # type: ignore
        from reportlab.lib import colors # type: ignore
        """Draw footer on each PDF page."""
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        # Footer line
        canvas.setStrokeColor(colors.HexColor('#CCCCCC'))
        canvas.line(2*cm, 1.5*cm, A4[0] - 2*cm, 1.5*cm)
        # Page number
        page_text = f"Page {doc.page}"
        canvas.drawRightString(A4[0] - 2*cm, 1*cm, page_text)
        # App name
        canvas.drawString(2*cm, 1*cm, "Facial Landmark Analysis")
        canvas.restoreState()


class FacialLandmarkAnalysisTest(ScriptedLoadableModuleTest): # type: ignore
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        logic = FacialLandmarkAnalysisLogic()
        assert logic.dist((0, 0), (3, 4)) == 5.0
        print("Tests passed!")
