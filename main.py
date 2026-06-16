from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
VERSION_FILE = WEB_ROOT / "version.json"
YOLO26_DETECTION_MODELS = ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt")
YOLO26_SEGMENTATION_MODELS = ("yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt")
YOLO26_CLASSIFICATION_MODELS = ("yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt", "yolo26l-cls.pt", "yolo26x-cls.pt")
YOLO26_MODELS = (*YOLO26_DETECTION_MODELS, *YOLO26_SEGMENTATION_MODELS, *YOLO26_CLASSIFICATION_MODELS)
YOLO26_MODEL_SET = set(YOLO26_MODELS)
RFDETR_DETECTION_MODELS = ("rfdetr-nano", "rfdetr-small", "rfdetr-medium", "rfdetr-large")
RFDETR_SEGMENTATION_MODELS = ("rfdetr-seg-nano", "rfdetr-seg-small", "rfdetr-seg-medium", "rfdetr-seg-large")
RFDETR_MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
    "rfdetr-seg-nano": "RFDETRSegNano",
    "rfdetr-seg-small": "RFDETRSegSmall",
    "rfdetr-seg-medium": "RFDETRSegMedium",
    "rfdetr-seg-large": "RFDETRSegLarge",
}
RFDETR_MODEL_SET = set(RFDETR_MODEL_CLASSES)
SAM3_MODELS = ("facebook/sam3",)
INPUT_NODE_ID = "input"
LEGACY_CAMERA_NODE_ID = "camera"
INFERENCE_NODE_IDS = ("detector", "segmenter", "classifier")
IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".tif", ".tiff"}
MAX_EVENTS = 32
MAX_TERMINAL_LINES = 500

_model_cache = {}
_model_lock = Lock()


def load_version_info() -> dict:
    try:
        with VERSION_FILE.open("r", encoding="utf-8") as version_file:
            payload = json.load(version_file)
    except (OSError, json.JSONDecodeError):
        payload = {}
    app = str(payload.get("app") or "VisoNode")
    version = str(payload.get("version") or "0.0.0")
    return {"app": app, "version": version}


VERSION_INFO = load_version_info()
APP_VERSION = VERSION_INFO["version"]


def git_version_info() -> dict:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        dirty_result = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {
        "commit": commit_result.stdout.strip() or None,
        "dirty": dirty_result.returncode != 0,
    }


class TerminalBuffer:
    def __init__(self, max_lines: int = MAX_TERMINAL_LINES) -> None:
        self._lock = Lock()
        self._lines: list[dict] = []
        self._seq = 0
        self._max = max_lines

    def append(self, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._lines.append({
                "seq": self._seq,
                "time": time.strftime("%H:%M:%S"),
                "text": text,
            })
            overflow = len(self._lines) - self._max
            if overflow > 0:
                del self._lines[:overflow]

    def since(self, cursor: int) -> dict:
        with self._lock:
            new_lines = [line for line in self._lines if line["seq"] > cursor]
            return {"cursor": self._seq, "lines": new_lines}


terminal = TerminalBuffer()


def terminal_log(text: str) -> None:
    print(text, flush=True)
    terminal.append(text)


def json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        raise ValueError("Missing request body.")
    return json.loads(handler.rfile.read(content_length).decode("utf-8"))


def open_external_terminal() -> str:
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        windows_terminal = shutil.which("wt.exe") or shutil.which("wt")
        if windows_terminal:
            subprocess.Popen(
                [windows_terminal, "-d", str(ROOT)],
                cwd=str(ROOT),
                creationflags=creationflags,
            )
            return "Opened Windows Terminal"

        shell_path = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
        if shell_path:
            subprocess.Popen(
                [shell_path, "-NoExit", "-Command", "Set-Location -LiteralPath $PWD"],
                cwd=str(ROOT),
                creationflags=creationflags,
            )
            return f"Opened {Path(shell_path).name}"

        subprocess.Popen(["cmd.exe", "/K"], cwd=str(ROOT), creationflags=creationflags)
        return "Opened Command Prompt"

    candidates = (
        ("x-terminal-emulator", []),
        ("gnome-terminal", ["--working-directory", str(ROOT)]),
        ("konsole", ["--workdir", str(ROOT)]),
        ("xfce4-terminal", ["--working-directory", str(ROOT)]),
        ("xterm", ["-e", f"cd {ROOT} && exec sh"]),
    )
    for executable, args in candidates:
        path = shutil.which(executable)
        if path:
            subprocess.Popen([path, *args], cwd=str(ROOT), start_new_session=True)
            return f"Opened {executable}"
    raise RuntimeError("No supported external terminal application was found.")


def open_file_dialog() -> str | None:
    if os.name == "nt":
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not powershell:
            raise RuntimeError("PowerShell is required to open the Windows file picker.")

        filter_spec = (
            "Vision files (*.bmp;*.dib;*.jpg;*.jpeg;*.jpe;*.jp2;*.png;*.webp;*.pbm;*.pgm;*.ppm;*.pxm;*.pnm;*.tif;*.tiff;*.mp4;*.avi;*.mov;*.mkv;*.webm;*.m4v;*.wmv)"
            "|*.bmp;*.dib;*.jpg;*.jpeg;*.jpe;*.jp2;*.png;*.webp;*.pbm;*.pgm;*.ppm;*.pxm;*.pnm;*.tif;*.tiff;*.mp4;*.avi;*.mov;*.mkv;*.webm;*.m4v;*.wmv"
            "|Image files (*.bmp;*.jpg;*.jpeg;*.png;*.webp;*.tif;*.tiff)|*.bmp;*.jpg;*.jpeg;*.png;*.webp;*.tif;*.tiff"
            "|Video files (*.mp4;*.avi;*.mov;*.mkv;*.webm;*.m4v;*.wmv)|*.mp4;*.avi;*.mov;*.mkv;*.webm;*.m4v;*.wmv"
            "|All files (*.*)|*.*"
        )
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Select input file'
$dialog.Filter = @'
{filter_spec}
'@
$dialog.Multiselect = $false
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
  Write-Output $dialog.FileName
}}
"""
        result = subprocess.run(
            [powershell, "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Windows file picker failed.").strip()
            raise RuntimeError(detail)
        selected = result.stdout.strip()
        return str(Path(selected).resolve()) if selected else None

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python installation.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[
                ("Vision files", "*.bmp *.dib *.jpg *.jpeg *.jpe *.jp2 *.png *.webp *.pbm *.pgm *.ppm *.pxm *.pnm *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"),
                ("Image files", "*.bmp *.dib *.jpg *.jpeg *.jpe *.jp2 *.png *.webp *.pbm *.pgm *.ppm *.pxm *.pnm *.tif *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        return str(Path(path).resolve()) if path else None
    finally:
        root.destroy()


def runtime_devices() -> dict:
    payload = {
        "torchInstalled": False,
        "torchVersion": None,
        "cudaAvailable": False,
        "cudaVersion": None,
        "nvidiaSmi": bool(shutil.which("nvidia-smi")),
        "nvidiaGpus": [],
        "devices": [],
        "recommendation": "Use CPU, or run scripts\\install-gpu.ps1 on a machine with an NVIDIA GPU.",
    }

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 3:
                        payload["nvidiaGpus"].append({
                            "name": parts[0],
                            "memoryMb": int(float(parts[1])),
                            "driver": parts[2],
                        })
        except Exception:
            pass

    try:
        import torch
    except ImportError:
        return payload

    payload["torchInstalled"] = True
    payload["torchVersion"] = torch.__version__
    payload["cudaVersion"] = torch.version.cuda
    payload["cudaAvailable"] = bool(torch.cuda.is_available())
    if payload["cudaAvailable"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            payload["devices"].append({
                "id": f"cuda:{index}",
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memoryMb": round(props.total_memory / (1024 * 1024)),
            })
        payload["recommendation"] = "CUDA is available. Select Auto or a CUDA device in Object Detection."
    elif payload["nvidiaGpus"]:
        payload["recommendation"] = "NVIDIA GPU detected, but PyTorch CUDA is unavailable. Run scripts\\install-gpu.ps1."
    return payload


def resolve_inference_device(config: dict) -> tuple[str, str]:
    requested = str(config.get("device", "auto") or "auto").strip().lower()
    if requested in ("", "auto"):
        try:
            import torch
        except ImportError:
            return "cpu", "CPU"
        if torch.cuda.is_available():
            return "cuda:0", f"CUDA 0 - {torch.cuda.get_device_name(0)}"
        return "cpu", "CPU"

    if requested == "cpu":
        return "cpu", "CPU"

    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("GPU was selected, but PyTorch is not installed. Run scripts\\install-gpu.ps1.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was selected, but PyTorch CUDA is not available. Run scripts\\check-gpu.ps1.")
        try:
            index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device '{requested}'.") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is not available.")
        return f"cuda:{index}", f"CUDA {index} - {torch.cuda.get_device_name(index)}"

    raise RuntimeError(f"Unsupported inference device '{requested}'.")


def load_yolo_model(model_name: str, device: str = "cpu"):
    if model_name not in YOLO26_MODEL_SET:
        raise ValueError(f"Unsupported YOLO26 model '{model_name}'.")

    cache_key = (model_name, device)
    with _model_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
            ) from exc

        model = YOLO(model_name)
        model.to(device)
        _model_cache[cache_key] = model
        return model


def default_rfdetr_model(source_node_id: str) -> str:
    return "rfdetr-seg-nano" if source_node_id == "segmenter" else "rfdetr-nano"


def inference_model_name(inference_node: dict) -> str:
    config = inference_node.get("config", {})
    engine = config.get("engine", "yolo26")
    if engine == "sam3":
        return str(sam3_checkpoint_path(config) or "facebook/sam3")
    if engine == "rfdetr":
        return config.get("rfdetrModel") or default_rfdetr_model(inference_node.get("id", "detector"))
    return config.get("yoloModel") or "yolo26n.pt"


def load_rfdetr_model(model_name: str, device: str = "cpu"):
    if model_name not in RFDETR_MODEL_SET:
        raise ValueError(f"Unsupported RF-DETR model '{model_name}'.")

    cache_key = ("rfdetr", model_name, device)
    with _model_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR is not installed. Run '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
            ) from exc

        class_name = RFDETR_MODEL_CLASSES[model_name]
        model_class = getattr(rfdetr, class_name, None)
        if model_class is None:
            raise RuntimeError(
                f"The installed RF-DETR package does not provide {class_name}. Update with '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
            )
        try:
            model = model_class(device=device)
        except TypeError:
            model = model_class()
        _model_cache[cache_key] = model
        return model


def resolve_model_file(value: str, default_name: str = "") -> Path:
    model_path = Path(str(value or default_name).strip().strip('"') or default_name).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    return model_path.resolve()


def sam3_concepts(config: dict) -> list[str]:
    concepts = [
        item.strip()
        for item in str(config.get("concepts") or "person").split(",")
        if item.strip()
    ]
    if not concepts:
        raise RuntimeError("SAM 3 concept prompts are empty. Add one or more noun phrases, for example 'person, car'.")
    return concepts


def sam3_checkpoint_path(config: dict) -> Path | None:
    raw_value = str(config.get("samCheckpoint") or "").strip().strip('"')
    if not raw_value:
        return None
    checkpoint_path = resolve_model_file(raw_value)
    if not checkpoint_path.exists():
        raise RuntimeError(f"SAM 3 checkpoint was not found at {checkpoint_path}.")
    return checkpoint_path


def load_sam3_processor(config: dict, device: str = "cpu"):
    checkpoint_path = sam3_checkpoint_path(config)
    threshold = float(config.get("threshold", 0.25))
    imgsz = int(config.get("imgsz", 640))
    cache_key = ("official-sam3", str(checkpoint_path or "hf"), device, threshold, imgsz)
    with _model_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            if getattr(exc, "name", "") == "triton":
                raise RuntimeError(
                    "Meta's official SAM 3 package requires Triton and the official CUDA/Linux stack. Use WSL/Linux with CUDA 12.6+ and PyTorch 2.7+, or install a compatible Triton build for this environment."
                ) from exc
            raise RuntimeError(
                "Meta's official SAM 3 package is not installed. Run '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
            ) from exc

        try:
            builder_device = "cuda" if device.startswith("cuda") else device
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                device=builder_device,
                load_from_HF=checkpoint_path is None,
            )
            model.to(device)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load Meta SAM 3. If you are using Hugging Face download, request access to facebook/sam3 and run 'hf auth login'."
            ) from exc

        processor = Sam3Processor(model, resolution=imgsz, device=device, confidence_threshold=threshold)
        _model_cache[cache_key] = processor
        return processor


def enabled_node(workflow: dict, node_id: str) -> dict | None:
    node = node_by_id(workflow, node_id)
    return node if node and node.get("enabled", True) else None


def node_by_id(workflow: dict, node_id: str) -> dict | None:
    return next((node for node in workflow.get("nodes", []) if node.get("id") == node_id), None)


def normalize_workflow(workflow: dict) -> dict:
    for node in workflow.get("nodes", []):
        if node.get("type") == LEGACY_CAMERA_NODE_ID:
            node["type"] = INPUT_NODE_ID
            if node.get("id") == LEGACY_CAMERA_NODE_ID and not node_by_id(workflow, INPUT_NODE_ID):
                node["id"] = INPUT_NODE_ID
        if node.get("type") == INPUT_NODE_ID:
            config = node.setdefault("config", {})
            config.setdefault("sourceType", "camera")
            if "source" not in config:
                config["source"] = config.get("cameraIndex", config.get("deviceId", 0))
        if node.get("type") == "detector":
            config = node.setdefault("config", {})
            if config.get("engine") not in ("yolo26", "rfdetr"):
                config["engine"] = "yolo26"
            config.setdefault("rfdetrModel", "rfdetr-nano")
        if node.get("type") == "segmenter":
            config = node.setdefault("config", {})
            if config.get("engine") not in ("yolo26", "sam3", "rfdetr"):
                config["engine"] = "yolo26"
            config.setdefault("rfdetrModel", "rfdetr-seg-nano")
        if node.get("type") == "classifier":
            config = node.setdefault("config", {})
            config["engine"] = "yolo26"

    for edge in workflow.get("edges", []):
        if len(edge) < 2:
            continue
        if edge[0] == LEGACY_CAMERA_NODE_ID:
            edge[0] = INPUT_NODE_ID
        if edge[1] == LEGACY_CAMERA_NODE_ID:
            edge[1] = INPUT_NODE_ID
    return workflow


def source_node_id(workflow: dict) -> str:
    if node_by_id(workflow, INPUT_NODE_ID):
        return INPUT_NODE_ID
    return LEGACY_CAMERA_NODE_ID


def active_input_node(workflow: dict) -> dict | None:
    return enabled_node(workflow, source_node_id(workflow))


def has_active_path(workflow: dict, from_id: str, to_id: str) -> bool:
    if not enabled_node(workflow, from_id) or not enabled_node(workflow, to_id):
        return False
    if from_id == to_id:
        return True

    visited = set()
    queue = [from_id]
    edges = workflow.get("edges", [])
    while queue:
        current_id = queue.pop(0)
        if current_id == to_id:
            return True
        if current_id in visited:
            continue
        visited.add(current_id)
        for edge_from_id, edge_to_id in edges:
            if edge_from_id == current_id and edge_to_id not in visited and enabled_node(workflow, edge_to_id):
                queue.append(edge_to_id)
    return False


def active_detector_node(workflow: dict) -> dict | None:
    detector = enabled_node(workflow, "detector")
    input_id = source_node_id(workflow)
    if detector and has_active_path(workflow, input_id, "detector"):
        return detector
    return None


def active_segmenter_node(workflow: dict) -> dict | None:
    segmenter = enabled_node(workflow, "segmenter")
    input_id = source_node_id(workflow)
    if segmenter and has_active_path(workflow, input_id, "segmenter"):
        return segmenter
    return None


def active_classifier_node(workflow: dict) -> dict | None:
    classifier = enabled_node(workflow, "classifier")
    input_id = source_node_id(workflow)
    if classifier and has_active_path(workflow, input_id, "classifier"):
        return classifier
    return None


def active_inference_nodes(workflow: dict) -> list[dict]:
    nodes = []
    input_id = source_node_id(workflow)
    for node_id in INFERENCE_NODE_IDS:
        node = enabled_node(workflow, node_id)
        if node and has_active_path(workflow, input_id, node_id):
            nodes.append(node)
    return nodes


def active_filter_node(workflow: dict) -> dict | None:
    filter_node = enabled_node(workflow, "filter")
    if filter_node and any(
        has_active_path(workflow, node.get("id"), "filter")
        for node in active_inference_nodes(workflow)
    ):
        return filter_node
    return None


def inference_results_for_node(workflow: dict, node_id: str, detections: list[dict]) -> list[dict]:
    return [
        detection for detection in detections
        if has_active_path(workflow, detection.get("sourceNodeId", "detector"), node_id)
    ]


def detections_for_node(workflow: dict, node_id: str, detections: list[dict], filtered: list[dict]) -> list[dict]:
    node = enabled_node(workflow, node_id)
    input_id = source_node_id(workflow)
    if not node or not has_active_path(workflow, input_id, node_id):
        return []
    if node_id == "preview" and not bool(node.get("config", {}).get("useFilter", False)):
        direct_detections = inference_results_for_node(workflow, node_id, detections)
        if direct_detections:
            return direct_detections
    if active_filter_node(workflow) and has_active_path(workflow, "filter", node_id):
        return filtered
    return inference_results_for_node(workflow, node_id, detections)


def normalize_camera_source(value) -> int | str:
    if value is None or value == "":
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def resolve_file_input_path(value) -> Path:
    text = str(value or "").strip().strip('"')
    if not text:
        raise RuntimeError("File input path is empty.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"File input does not exist: {path}")
    return path


class OpenCVInputSource:
    def __init__(self, cv2_module, config: dict) -> None:
        self.cv2 = cv2_module
        self.config = config
        self.capture = None
        self.image = None
        self.loop = bool(config.get("loop", True))
        self.label = "input"

    def open(self) -> tuple[str, int, int]:
        source_type = str(self.config.get("sourceType", "camera") or "camera").lower()
        if source_type == "file":
            path = resolve_file_input_path(self.config.get("filePath"))
            self.label = str(path)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                self.image = self.cv2.imread(str(path))
                if self.image is None:
                    raise RuntimeError(f"Unable to read image file: {path}")
                height, width = self.image.shape[:2]
                return f"image file {path}", width, height
            self.capture = self.cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                raise RuntimeError(f"Unable to open video file: {path}")
            return f"video file {path}", self.width, self.height

        source = normalize_camera_source(
            self.config.get("source", self.config.get("cameraIndex", self.config.get("deviceId", 0)))
        )
        self.label = repr(source)
        self.capture = self.cv2.VideoCapture(source)
        self.capture.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
        width = int(self.config.get("width", 0) or 0)
        height = int(self.config.get("height", 0) or 0)
        if width > 0:
            self.capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera source {source!r}.")
        return f"camera source {source!r}", self.width, self.height

    @property
    def width(self) -> int:
        if self.image is not None:
            return int(self.image.shape[1])
        return int(self.capture.get(self.cv2.CAP_PROP_FRAME_WIDTH) or 0) if self.capture is not None else 0

    @property
    def height(self) -> int:
        if self.image is not None:
            return int(self.image.shape[0])
        return int(self.capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT) or 0) if self.capture is not None else 0

    @property
    def is_static(self) -> bool:
        return self.image is not None

    def read(self):
        if self.image is not None:
            return True, self.image.copy()
        ok, frame = self.capture.read()
        if ok or not self.loop:
            return ok, frame
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
        return self.capture.read()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


def run_yolo26_frame(frame, inference_node: dict) -> list[dict]:
    config = inference_node.get("config", {})
    source_node_id = inference_node.get("id", "detector")
    is_classifier = source_node_id == "classifier"
    model_name = config.get("yoloModel") or "yolo26n.pt"
    threshold = float(config.get("threshold", 0.55))
    imgsz = int(config.get("imgsz", 640))
    device, _ = resolve_inference_device(config)
    model = load_yolo_model(model_name, device)
    predict_args = {
        "imgsz": imgsz,
        "device": device,
        "verbose": False,
    }
    if not is_classifier:
        predict_args["conf"] = threshold
    if not is_classifier and "end2end" in config:
        predict_args["end2end"] = bool(config.get("end2end"))

    try:
        results = model.predict(frame, **predict_args)
    except TypeError:
        predict_args.pop("end2end", None)
        results = model.predict(frame, **predict_args)

    detections = []
    for result in results:
        names = result.names or {}
        probs = getattr(result, "probs", None)
        if is_classifier and probs is not None:
            class_id = int(probs.top1)
            confidence = probs.top1conf
            if hasattr(confidence, "cpu"):
                confidence = confidence.cpu().item()
            detections.append({
                "class": names.get(class_id, str(class_id)),
                "score": float(confidence),
                "sourceNodeId": source_node_id,
                "kind": "classification",
            })
            continue
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        masks = []
        if getattr(result, "masks", None) is not None and result.masks is not None:
            masks = getattr(result.masks, "xy", None)
            if masks is None:
                masks = []
        for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box
            detection = {
                "class": names.get(int(class_id), str(int(class_id))),
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "sourceNodeId": source_node_id,
                "kind": "segmentation" if source_node_id == "segmenter" else "detection",
            }
            if index < len(masks):
                polygon = masks[index]
                if len(polygon):
                    detection["mask"] = [[float(point[0]), float(point[1])] for point in polygon]
            detections.append(detection)
    return detections


def _rfdetr_class_name(model, detections, index: int, class_id) -> str:
    data = getattr(detections, "data", {}) or {}
    for key in ("class_name", "class_names", "label", "labels"):
        labels = data.get(key)
        if labels is not None and index < len(labels):
            return str(labels[index])

    class_names = getattr(model, "class_names", {}) or {}
    try:
        class_key = int(class_id)
    except (TypeError, ValueError):
        return "object"
    return str(class_names.get(class_key, str(class_key)))


def run_rfdetr_frame(frame, inference_node: dict) -> list[dict]:
    import cv2
    import numpy as np

    config = inference_node.get("config", {})
    source_node_id = inference_node.get("id", "detector")
    model_name = config.get("rfdetrModel") or default_rfdetr_model(source_node_id)
    if source_node_id == "segmenter" and model_name not in RFDETR_SEGMENTATION_MODELS:
        raise RuntimeError(f"RF-DETR model '{model_name}' is not an instance segmentation model.")
    if source_node_id == "detector" and model_name not in RFDETR_DETECTION_MODELS:
        raise RuntimeError(f"RF-DETR model '{model_name}' is not an object detection model.")

    threshold = float(config.get("threshold", 0.55))
    device, _ = resolve_inference_device(config)
    model = load_rfdetr_model(model_name, device)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, threshold=threshold)
    if isinstance(results, list):
        results = results[0] if results else None
    if results is None:
        return []

    xyxy = getattr(results, "xyxy", None)
    if xyxy is None:
        return []
    boxes = np.asarray(xyxy).tolist()
    confidences = getattr(results, "confidence", None)
    scores = np.asarray(confidences).tolist() if confidences is not None else [1.0] * len(boxes)
    class_ids_raw = getattr(results, "class_id", None)
    class_ids = np.asarray(class_ids_raw).tolist() if class_ids_raw is not None else [None] * len(boxes)
    masks = getattr(results, "mask", None)

    detections = []
    for index, box in enumerate(boxes):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        class_id = class_ids[index] if index < len(class_ids) else None
        score = scores[index] if index < len(scores) else 1.0
        detection = {
            "class": _rfdetr_class_name(model, results, index, class_id),
            "score": float(score),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "sourceNodeId": source_node_id,
            "kind": "segmentation" if source_node_id == "segmenter" else "detection",
        }
        if masks is not None and index < len(masks):
            mask = np.squeeze(np.asarray(masks[index]))
            if mask.ndim == 2:
                mask = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    if len(contour) >= 3:
                        detection["mask"] = [[float(point[0]), float(point[1])] for point in contour]
        detections.append(detection)
    return detections


def run_sam3_frame(frame, inference_node: dict) -> list[dict]:
    from contextlib import nullcontext

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    config = inference_node.get("config", {})
    source_node_id = inference_node.get("id", "segmenter")
    device, _ = resolve_inference_device(config)
    processor = load_sam3_processor(config, device)
    concepts = sam3_concepts(config)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    def sam3_autocast():
        if device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    with sam3_autocast():
        image_state = processor.set_image(image)

    detections = []
    for concept in concepts:
        prompt_state = dict(image_state)
        prompt_state["backbone_out"] = dict(image_state["backbone_out"])
        with sam3_autocast():
            output = processor.set_text_prompt(state=prompt_state, prompt=concept)
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or boxes is None or scores is None:
            continue
        boxes = boxes.detach().cpu().tolist()
        scores = scores.detach().cpu().tolist()
        masks = masks.detach().cpu().numpy()
        for index, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = [float(value) for value in box]
            detection = {
                "class": concept,
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "sourceNodeId": source_node_id,
                "kind": "segmentation",
            }
            if index < len(masks):
                mask = np.squeeze(masks[index]).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    if len(contour) >= 3:
                        detection["mask"] = [[float(point[0]), float(point[1])] for point in contour]
            detections.append(detection)
    return detections


def filter_detections(workflow: dict, detections: list[dict]) -> list[dict]:
    filter_node = active_filter_node(workflow)
    if not filter_node:
        return detections
    config = filter_node.get("config", {})
    classes = {
        item.strip().lower()
        for item in str(config.get("classes", "")).split(",")
        if item.strip()
    }
    filtered = [
        item for item in detections
        if not classes or item.get("class", "").lower() in classes
    ]
    return filtered if len(filtered) >= int(config.get("minCount", 1)) else []


def detection_color(label: str) -> tuple[int, int, int]:
    palette = (
        (41, 211, 145),
        (56, 189, 248),
        (168, 85, 247),
        (245, 158, 11),
        (239, 68, 68),
        (99, 102, 241),
    )
    index = sum(label.encode("utf-8")) % len(palette)
    return palette[index]


def draw_detections(frame, detections: list[dict], preview: dict) -> None:
    import cv2
    import numpy as np

    config = preview.get("config", {})
    show_boxes = bool(config.get("showBoxes", True))
    show_labels = bool(config.get("showLabels", True))
    show_masks = bool(config.get("showMasks", True))
    mask_opacity = min(0.85, max(0.05, float(config.get("maskOpacity", 0.35))))
    classification_y = 12
    for detection in detections:
        label_text = detection.get("class", "object")
        color = detection_color(label_text)
        label = f"{label_text} {round(float(detection.get('score', 0)) * 100)}%"
        if detection.get("kind") == "classification":
            if show_labels:
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                x = 12
                y = classification_y
                cv2.rectangle(frame, (x, y), (x + text_size[0] + 14, y + text_size[1] + 12), (8, 18, 27), -1)
                cv2.rectangle(frame, (x, y), (x + text_size[0] + 14, y + text_size[1] + 12), color, 2)
                cv2.putText(frame, label, (x + 7, y + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                classification_y += text_size[1] + 18
            continue

        x, y, width, height = [int(round(value)) for value in detection.get("bbox", [0, 0, 0, 0])]
        mask = detection.get("mask")
        if show_masks and mask:
            points = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, mask_opacity, frame, 1 - mask_opacity, 0, dst=frame)
            cv2.polylines(frame, [points], True, color, 2)
        if show_boxes:
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        if show_labels:
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_y = max(0, y - text_size[1] - 8)
            cv2.rectangle(frame, (x, label_y), (x + text_size[0] + 10, label_y + text_size[1] + 8), (8, 18, 27), -1)
            cv2.putText(frame, label, (x + 5, label_y + text_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


class WorkflowRunner:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._status = {
            "running": False,
            "state": "idle",
            "fps": 0,
            "objectCount": 0,
            "model": None,
            "device": "CPU",
            "error": None,
            "events": [],
        }

    def start(self, workflow: dict) -> dict:
        self.stop()
        workflow = normalize_workflow(workflow)
        if not active_input_node(workflow):
            raise ValueError("Input node is disabled.")

        self._stop_event = Event()
        with self._lock:
            self._status.update({
                "running": True,
                "state": "starting",
                "fps": 0,
                "objectCount": 0,
                "model": None,
                "device": "CPU",
                "error": None,
                "events": [],
            })
        self._add_terminal("Workflow launched from browser Run button")
        self._add_terminal("Starting workflow runner")
        self._thread = Thread(target=self._run, args=(workflow,), daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> dict:
        thread = self._thread
        if thread and thread.is_alive():
            self._stop_event.set()
            thread.join(timeout=5)
        self._thread = None
        with self._lock:
            if self._status["state"] != "error":
                self._status.update({"running": False, "state": "stopped", "fps": 0, "objectCount": 0})
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return dict(
                self._status,
                events=list(self._status["events"]),
            )

    def _run(self, workflow: dict) -> None:
        import cv2

        input_source = None
        window_name = "VisoNode"
        last_fps_at = time.monotonic()
        frame_count = 0
        last_alert_at = 0.0
        last_terminal_detection_at = 0.0
        detections: list[dict] = []
        filtered: list[dict] = []
        try:
            input_node = active_input_node(workflow)
            input_source = OpenCVInputSource(cv2, input_node.get("config", {}))
            source_label, actual_width, actual_height = input_source.open()
            self._add_terminal(f"Opened {source_label} ({actual_width}x{actual_height})")

            inference_nodes = active_inference_nodes(workflow)
            if inference_nodes:
                loaded_models = []
                device_labels = []
                for inference_node in inference_nodes:
                    engine = inference_node.get("config", {}).get("engine", "yolo26")
                    if engine not in ("yolo26", "sam3", "rfdetr"):
                        raise RuntimeError(f"Backend runtime does not support inference engine '{engine}'.")
                    if engine == "sam3" and inference_node.get("id") != "segmenter":
                        raise RuntimeError("SAM 3 is available only on the Object Segmentation node.")
                    if engine == "rfdetr" and inference_node.get("id") not in ("detector", "segmenter"):
                        raise RuntimeError("RF-DETR is available only on Object Detection and Object Segmentation nodes.")
                    config = inference_node.get("config", {})
                    model_name = inference_model_name(inference_node)
                    device, device_label = resolve_inference_device(inference_node.get("config", {}))
                    task_labels = {
                        "classifier": "classification",
                        "segmenter": "segmentation",
                    }
                    task_label = task_labels.get(inference_node.get("id"), "detector")
                    if engine == "sam3":
                        task_label = f"SAM 3 {task_label}"
                    elif engine == "rfdetr":
                        task_label = f"RF-DETR {task_label}"
                    self._add_terminal(f"Loading {task_label} model {model_name}")
                    self._add_terminal(f"Using inference device {device_label}")
                    if engine == "sam3":
                        concepts = ", ".join(sam3_concepts(config))
                        self._add_terminal(f"SAM 3 concept prompt(s): {concepts}")
                        load_sam3_processor(config, device)
                    elif engine == "rfdetr":
                        load_rfdetr_model(model_name, device)
                    else:
                        load_yolo_model(model_name, device)
                    loaded_models.append(model_name)
                    device_labels.append(device_label)
                    self._add_terminal(f"Loaded {task_label} model {model_name}")
                device_status = device_labels[0] if len(set(device_labels)) == 1 else "Mixed devices"
                self._set_status(state="running", model=", ".join(loaded_models), device=device_status)
            else:
                self._set_status(state="running")
                self._add_terminal("No inference node is active; streaming input frames only")

            self._add_event("Workflow running", "Python is loading, processing, and displaying frames with OpenCV.")
            self._add_terminal("Workflow running")

            while not self._stop_event.is_set():
                ok, frame = input_source.read()
                if not ok:
                    raise RuntimeError("Input frame could not be read.")

                frame_detections = detections if input_source.is_static else []
                frame_filtered = filtered if input_source.is_static else []
                inference_nodes = active_inference_nodes(workflow)
                if inference_nodes:
                    detections = []
                    for inference_node in inference_nodes:
                        config = inference_node.get("config", {})
                        threshold = float(config.get("threshold", 0.55))
                        engine = config.get("engine", "yolo26")
                        if engine == "sam3":
                            node_detections = run_sam3_frame(frame, inference_node)
                        elif engine == "rfdetr":
                            node_detections = run_rfdetr_frame(frame, inference_node)
                        elif engine == "yolo26":
                            node_detections = run_yolo26_frame(frame, inference_node)
                        else:
                            raise RuntimeError(f"Backend runtime does not support inference engine '{engine}'.")
                        detections.extend(
                            item for item in node_detections
                            if float(item.get("score", 0)) >= threshold
                        )
                    filter_candidates = (
                        inference_results_for_node(workflow, "filter", detections)
                        if active_filter_node(workflow)
                        else detections
                    )
                    filtered = filter_detections(workflow, filter_candidates)
                    frame_detections = detections
                    frame_filtered = filtered

                    now = time.monotonic()
                    if detections and now - last_terminal_detection_at >= 1:
                        last_terminal_detection_at = now
                        labels = ", ".join(
                            f"{item.get('class', 'object')}:{float(item.get('score', 0)):.2f}"
                            for item in detections[:6]
                        )
                        if len(detections) > 6:
                            labels = f"{labels}, +{len(detections) - 6} more"
                        self._add_terminal(f"Detected {len(detections)} object(s): {labels}")
                else:
                    now = time.monotonic()

                preview = enabled_node(workflow, "preview")
                display_frame = frame.copy()
                preview_detections = detections_for_node(workflow, "preview", frame_detections, frame_filtered)
                preview_shown = False
                if preview and has_active_path(workflow, source_node_id(workflow), "preview"):
                    draw_detections(display_frame, preview_detections, preview)
                    cv2.imshow(window_name, display_frame)
                    preview_shown = True
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        self._stop_event.set()
                else:
                    cv2.waitKey(1)

                alert_detections = detections_for_node(workflow, "alert", frame_detections, frame_filtered)
                alert_node = enabled_node(workflow, "alert")
                if alert_node and alert_detections and has_active_path(workflow, source_node_id(workflow), "alert"):
                    cooldown = max(0, float(alert_node.get("config", {}).get("cooldownSeconds", 5)))
                    if now - last_alert_at >= cooldown:
                        last_alert_at = now
                        labels = ", ".join(item.get("class", "object") for item in alert_detections)
                        self._add_event(
                            str(alert_node.get("config", {}).get("message") or "Detected target object"),
                            f"{len(alert_detections)} match(es): {labels}",
                        )
                        self._add_terminal(f"Alert emitted for {len(alert_detections)} match(es): {labels}")

                frame_count += 1
                elapsed = now - last_fps_at
                if elapsed >= 1:
                    visible_count = len(preview_detections) or len(alert_detections)
                    self._set_status(
                        state="running",
                        fps=round(frame_count / elapsed),
                        objectCount=visible_count,
                    )
                    frame_count = 0
                    last_fps_at = now
        except Exception as exc:
            self._set_status(running=False, state="error", error=str(exc))
            self._add_event("Workflow error", str(exc))
            self._add_terminal(f"ERROR: {exc}")
        finally:
            if input_source is not None:
                input_source.release()
            cv2.destroyAllWindows()
            if self._stop_event.is_set():
                self._set_status(running=False, state="stopped", fps=0, objectCount=0)
                self._add_event("Workflow stopped", "Python runtime stopped and released the input source.")
                self._add_terminal("Workflow stopped")

    def _set_status(self, **updates) -> None:
        with self._lock:
            self._status.update(updates)

    def _add_event(self, title: str, detail: str) -> None:
        event = {
            "time": time.strftime("%H:%M:%S"),
            "title": title,
            "detail": detail,
        }
        with self._lock:
            self._status["events"].insert(0, event)
            del self._status["events"][MAX_EVENTS:]

    def _add_terminal(self, line: str) -> None:
        terminal_log(line)


runner = WorkflowRunner()


class NoCacheHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/app/version":
            json_response(self, 200, {
                **VERSION_INFO,
                **git_version_info(),
                "versionFile": VERSION_FILE.name,
            })
            return
        if path == "/api/yolo26/models":
            json_response(self, 200, {
                "models": list(YOLO26_MODELS),
                "detectionModels": list(YOLO26_DETECTION_MODELS),
                "segmentationModels": list(YOLO26_SEGMENTATION_MODELS),
                "classificationModels": list(YOLO26_CLASSIFICATION_MODELS),
                "rfdetrDetectionModels": list(RFDETR_DETECTION_MODELS),
                "rfdetrSegmentationModels": list(RFDETR_SEGMENTATION_MODELS),
                "samModels": list(SAM3_MODELS),
            })
            return
        if path == "/api/workflow/status":
            json_response(self, 200, runner.status())
            return
        if path == "/api/runtime/devices":
            json_response(self, 200, runtime_devices())
            return
        if path == "/api/terminal/logs":
            cursor_raw = parse_qs(parsed.query).get("since", ["0"])[0]
            try:
                cursor = int(cursor_raw)
            except ValueError:
                cursor = 0
            json_response(self, 200, terminal.since(cursor))
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/workflow/start":
                payload = read_json_body(self)
                workflow = payload.get("workflow")
                if not isinstance(workflow, dict):
                    raise ValueError("Missing workflow payload.")
                json_response(self, 200, runner.start(workflow))
                return
            if path == "/api/workflow/stop":
                json_response(self, 200, runner.stop())
                return
            if path == "/api/terminal/open":
                message = open_external_terminal()
                terminal_log(f"{message} in {ROOT}")
                json_response(self, 200, {"message": message, "cwd": str(ROOT)})
                return
            if path == "/api/file-dialog/open":
                selected_path = open_file_dialog()
                if selected_path:
                    terminal_log(f"Selected input file {selected_path}")
                json_response(self, 200, {"path": selected_path})
                return
            json_response(self, 404, {"error": "Not found"})
        except Exception as exc:
            json_response(self, 400, {"error": str(exc)})

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VisoNode, the no-code computer vision workflow builder."
    )
    parser.add_argument("--version", action="version", version=f"VisoNode {APP_VERSION}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), NoCacheHandler)
    terminal_log(f"VisoNode v{APP_VERSION} running at http://{args.host}:{args.port}")
    terminal_log("The browser edits the workflow; Python owns capture, inference, and OpenCV display.")
    terminal_log("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
