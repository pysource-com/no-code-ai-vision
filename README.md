<div align="center">

# 🟢 VisoNode

### Build computer vision workflows without writing any code.

Connect a camera or video to an AI detection, segmentation, or classification model and watch the results live —
all by dragging and linking boxes in a native desktop app. No machine learning experience required.

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/model-YOLO26%20%2B%20RF--DETR%20%2B%20SAM3-purple)
![Runs locally](https://img.shields.io/badge/runs-100%25%20local-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

[**Quick start**](#-quick-start) · [**Highlights**](#-highlights) · [**How it works**](#-how-it-works) · [**The nodes**](#-the-nodes) · [**Roadmap**](#-roadmap)

![The VisoNode workflow editor with all nodes running](docs/workflow-editor.png)

</div>

---

## ⚡ Quick start

```powershell
# 1. Install dependencies (CPU build — works on any computer)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Start the desktop app
.\.venv\Scripts\python.exe app.py
```

The native **PySide6 desktop window** opens with the node editor. Configure the nodes, click **Run**,
and the detections appear in a live preview window embedded in the app.
Press **Stop**, or close the preview window (`q` / `Esc`) to stop.

> The editor is now a native desktop app — there is no browser or local web server involved.
> Your graph is saved to `workflow.json` next to the app and reloaded on the next launch.
> The legacy browser editor (`python main.py`, served at `http://127.0.0.1:8000`) still works if you prefer it.

> **Have an NVIDIA GPU?** Run `.\scripts\install-gpu.ps1` instead of step 1 for a big speed-up.
> See [GPU acceleration](#gpu-acceleration) below.

---

## Versioning

The running version is shown in the top bar next to **VisoNode**.

The version source of truth is [`web/version.json`](web/version.json). Update that file before a release. Python also reads it for:

```powershell
.\.venv\Scripts\python.exe main.py --version
```

When the backend is restarted, `/api/app/version` also reports the app version and current Git commit.

---

## ✨ Highlights

- **No code.** Build the whole pipeline by connecting nodes in the browser.
- **100% local.** Your camera and video never leave your computer — nothing is sent to the cloud.
- **State-of-the-art detection, segmentation, and classification.** Powered by Ultralytics **YOLO26**, Roboflow **RF-DETR**, and Meta's official **SAM 3** concept segmentation.
- **CPU or GPU.** Works on any machine; uses your NVIDIA GPU automatically when available.
- **Live feedback.** See frame rate, object count, and the active device in real time.
- **Flexible inputs.** Webcams, stream URLs, capture devices, or local image/video files.

A workflow is just a chain of nodes:

```text
Input  →  Object Detection, Object Segmentation, or Object Classification  →  Class Filter  →  OpenCV Preview  →  Alert Output
```

Each node does one job: read frames, find objects, segment object masks, classify the frame, keep only the classes you care about,
draw the results, and log alerts. The results appear in a separate window, with boxes, labels, classifications, and masks drawn over the video:

![Object detection output with bounding boxes over a street scene](docs/detection-output.png)

---

## 🔍 How it works

VisoNode runs entirely on **your own computer**:

- The **browser** is just the editor. You use it to lay out the nodes and change their settings.
- **Python** does the real work behind the scenes: opening the camera or file, running the
  selected AI model, filtering results, drawing boxes, and logging alerts.

When you click **Run**, the browser hands your workflow to Python, and Python opens a native
preview window showing the live detections. In the editor, each node turns green and shows
**RUNNING**, and the top bar reports the live frame rate, object count, and active device.

> **First run note:** the first time you use Object Detection, it automatically downloads the
> YOLO26 model weights (for example `yolo26n.pt`). This happens once and may take a moment.
> RF-DETR also downloads its selected starter weights the first time you run an RF-DETR model.
> SAM 3 uses Meta's official implementation. Request access to `facebook/sam3` on Hugging Face,
> run `hf auth login`, then choose **SAM 3 concept segmentation**. You can also set a local
> checkpoint path in Object Segmentation.
>
> Meta's official SAM 3 prerequisites are stricter than YOLO26: Python 3.12+, PyTorch 2.7+,
> and a CUDA-compatible GPU with CUDA 12.6+.

### GPU acceleration

The [Quick start](#-quick-start) installs the CPU build, which works on any computer but is
slower. If you have an **NVIDIA GPU**, install the GPU-enabled build instead. This script
installs a CUDA build of PyTorch, installs the app, and runs a quick test to confirm the GPU
is detected:

```powershell
.\scripts\install-gpu.ps1
```

By default it installs the `cu128` driver wheels. To target a different CUDA version:

```powershell
.\scripts\install-gpu.ps1 -CudaWheel cu126
```

To check whether an existing install can see your GPU:

```powershell
.\scripts\check-gpu.ps1
```

---

## 🧩 The nodes

| Node | What it does |
| --- | --- |
| **Input** | Chooses where frames come from. *Camera mode* takes an OpenCV camera index, a stream URL, or a capture source plus resolution. *File mode* takes a local image or video path and can loop videos. |
| **Object Detection** | Runs an Ultralytics **YOLO26** or Roboflow **RF-DETR** detection model. Lets you pick the engine, model size, confidence threshold, and CPU/GPU device. |
| **Object Segmentation** | Runs an Ultralytics **YOLO26-seg**, Roboflow **RF-DETR-Seg**, or Meta **SAM 3** segmentation model. SAM 3 takes comma-separated noun phrases such as `person, car, bottle`; all engines return masks or boxes, labels, and confidence scores downstream. |
| **Object Classification** | Runs an Ultralytics **YOLO26-cls** image classification model and passes the top label and confidence score downstream. |
| **Class Filter** | Keeps only the object types you list — for example `person, car, dog`. |
| **OpenCV Preview** | Draws masks, bounding boxes, classification labels, and object labels in a native preview window. |
| **Alert Output** | Logs detection events on the backend, with a cooldown so you aren't flooded. |

### Choosing CPU or GPU

The **Object Detection** node has a **Device** setting:

| Setting | What it does |
| --- | --- |
| **Auto** | Uses your NVIDIA GPU if one is available, otherwise falls back to CPU. |
| **CPU** | Always runs on the CPU. Works everywhere, slower. |
| **CUDA device** | Forces the GPU. Requires a working CUDA + PyTorch install, and reports an error if the GPU can't be used. |

If you're not sure, leave it on **Auto**.

---

## 🗺️ Roadmap

Planned improvements:

- Save and load workflows from a file.
- Built-in presets for RTSP / IP cameras.
- Use a monitor / screen as an input source.
- Extra inference nodes for ONNX and TensorRT.
- More output nodes: webhooks, database logging, and snapshot saving.

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0** — the same
open-source license used by Ultralytics. See [LICENSE](LICENSE).
