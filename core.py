"""Backend engine for the VisoNode desktop GUI.

This module reuses the inference / capture / drawing helpers that already live in
``main.py`` (the original browser backend) and wraps them in a callback-driven
workflow runner. Instead of pushing frames to a separate OpenCV window and the
status to an HTTP polling endpoint, the runner forwards everything to plain
Python callbacks so the PySide GUI can render it natively.

The callbacks are invoked from the runner's worker thread, so the GUI is
expected to marshal them onto the Qt thread (e.g. by passing bound Qt
``Signal.emit`` methods as the callbacks).
"""

from __future__ import annotations

import time
from threading import Event, Lock, Thread
from typing import Callable

import main as engine

# Re-export the pieces the GUI needs so it only has to import from ``core``.
APP_VERSION = engine.APP_VERSION
VERSION_INFO = engine.VERSION_INFO
INPUT_NODE_ID = engine.INPUT_NODE_ID
YOLO26_DETECTION_MODELS = engine.YOLO26_DETECTION_MODELS
YOLO26_SEGMENTATION_MODELS = engine.YOLO26_SEGMENTATION_MODELS
YOLO26_CLASSIFICATION_MODELS = engine.YOLO26_CLASSIFICATION_MODELS
RFDETR_DETECTION_MODELS = engine.RFDETR_DETECTION_MODELS
RFDETR_SEGMENTATION_MODELS = engine.RFDETR_SEGMENTATION_MODELS
SAM3_MODELS = engine.SAM3_MODELS

git_version_info = engine.git_version_info
runtime_devices = engine.runtime_devices


LogCallback = Callable[[str], None]
EventCallback = Callable[[str, str], None]
StatusCallback = Callable[[dict], None]
FrameCallback = Callable[[object, bool], None]


def _noop(*_args, **_kwargs) -> None:
    pass


class WorkflowRunner:
    """Runs a workflow on a background thread and reports through callbacks.

    Mirrors the control flow of ``main.WorkflowRunner`` but is decoupled from the
    HTTP server and from ``cv2.imshow``. Frames are handed to ``on_frame`` so the
    GUI can paint them; logs, events, and status go to their respective
    callbacks.
    """

    def __init__(
        self,
        on_log: LogCallback | None = None,
        on_event: EventCallback | None = None,
        on_status: StatusCallback | None = None,
        on_frame: FrameCallback | None = None,
    ) -> None:
        self._on_log = on_log or _noop
        self._on_event = on_event or _noop
        self._on_status = on_status or _noop
        self._on_frame = on_frame or _noop

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
        }

    # -- public control -------------------------------------------------
    def is_running(self) -> bool:
        with self._lock:
            return bool(self._status["running"])

    def start(self, workflow: dict) -> None:
        self.stop()
        workflow = engine.normalize_workflow(workflow)
        if not engine.active_input_node(workflow):
            raise ValueError("Input node is disabled.")

        self._stop_event = Event()
        self._set_status(
            running=True,
            state="starting",
            fps=0,
            objectCount=0,
            model=None,
            device="CPU",
            error=None,
        )
        self._emit_log("Workflow launched from desktop Run button")
        self._emit_log("Starting workflow runner")
        self._thread = Thread(target=self._run, args=(workflow,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread and thread.is_alive():
            self._stop_event.set()
            thread.join(timeout=5)
        self._thread = None
        with self._lock:
            if self._status["state"] != "error":
                self._status.update(
                    {"running": False, "state": "stopped", "fps": 0, "objectCount": 0}
                )
                snapshot = dict(self._status)
        # emit outside the lock to avoid re-entrancy
        if not (thread and thread.is_alive()):
            self._on_status(self.status())

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    # -- worker ---------------------------------------------------------
    def _run(self, workflow: dict) -> None:
        import cv2

        input_source = None
        last_fps_at = time.monotonic()
        frame_count = 0
        last_alert_at = 0.0
        last_terminal_detection_at = 0.0
        detections: list[dict] = []
        filtered: list[dict] = []
        source_id = engine.source_node_id(workflow)
        try:
            input_node = engine.active_input_node(workflow)
            input_source = engine.OpenCVInputSource(cv2, input_node.get("config", {}))
            source_label, actual_width, actual_height = input_source.open()
            self._emit_log(f"Opened {source_label} ({actual_width}x{actual_height})")

            inference_nodes = engine.active_inference_nodes(workflow)
            if inference_nodes:
                loaded_models = []
                device_labels = []
                for inference_node in inference_nodes:
                    config = inference_node.get("config", {})
                    eng = config.get("engine", "yolo26")
                    if eng not in ("yolo26", "sam3", "rfdetr"):
                        raise RuntimeError(
                            f"Backend runtime does not support inference engine '{eng}'."
                        )
                    if eng == "sam3" and inference_node.get("id") != "segmenter":
                        raise RuntimeError(
                            "SAM 3 is available only on the Object Segmentation node."
                        )
                    if eng == "rfdetr" and inference_node.get("id") not in ("detector", "segmenter"):
                        raise RuntimeError(
                            "RF-DETR is available only on Object Detection and Object Segmentation nodes."
                        )
                    model_name = engine.inference_model_name(inference_node)
                    device, device_label = engine.resolve_inference_device(config)
                    task_labels = {"classifier": "classification", "segmenter": "segmentation"}
                    task_label = task_labels.get(inference_node.get("id"), "detector")
                    if eng == "sam3":
                        task_label = f"SAM 3 {task_label}"
                    elif eng == "rfdetr":
                        task_label = f"RF-DETR {task_label}"
                    self._emit_log(f"Loading {task_label} model {model_name}")
                    self._emit_log(f"Using inference device {device_label}")
                    if eng == "sam3":
                        concepts = ", ".join(engine.sam3_concepts(config))
                        self._emit_log(f"SAM 3 concept prompt(s): {concepts}")
                        engine.load_sam3_processor(config, device)
                    elif eng == "rfdetr":
                        engine.load_rfdetr_model(model_name, device)
                    else:
                        engine.load_yolo_model(model_name, device)
                    loaded_models.append(model_name)
                    device_labels.append(device_label)
                    self._emit_log(f"Loaded {task_label} model {model_name}")
                device_status = (
                    device_labels[0] if len(set(device_labels)) == 1 else "Mixed devices"
                )
                self._set_status(
                    state="running", model=", ".join(loaded_models), device=device_status
                )
            else:
                self._set_status(state="running")
                self._emit_log("No inference node is active; streaming input frames only")

            self._emit_event(
                "Workflow running",
                "Python is loading, processing, and displaying frames natively.",
            )
            self._emit_log("Workflow running")

            while not self._stop_event.is_set():
                ok, frame = input_source.read()
                if not ok:
                    raise RuntimeError("Input frame could not be read.")

                frame_detections = detections if input_source.is_static else []
                frame_filtered = filtered if input_source.is_static else []
                inference_nodes = engine.active_inference_nodes(workflow)
                now = time.monotonic()
                if inference_nodes:
                    detections = []
                    for inference_node in inference_nodes:
                        config = inference_node.get("config", {})
                        threshold = float(config.get("threshold", 0.55))
                        eng = config.get("engine", "yolo26")
                        if eng == "sam3":
                            node_detections = engine.run_sam3_frame(frame, inference_node)
                        elif eng == "rfdetr":
                            node_detections = engine.run_rfdetr_frame(frame, inference_node)
                        elif eng == "yolo26":
                            node_detections = engine.run_yolo26_frame(frame, inference_node)
                        else:
                            raise RuntimeError(
                                f"Backend runtime does not support inference engine '{eng}'."
                            )
                        detections.extend(
                            item
                            for item in node_detections
                            if float(item.get("score", 0)) >= threshold
                        )
                    filter_candidates = (
                        engine.inference_results_for_node(workflow, "filter", detections)
                        if engine.active_filter_node(workflow)
                        else detections
                    )
                    filtered = engine.filter_detections(workflow, filter_candidates)
                    frame_detections = detections
                    frame_filtered = filtered

                    if detections and now - last_terminal_detection_at >= 1:
                        last_terminal_detection_at = now
                        labels = ", ".join(
                            f"{item.get('class', 'object')}:{float(item.get('score', 0)):.2f}"
                            for item in detections[:6]
                        )
                        if len(detections) > 6:
                            labels = f"{labels}, +{len(detections) - 6} more"
                        self._emit_log(f"Detected {len(detections)} object(s): {labels}")

                preview = engine.enabled_node(workflow, "preview")
                display_frame = frame.copy()
                preview_detections = engine.detections_for_node(
                    workflow, "preview", frame_detections, frame_filtered
                )
                preview_shown = False
                if preview and engine.has_active_path(workflow, source_id, "preview"):
                    engine.draw_detections(display_frame, preview_detections, preview)
                    preview_shown = True
                self._on_frame(display_frame, preview_shown)

                alert_detections = engine.detections_for_node(
                    workflow, "alert", frame_detections, frame_filtered
                )
                alert_node = engine.enabled_node(workflow, "alert")
                if (
                    alert_node
                    and alert_detections
                    and engine.has_active_path(workflow, source_id, "alert")
                ):
                    cooldown = max(
                        0, float(alert_node.get("config", {}).get("cooldownSeconds", 5))
                    )
                    if now - last_alert_at >= cooldown:
                        last_alert_at = now
                        labels = ", ".join(
                            item.get("class", "object") for item in alert_detections
                        )
                        self._emit_event(
                            str(
                                alert_node.get("config", {}).get("message")
                                or "Detected target object"
                            ),
                            f"{len(alert_detections)} match(es): {labels}",
                        )
                        self._emit_log(
                            f"Alert emitted for {len(alert_detections)} match(es): {labels}"
                        )

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
        except Exception as exc:  # noqa: BLE001 - surfaced to the GUI
            self._set_status(running=False, state="error", error=str(exc))
            self._emit_event("Workflow error", str(exc))
            self._emit_log(f"ERROR: {exc}")
        finally:
            if input_source is not None:
                input_source.release()
            if self._stop_event.is_set():
                self._set_status(running=False, state="stopped", fps=0, objectCount=0)
                self._emit_event(
                    "Workflow stopped", "Python runtime stopped and released the input source."
                )
                self._emit_log("Workflow stopped")

    # -- emit helpers ---------------------------------------------------
    def _set_status(self, **updates) -> None:
        with self._lock:
            self._status.update(updates)
            snapshot = dict(self._status)
        self._on_status(snapshot)

    def _emit_log(self, line: str) -> None:
        self._on_log(line)

    def _emit_event(self, title: str, detail: str) -> None:
        self._on_event(title, detail)
