"""
detector_pi5.py
===============
Real-time YOLOv8n Object Detection — Raspberry Pi 5 + PiCamera2
Optimized for maximum FPS and minimum latency on Pi 5 hardware.
"""

import cv2
import time
import threading
import queue
import sys
import argparse
import numpy as np
from pathlib import Path

# ── MQTT ──────────────────────────────────────
import paho.mqtt.client as mqtt
import json
import ssl
from datetime import datetime

BROKER    = "0c841676725b4046a512c2491cd5da74.s1.eu.hivemq.cloud"
PORT      = 8883
USERNAME  = "Mariam"
PASSWORD  = "Mariam23"
DEVICE_ID = "rpi01"

def setup_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    client.connect(BROKER, PORT)
    client.loop_start()
    print("[MQTT] Connected ✅")
    return client

def publish_if_detected(client, class_name: str, confidence: float):
    topic   = f"ai/{DEVICE_ID}/{class_name}"
    payload = json.dumps({
        "confidence": round(confidence, 4),
        "timestamp":  datetime.utcnow().isoformat()
    })
    client.publish(topic, payload)
    print(f"[MQTT] Sent {topic} → {confidence:.2f}")
# ─────────────────────────────────────────────


# ── Import guard: prefer ONNX Runtime, fall back to Ultralytics ──────────────
try:
    import onnxruntime as ort
    BACKEND = 'onnx'
    print("[BACKEND] ONNX Runtime detected — using fast ONNX inference")
except ImportError:
    from ultralytics import YOLO
    BACKEND = 'ultralytics'
    print("[BACKEND] ONNX Runtime not found — falling back to Ultralytics PyTorch")

# ── PiCamera2 import guard ───────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("[CAMERA] picamera2 not found — falling back to cv2.VideoCapture")


# ══════════════════════════════════════════════
#  GLOBAL CONFIGURATION
# ══════════════════════════════════════════════

CONFIG = {
    'model_path':      'best.onnx',
    'model_path_pt':   'best.pt',
    'conf_threshold':   0.35,
    'iou_threshold':    0.45,
    'camera_index':     0,
    'cam_width':        640,
    'cam_height':       480,
    'inf_width':        320,
    'inf_height':       320,
    'skip_frames':      2,
    'disp_width':       800,
    'disp_height':      600,
    'window_name':      'YOLOv8n — Safety Detection (Q to quit)',
    'headless':         False,
    'queue_size':       1,
}

CLASS_NAMES = ['food', 'smoking', 'drinking', 'phone', 'danger', 'fire']

CLASS_COLORS = {
    'food':     (71,  99,  255),
    'smoking':  (0,   165, 255),
    'drinking': (255, 144, 30),
    'phone':    (50,  205, 50),
    'danger':   (211, 0,   148),
    'fire':     (0,   69,  255),
}


# ══════════════════════════════════════════════
#  MODULE 1 — CAMERA INITIALIZATION
# ══════════════════════════════════════════════

def initialize_camera(camera_index: int = 0):
    if PICAMERA2_AVAILABLE:
        print("[CAMERA] Initializing PiCamera2 (CSI) ...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={
                "size":   (CONFIG['cam_width'], CONFIG['cam_height']),
                "format": "BGR888",
            },
            buffer_count=2,
        )
        picam2.configure(config)
        picam2.set_controls({
            "AeEnable":            True,
            "AwbEnable":           True,
            "FrameDurationLimits": (33333, 33333),
        })
        picam2.start()
        time.sleep(0.5)
        print(f"[CAMERA] PiCamera2 ready → {CONFIG['cam_width']}x{CONFIG['cam_height']} @ 30fps")
        return picam2

    else:
        print(f"[CAMERA] Opening VideoCapture device={camera_index} ...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG['cam_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['cam_height'])
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] VideoCapture ready → {w}x{h}")
        return cap


# ══════════════════════════════════════════════
#  MODULE 2 — MODEL LOADING
# ══════════════════════════════════════════════

def load_model(model_path: str):
    if BACKEND == 'onnx':
        onnx_path = Path(model_path)
        if not onnx_path.exists():
            onnx_path = Path(model_path).with_suffix('.onnx')
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: '{model_path}'")

        print(f"[MODEL] Loading ONNX model: {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=['CPUExecutionProvider'],
        )
        model = OnnxModel(session)

    else:
        pt_path = Path(model_path)
        if not pt_path.exists():
            pt_path = Path(CONFIG['model_path_pt'])
        if not pt_path.exists():
            raise FileNotFoundError(f"Model not found: '{model_path}'")

        print(f"[MODEL] Loading Ultralytics model: {pt_path}")
        model = YOLO(str(pt_path))
        model.to('cpu')

    print("[MODEL] Warming up ...")
    dummy = np.zeros((CONFIG['inf_height'], CONFIG['inf_width'], 3), dtype=np.uint8)
    for _ in range(3):
        run_inference(model, dummy)
    print("[MODEL] Ready ✓\n")

    return model


class OnnxModel:
    def __init__(self, session):
        self.session     = session
        self.input_name  = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis]
        return nchw.astype(np.float32) / 255.0

    def decode(self, output: np.ndarray, conf_thresh: float, iou_thresh: float) -> list:
        preds = output[0]
        boxes, scores, class_ids = [], [], []
        iw = CONFIG['inf_width']
        ih = CONFIG['inf_height']

        for i in range(preds.shape[1]):
            class_scores = preds[4:, i]
            cls_id       = int(np.argmax(class_scores))
            conf         = float(class_scores[cls_id])
            if conf < conf_thresh:
                continue
            cx, cy, w, h = preds[:4, i]
            x1 = (cx - w / 2) * iw
            y1 = (cy - h / 2) * ih
            x2 = (cx + w / 2) * iw
            y2 = (cy + h / 2) * ih
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices    = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        detections = []
        for idx in (indices.flatten() if len(indices) else []):
            x, y, w, h = boxes[idx]
            cls_id = class_ids[idx]
            name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
            detections.append({
                'class_id':   cls_id,
                'class_name': name,
                'confidence': scores[idx],
                'bbox':       (int(x), int(y), int(x + w), int(y + h)),
            })
        return detections


# ══════════════════════════════════════════════
#  MODULE 3 — FRAME PROCESSING
# ══════════════════════════════════════════════

def process_frame(frame: np.ndarray) -> tuple:
    display_frame   = frame.copy()
    inference_frame = cv2.resize(
        frame,
        (CONFIG['inf_width'], CONFIG['inf_height']),
        interpolation=cv2.INTER_NEAREST,
    )
    return inference_frame, display_frame


# ══════════════════════════════════════════════
#  MODULE 4 — INFERENCE
# ══════════════════════════════════════════════

def run_inference(model, frame: np.ndarray) -> list:
    if BACKEND == 'onnx':
        tensor  = model.preprocess(frame)
        outputs = model.session.run([model.output_name], {model.input_name: tensor})
        return model.decode(outputs[0], CONFIG['conf_threshold'], CONFIG['iou_threshold'])

    else:
        results    = model.predict(
            source=frame, conf=CONFIG['conf_threshold'],
            iou=CONFIG['iou_threshold'], verbose=False,
            device='cpu', half=False, imgsz=CONFIG['inf_width'],
        )
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
                detections.append({
                    'class_id':   cls_id,
                    'class_name': name,
                    'confidence': conf_val,
                    'bbox':       (x1, y1, x2, y2),
                })
        return detections


# ══════════════════════════════════════════════
#  MODULE 5 — DISPLAY OUTPUT
# ══════════════════════════════════════════════

def display_output(display_frame, detections, fps, frame_count, is_inference_frame) -> bool:
    if CONFIG['headless']:
        if detections:
            items = [f"{d['class_name']}({d['confidence']:.0%})" for d in detections]
            print(f"  Frame {frame_count:6d} │ {' │ '.join(items)}")
        return True

    dh, dw  = display_frame.shape[:2]
    scale_x = dw / CONFIG['inf_width']
    scale_y = dh / CONFIG['inf_height']

    for det in detections:
        name  = det['class_name']
        conf  = det['confidence']
        color = CLASS_COLORS.get(name, (200, 200, 200))
        x1, y1, x2, y2 = det['bbox']
        x1 = int(x1 * scale_x); x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y); y2 = int(y2 * scale_y)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(display_frame, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    fps_color = (0, 255, 0) if fps >= 10 else (0, 200, 255) if fps >= 6 else (0, 0, 255)
    cv2.putText(display_frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, fps_color, 2, cv2.LINE_AA)
    cv2.putText(display_frame, f"Detected: {len(detections)}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    inf_label = "● INFER" if is_inference_frame else "○ SKIP"
    inf_color = (0, 255, 255) if is_inference_frame else (120, 120, 120)
    cv2.putText(display_frame, inf_label,
                (dw - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, inf_color, 1, cv2.LINE_AA)
    cv2.putText(display_frame, f"YOLOv8n | {BACKEND.upper()}",
                (dw - 160, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "Press Q to quit",
                (10, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    output = cv2.resize(display_frame, (CONFIG['disp_width'], CONFIG['disp_height']),
                        interpolation=cv2.INTER_NEAREST)
    cv2.imshow(CONFIG['window_name'], output)
    return (cv2.waitKey(1) & 0xFF) != ord('q')


# ══════════════════════════════════════════════
#  CAMERA THREAD
# ══════════════════════════════════════════════

class CameraThread(threading.Thread):
    def __init__(self, cap, queue_size: int = 1):
        super().__init__(daemon=True)
        self.cap          = cap
        self.q            = queue.Queue(maxsize=queue_size)
        self.stopped      = False
        self._is_picamera = PICAMERA2_AVAILABLE and isinstance(cap, Picamera2)

    def run(self):
        while not self.stopped:
            try:
                if self._is_picamera:
                    frame = self.cap.capture_array()
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.005)
                        continue

                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)

            except Exception as e:
                print(f"[CAMERA THREAD] Error: {e}")
                time.sleep(0.01)

    def read(self) -> tuple:
        try:
            return True, self.q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self):
        self.stopped = True
        if self._is_picamera:
            try:
                self.cap.stop()
            except Exception:
                pass


# ══════════════════════════════════════════════
#  PROCESS PRIORITY
# ══════════════════════════════════════════════

def set_process_priority():
    try:
        import os
        os.nice(-10)
        print("[PERF] Process priority raised (nice -10)")
    except (AttributeError, PermissionError):
        pass


# ══════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ══════════════════════════════════════════════

def main(args):
    print("=" * 55)
    print("  YOLOv8n Real-time Safety Detection")
    print("  Raspberry Pi 5 | PiCamera2 | ONNX Runtime")
    print("=" * 55)

    set_process_priority()

    # ── MQTT Setup ────────────────────────────
    mqtt_client = setup_mqtt()

    # ── Load model ────────────────────────────
    try:
        model = load_model(args.model)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Initialize camera ─────────────────────
    try:
        cap = initialize_camera(args.camera)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Start camera thread ───────────────────
    cam = CameraThread(cap, queue_size=CONFIG['queue_size'])
    cam.start()
    print("[INFO] Camera thread started.")
    print(f"[INFO] Backend: {BACKEND.upper()}")
    print(f"[INFO] Headless: {CONFIG['headless']}")
    print("[INFO] Detection loop started — press Q to quit.\n")

    # ── Loop state ────────────────────────────
    frame_count = 0
    detections  = []
    fps         = 0.0
    fps_counter = 0
    fps_timer   = time.perf_counter()

    # ── Main loop ─────────────────────────────
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame_count  += 1
        is_inf_frame  = (frame_count % CONFIG['skip_frames'] == 0)

        if is_inf_frame:
            inf_frame, display_frame = process_frame(frame)
            detections = run_inference(model, inf_frame)

            # ── لما يلاقي حاجة يطبع ويبعت MQTT ──
            if detections:
                items = [
                    f"{d['class_name']}({d['confidence']:.0%})"
                    for d in detections
                ]
                print(f"  Frame {frame_count:6d} │ {' │ '.join(items)}")

                for d in detections:
                    publish_if_detected(mqtt_client, d['class_name'], d['confidence'])

        else:
            _, display_frame = process_frame(frame)

        fps_counter += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 0.5:
            fps         = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.perf_counter()

        keep_running = display_output(
            display_frame, detections, fps, frame_count, is_inf_frame
        )
        if not keep_running:
            print("\n[INFO] Q pressed — shutting down.")
            break

    # ── Shutdown ──────────────────────────────
    print("[INFO] Stopping camera thread ...")
    cam.stop()
    cam.join(timeout=2)
    if not (PICAMERA2_AVAILABLE and isinstance(cap, Picamera2)):
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] All done.")


# ══════════════════════════════════════════════
#  ENTRY POINT + CLI
# ══════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLOv8n Safety Detection — Raspberry Pi 5'
    )
    parser.add_argument('--model',    type=str,   default=CONFIG['model_path'])
    parser.add_argument('--camera',   type=int,   default=CONFIG['camera_index'])
    parser.add_argument('--conf',     type=float, default=CONFIG['conf_threshold'])
    parser.add_argument('--skip',     type=int,   default=CONFIG['skip_frames'])
    parser.add_argument('--size',     type=int,   default=CONFIG['inf_width'])
    parser.add_argument('--headless', action='store_true')

    args = parser.parse_args()

    CONFIG['conf_threshold'] = args.conf
    CONFIG['skip_frames']    = args.skip
    CONFIG['inf_width']      = args.size
    CONFIG['inf_height']     = args.size
    CONFIG['headless']       = args.headless

    main(args)