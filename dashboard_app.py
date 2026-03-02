from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, send_from_directory
import json
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from datetime import datetime
import threading
import time
import os
import hashlib
import sqlite3
from functools import wraps
from firebase_config import firebase_config
from collections import deque
from werkzeug.utils import secure_filename
import uuid
import queue
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import torch
except Exception as _torch_err:
    torch = None
    print(f"⚠️ Torch not available ({_torch_err}); YOLO will run on CPU.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DOWNLOADS_DIR = os.path.join(BASE_DIR, 'downloads')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
SNAPSHOTS_DIR = os.path.join(BASE_DIR, 'snapshots')
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

CUDA_AVAILABLE = False
DEVICE = 'cpu'
USE_HALF = False
if torch is not None:
    try:
        CUDA_AVAILABLE = torch.cuda.is_available()
        DEVICE = 0 if CUDA_AVAILABLE else 'cpu'
        USE_HALF = True if CUDA_AVAILABLE else False
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception as _cuda_err:
        print(f"⚠️ Torch threading/CUDA init issue: {_cuda_err}")

def init_db():
    try:
        if firebase_config.db is not None:
            print("✅ Using Firebase for user storage")
            return
    except Exception as e:
        print(f"⚠️ Firebase not available, using SQLite fallback: {e}")
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
    print("✅ Using SQLite for user storage")

def load_config():
    try:
        cfg_path = os.path.join(BASE_DIR, 'config.json')
        if not os.path.exists(cfg_path):
            alt = os.path.join(ROOT_DIR, 'config.json')
            cfg_path = alt if os.path.exists(alt) else cfg_path
        with open(cfg_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "camera_settings": {"camera_index": 0, "width": 1280, "height": 720},
            "detection_settings": {"model_path": "yolov8s.pt", "grid_size": {"rows": 3, "cols": 3}},
            "zone_thresholds": {"low": 3, "medium": 6, "high": 10},
            "alert_settings": {"enable_sound": True, "log_file": "alerts_log.txt"}
        }

config = load_config()

_model_path = config["detection_settings"]["model_path"]
_try_model_paths = [
    os.path.join(BASE_DIR, _model_path),
    os.path.join(ROOT_DIR, _model_path),
    _model_path,
]
_model_path = next((p for p in _try_model_paths if os.path.exists(p)), _model_path)
model = YOLO(_model_path)
try:
    if CUDA_AVAILABLE:
        model.to('cuda')
        if hasattr(model, 'fuse'):
            model.fuse()
except Exception:
    pass

cap = None
current_zone_data = {"total": 0, "zones": []}
alerted_zones = set()
is_streaming = False
camera_active = False
alerts_log = deque(maxlen=200)
last_alert_state = {}
state_lock = threading.Lock()
_capture_lock = threading.Lock()
_read_fail_count = 0
_READ_FAIL_REINIT_THRESHOLD = 15
_last_frame_jpeg = None
_placeholder_jpeg = None
upload_sessions = {}
upload_lock = threading.Lock()
upload_analysis_active = False
upload_analysis_stop = False
upload_analysis_thread = None
upload_analysis_path = None

FRAME_SKIP = 2
CONFIDENCE_THRESHOLD = float(config.get("detection_settings", {}).get("confidence_threshold", 0.35))
CONFIDENCE_THRESHOLD = max(0.05, min(CONFIDENCE_THRESHOLD, 0.9))
INFERENCE_IMG_SIZE = int(config.get("detection_settings", {}).get("imgsz", 512))
if INFERENCE_IMG_SIZE < 256 or INFERENCE_IMG_SIZE > 1280:
    INFERENCE_IMG_SIZE = 512
HIGH_ACCURACY = config.get("detection_settings", {}).get("high_accuracy", True)
SIMPLE_MODE = config.get("detection_settings", {}).get("simple_mode", True)
DEBUG_DETECTION = config.get("detection_settings", {}).get("debug_detection", True)
ADAPTIVE_ENABLED = True
MIN_BOX_AREA = 300
IOU_DEDUP_THRESHOLD = 0.8
CENTER_DIST_DEDUP = 25

_zero_streak = 0
latest_boxes = []
adaptive_conf = CONFIDENCE_THRESHOLD
zero_frame_streak = 0
_model_warmed = False

# ─── Multi-Camera State ─────────────────────────────────────────────────────
# camera_id -> {zone_id, last_count, last_seen, active}
multi_camera_streams = {}
multi_camera_lock = threading.Lock()

# Zone override from multi-camera (zone_id -> count from browser cameras)
browser_camera_zone_counts = {}
browser_camera_lock = threading.Lock()

# ─── Snapshot & Lost+Find State ─────────────────────────────────────────────
# person_uid -> {zone, timestamp, camera_id, snapshot_rel, full_frame_rel, total_in_frame}
snapshots_db = {}
snapshots_lock = threading.Lock()
# Track previous counts per camera to detect changes
prev_camera_counts = {}

def _warmup_model():
    global _model_warmed
    if _model_warmed:
        return
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(dummy, imgsz=INFERENCE_IMG_SIZE, classes=[0], conf=0.2, verbose=False)
        _model_warmed = True
        print("⚡ YOLO model warmed up")
    except Exception as e:
        print(f"⚠️ YOLO warmup skipped: {e}")

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def _compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    area_b = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _deduplicate_boxes(boxes):
    filtered = [b for b in boxes if max(0, b[2]-b[0]) * max(0, b[3]-b[1]) >= MIN_BOX_AREA]
    filtered.sort(key=lambda x: x[4], reverse=True)
    kept = []
    for b in filtered:
        bx_cx = (b[0]+b[2]) / 2.0; bx_cy = (b[1]+b[3]) / 2.0
        dup = False
        for k in kept:
            if _compute_iou(b, k) >= IOU_DEDUP_THRESHOLD:
                dup = True; break
            kc_x = (k[0]+k[2]) / 2.0; kc_y = (k[1]+k[3]) / 2.0
            if abs(kc_x - bx_cx) < CENTER_DIST_DEDUP and abs(kc_y - bx_cy) < CENTER_DIST_DEDUP:
                dup = True; break
        if not dup:
            kept.append(b)
    return kept

def get_level(count):
    t = config["zone_thresholds"]
    if count <= t["low"]:     return "Low"
    if count <= t["medium"]:  return "Medium"
    if count <= t["high"]:    return "High"
    return "Critical"

def yolo_worker():
    while camera_active:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        original = frame
        h, w = original.shape[:2]
        scale_factor = 1.0
        proc = original
        max_dim = max(w, h)
        try:
            if max_dim > 960:
                scale_factor = 960.0 / max_dim
                proc = cv2.resize(original, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)
        except Exception:
            proc = original; scale_factor = 1.0
        try:
            results = model.predict(proc, imgsz=INFERENCE_IMG_SIZE, conf=CONFIDENCE_THRESHOLD,
                                    classes=[0], device=DEVICE, half=USE_HALF, verbose=False)
        except Exception:
            results = []
        try:
            while not result_queue.empty():
                result_queue.get_nowait()
        except Exception:
            pass
        result_queue.put((original, results, scale_factor))

def _ensure_placeholder_frame(width=640, height=480):
    global _placeholder_jpeg
    if _placeholder_jpeg is not None:
        return _placeholder_jpeg
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, 'No camera frame available', (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', img)
    if ok:
        _placeholder_jpeg = buf.tobytes()
    return _placeholder_jpeg

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if firebase_config.db is not None:
            try:
                users_ref = firebase_config.db.collection('users')
                query = users_ref.where('username', '==', username).limit(1)
                users = query.get()
                if users:
                    user_doc = users[0]
                    user_data = user_doc.to_dict()
                    if user_data.get('password_hash') == hash_password(password):
                        session['user_id'] = user_doc.id
                        session['username'] = user_data['username']
                        session['email'] = user_data['email']
                        flash('Login successful! Welcome to CrowdVision.', 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('Invalid username or password!', 'error')
                else:
                    flash('Invalid username or password!', 'error')
            except Exception as e:
                print(f"Firebase login error: {e}")
                flash('Login service temporarily unavailable.', 'error')
        else:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT id, username, password FROM users WHERE username = ?", (username,))
            user = c.fetchone()
            conn.close()
            if user and user[2] == hash_password(password):
                session['user_id'] = user[0]
                session['username'] = user[1]
                flash('Login successful! Welcome to CrowdVision.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password!', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        if firebase_config.db is not None:
            try:
                user_data = {'username': username, 'email': email, 'password_hash': hash_password(password),
                             'created_at': datetime.now(), 'is_active': True}
                users_ref = firebase_config.db.collection('users')
                if users_ref.where('username', '==', username).limit(1).get():
                    flash('Username already exists!', 'error')
                    return render_template('register.html')
                if users_ref.where('email', '==', email).limit(1).get():
                    flash('Email already exists!', 'error')
                    return render_template('register.html')
                firebase_config.db.collection('users').add(user_data)
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash('Registration service temporarily unavailable.', 'error')
                return render_template('register.html')
        else:
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                         (username, email, hash_password(password)))
                conn.commit()
                conn.close()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username or email already exists!', 'error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('dashboard'))

@app.route('/monitoring')
@login_required
def index():
    return render_template('monitoring.html')

@app.route('/index')
@login_required
def monitoring():
    return render_template('monitoring.html')

def _init_zero_zone_data():
    grid_size = config["detection_settings"]["grid_size"]
    rows, cols = grid_size["rows"], grid_size["cols"]
    zones = []
    for r in range(rows):
        for c in range(cols):
            zones.append({"id": f"Z{r * cols + c + 1}", "count": 0, "level": "Low"})
    return {"total": 0, "zones": zones}

def initialize_camera():
    global cap, is_streaming, camera_active, _read_fail_count, _last_frame_jpeg, current_zone_data
    cam_settings = config["camera_settings"]
    tried = set()
    try_indices = [cam_settings.get("camera_index", 0), 0, 1, 2]
    opened = False
    for idx in try_indices:
        if idx in tried:
            continue
        tried.add(idx)
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(idx)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_settings.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_settings.get("height", 720))
        warm_ok = False
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                warm_ok = True
                break
            time.sleep(0.05)
        if cap.isOpened() and warm_ok:
            print(f"✅ Camera initialized on index {idx}")
            opened = True
            break
        else:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
    if not opened:
        print("❌ Error: Could not open webcam.")
        camera_active = False
        return False
    camera_active = True
    is_streaming = True
    _read_fail_count = 0
    _last_frame_jpeg = None
    with state_lock:
        current_zone_data = _init_zero_zone_data()
    _warmup_model()
    print("✅ Camera initialized and monitoring started")
    return True

def _maybe_reinit_camera():
    global cap
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    time.sleep(0.2)
    return initialize_camera()

def stop_camera():
    global cap, is_streaming, camera_active
    is_streaming = False
    camera_active = False
    if cap is not None:
        try:
            cap.release()
        finally:
            cap = None
    print("🛑 Camera stopped")

def process_frame():
    global current_zone_data, alerted_zones, last_alert_state, _read_fail_count, adaptive_conf, zero_frame_streak
    if not camera_active:
        return None, None
    with _capture_lock:
        if cap is None or not cap.isOpened():
            _read_fail_count += 1
            if _read_fail_count >= _READ_FAIL_REINIT_THRESHOLD and camera_active:
                _read_fail_count = 0
                _maybe_reinit_camera()
            return None, None
        success, frame = cap.read()
    if not success or frame is None:
        _read_fail_count += 1
        if _read_fail_count >= _READ_FAIL_REINIT_THRESHOLD and camera_active:
            _read_fail_count = 0
            _maybe_reinit_camera()
        return None, None
    _read_fail_count = 0
    if hasattr(process_frame, 'frame_count'):
        process_frame.frame_count += 1
    else:
        process_frame.frame_count = 0
    if process_frame.frame_count % FRAME_SKIP != 0:
        return frame, current_zone_data
    try:
        results = model.predict(frame, imgsz=INFERENCE_IMG_SIZE, conf=adaptive_conf,
                               classes=[0], device=DEVICE, half=USE_HALF, verbose=False)
    except Exception as e:
        print(f"⚠️ Inline inference error: {e}")
        results = []
    grid_size = config["detection_settings"]["grid_size"]
    rows, cols = grid_size["rows"], grid_size["cols"]
    height, width = frame.shape[:2]
    zone_h, zone_w = height // rows, width // cols
    zone_counts = np.zeros((rows, cols), dtype=int)
    total_count = 0
    boxes_collected = []
    for r in results:
        for box in getattr(r, 'boxes', []) or []:
            try:
                cls = int(box.cls[0])
            except Exception:
                cls = -1
            if cls != 0:
                continue
            try:
                x1f, y1f, x2f, y2f = map(float, box.xyxy[0])
                x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
                conf_val = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
                total_count += 1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                row = min(max(cy // zone_h, 0), rows - 1)
                col = min(max(cx // zone_w, 0), cols - 1)
                zone_counts[row][col] += 1
                boxes_collected.append((x1, y1, x2, y2, conf_val))
            except Exception:
                pass
    global latest_boxes
    latest_boxes = boxes_collected
    frame_data = {"total": int(total_count), "zones": []}
    if ADAPTIVE_ENABLED:
        if total_count == 0:
            zero_frame_streak += 1
            if zero_frame_streak in (5, 10, 20):
                new_conf = max(0.15, adaptive_conf - 0.1)
                if new_conf < adaptive_conf:
                    adaptive_conf = round(new_conf, 3)
        else:
            if zero_frame_streak >= 5 and adaptive_conf < CONFIDENCE_THRESHOLD:
                adaptive_conf = round(min(CONFIDENCE_THRESHOLD, adaptive_conf + 0.05), 3)
            zero_frame_streak = 0
    for row in range(rows):
        for col in range(cols):
            zone_id = f"Z{row * cols + col + 1}"
            zone_count = int(zone_counts[row][col])
            level = get_level(zone_count)
            frame_data["zones"].append({"id": zone_id, "count": zone_count, "level": level})
            if level == "Critical":
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                alert_message = f"[{ts}] ALERT: {zone_id} is in CRITICAL state with {zone_count} people"
                if HAS_WINSOUND and config["alert_settings"]["enable_sound"]:
                    try:
                        winsound.Beep(1000, 500)
                    except Exception:
                        pass
                try:
                    with open(config["alert_settings"]["log_file"], "a") as log_file:
                        log_file.write(alert_message + "\n")
                except Exception:
                    pass
                with state_lock:
                    alerts_log.append({"timestamp": ts, "zone": zone_id, "level": "CRITICAL", "count": zone_count, "message": alert_message})
                last_alert_state[zone_id] = {"level": "Critical", "count": zone_count}
                alerted_zones.add(zone_id)
            else:
                alerted_zones.discard(zone_id)
                last_alert_state[zone_id] = {"level": level, "count": zone_count}
    with state_lock:
        current_zone_data = frame_data
    try:
        with open(os.path.join(BASE_DIR, "zone_data.json"), "w") as f:
            json.dump(frame_data, f)
    except Exception:
        pass
    return frame, frame_data

def _draw_grid_overlay(frame, frame_data):
    try:
        grid_size = config["detection_settings"]["grid_size"]
        rows, cols = grid_size["rows"], grid_size["cols"]
        h, w = frame.shape[:2]
        zone_h, zone_w = h // rows, w // cols
        zone_map = {}
        for z in (frame_data or {}).get("zones", []):
            zone_map[z.get("id")] = (int(z.get("count", 0)), z.get("level", "Low"))
        def color_for(level):
            return (0, 255, 0) if level == "Low" else (0, 255, 255) if level == "Medium" else (0, 165, 255) if level == "High" else (0, 0, 255)
        for r in range(rows):
            for c in range(cols):
                zone_id = f"Z{r * cols + c + 1}"
                count, level = zone_map.get(zone_id, (0, "Low"))
                x0, y0 = c * zone_w, r * zone_h
                x1, y1 = x0 + zone_w, y0 + zone_h
                cv2.rectangle(frame, (x0, y0), (x1, y1), color_for(level), 2)
                cv2.putText(frame, f"{zone_id} {level} ({count})", (x0 + 5, y0 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_for(level), 2)
        total = int((frame_data or {}).get("total", 0))
        cv2.putText(frame, f"Total People: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for (x1, y1, x2, y2, conf) in latest_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            if DEBUG_DETECTION:
                try:
                    cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
                except Exception:
                    pass
    except Exception:
        pass

def generate_frames():
    global is_streaming, _last_frame_jpeg
    if not hasattr(generate_frames, 'worker_started') or not generate_frames.worker_started:
        t = threading.Thread(target=yolo_worker, daemon=True)
        t.start()
        generate_frames.worker_started = True
    try:
        while is_streaming:
            try:
                frame, _ = process_frame()
            except Exception as e:
                print(f"⚠️ process_frame error: {e}")
                frame = None
            if frame is not None:
                with state_lock:
                    snapshot = dict(current_zone_data) if isinstance(current_zone_data, dict) else {"total":0, "zones":[]}
                _draw_grid_overlay(frame, snapshot)
            frame_bytes = None
            if frame is not None:
                try:
                    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        frame_bytes = buffer.tobytes()
                        _last_frame_jpeg = frame_bytes
                except Exception as e:
                    print(f"⚠️ encode error: {e}")
            if frame_bytes is None:
                frame_bytes = _last_frame_jpeg or _ensure_placeholder_frame()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.02)
    finally:
        pass

@app.route('/video_feed')
@login_required
def video_feed():
    global is_streaming
    if not is_streaming:
        is_streaming = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/zones')
def get_zones():
    with state_lock:
        return jsonify(current_zone_data)

@app.route('/api/start')
@login_required
def start_camera_api():
    if camera_active:
        return jsonify({"status": "already_active", "message": "Camera is already active"})
    if initialize_camera():
        return jsonify({"status": "started", "message": "Camera started successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to initialize camera"}), 500

@app.route('/api/stop')
@login_required
def stop_camera_api():
    if not camera_active:
        return jsonify({"status": "already_stopped", "message": "Camera is already stopped"})
    stop_camera()
    return jsonify({"status": "stopped", "message": "Camera stopped"})

@app.route('/api/alerts')
@login_required
def get_alerts():
    try:
        limit = int(request.args.get('limit', 3))
    except Exception:
        limit = 3
    limit = max(1, min(limit, 50))
    with state_lock:
        recent = list(alerts_log)[-limit:][::-1]
    return jsonify({"alerts": recent})

@app.route('/api/status')
@login_required
def get_status():
    return jsonify({
        "camera_active": camera_active,
        "is_streaming": is_streaming,
        "camera_initialized": cap is not None and cap.isOpened(),
        "total_zones": len(current_zone_data.get("zones", [])),
        "alerted_zones": list(alerted_zones)
    })

# ────────────────────────────────────────────────────────────────────────────
# Multi-Camera Browser API
# ────────────────────────────────────────────────────────────────────────────

def _auto_snapshot(frame, boxes, zone_id, camera_id):
    """Save per-person crops and full-frame snapshot with unique IDs."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    group_uid = str(uuid.uuid4())[:8]
    
    # Save full frame with all persons
    full_uid = f"group_{group_uid}"
    full_path = os.path.join(SNAPSHOTS_DIR, f"{full_uid}_{ts_file}.jpg")
    cv2.imwrite(full_path, frame)
    
    person_uids = []
    for i, box in enumerate(boxes):
        uid = f"{group_uid}_p{i}"
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        h, w = frame.shape[:2]
        x1c = max(0, x1 - 10); y1c = max(0, y1 - 10)
        x2c = min(w, x2 + 10); y2c = min(h, y2 + 10)
        crop = frame[y1c:y2c, x1c:x2c]
        snap_filename = f"{uid}_{ts_file}.jpg"
        snap_path = os.path.join(SNAPSHOTS_DIR, snap_filename)
        if crop.size > 0:
            cv2.imwrite(snap_path, crop)
        with snapshots_lock:
            snapshots_db[uid] = {
                'uid': uid,
                'group_uid': group_uid,
                'zone': zone_id,
                'timestamp': ts,
                'camera_id': camera_id,
                'person_index': i,
                'total_in_frame': len(boxes),
                'snapshot_file': snap_filename,
                'full_frame_file': f"{full_uid}_{ts_file}.jpg",
                'box': [x1, y1, x2, y2],
                'conf': box.get('conf', 0),
            }
        person_uids.append(uid)
    return person_uids

@app.route('/api/analyze_camera_frame', methods=['POST'])
def analyze_camera_frame():
    """Receive base64 frame from browser camera, run YOLO, auto-snapshot on change."""
    data = request.get_json(force=True)
    camera_id = data.get('camera_id', 'cam_default')
    zone_id = data.get('zone_id', 'Z1')
    frame_b64 = data.get('frame', '')
    
    if not frame_b64:
        return jsonify({'error': 'No frame data'}), 400
    
    try:
        # Strip data URL prefix if present
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]
        img_data = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
    except Exception as e:
        return jsonify({'error': f'Frame decode error: {e}'}), 400
    
    # Run YOLO
    try:
        _warmup_model()
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes=[0],
                                device=DEVICE, half=USE_HALF, verbose=False)
    except Exception as e:
        return jsonify({'error': f'YOLO inference error: {e}'}), 500
    
    boxes = []
    for r in results:
        for box in getattr(r, 'boxes', []) or []:
            try:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
                boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': round(conf, 3)})
            except Exception:
                pass
    
    count = len(boxes)
    
    # Auto-snapshot if count changed
    prev_count = prev_camera_counts.get(camera_id, -1)
    snapshot_uids = []
    if count != prev_count and count > 0:
        try:
            snapshot_uids = _auto_snapshot(frame, boxes, zone_id, camera_id)
        except Exception as e:
            print(f"⚠️ Snapshot error: {e}")
    prev_camera_counts[camera_id] = count
    
    # Update browser camera zone counts
    with browser_camera_lock:
        browser_camera_zone_counts[zone_id] = {
            'count': count,
            'camera_id': camera_id,
            'last_seen': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # Update multi-camera stream state
    with multi_camera_lock:
        multi_camera_streams[camera_id] = {
            'zone_id': zone_id,
            'last_count': count,
            'last_seen': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'active': True,
        }
    
    # Merge browser camera counts into current_zone_data
    _merge_browser_camera_counts()
    
    return jsonify({
        'count': count,
        'boxes': boxes,
        'zone_id': zone_id,
        'snapshot_uids': snapshot_uids,
        'changed': count != prev_count
    })

def _merge_browser_camera_counts():
    """Merge browser camera zone counts into the global current_zone_data."""
    with browser_camera_lock:
        cam_zone_data = dict(browser_camera_zone_counts)
    
    with state_lock:
        zones = current_zone_data.get('zones', [])
        for zone in zones:
            zid = zone['id']
            if zid in cam_zone_data:
                zone['count'] = cam_zone_data[zid]['count']
                zone['level'] = get_level(zone['count'])
        current_zone_data['total'] = sum(z['count'] for z in zones)

@app.route('/api/multi_camera_status')
def multi_camera_status():
    with multi_camera_lock:
        return jsonify(dict(multi_camera_streams))

@app.route('/api/browser_zone_counts')
def browser_zone_counts():
    with browser_camera_lock:
        return jsonify(dict(browser_camera_zone_counts))

# ────────────────────────────────────────────────────────────────────────────
# Snapshots & Lost+Find
# ────────────────────────────────────────────────────────────────────────────

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)

@app.route('/api/snapshots')
def get_snapshots():
    """Return recent snapshots list."""
    try:
        limit = int(request.args.get('limit', 50))
    except Exception:
        limit = 50
    with snapshots_lock:
        snaps = list(snapshots_db.values())
    snaps.sort(key=lambda x: x['timestamp'], reverse=True)
    result = []
    for s in snaps[:limit]:
        s_copy = dict(s)
        s_copy['snapshot_url'] = f"/snapshots/{s['snapshot_file']}" if os.path.exists(os.path.join(SNAPSHOTS_DIR, s.get('snapshot_file', ''))) else None
        s_copy['full_frame_url'] = f"/snapshots/{s['full_frame_file']}" if os.path.exists(os.path.join(SNAPSHOTS_DIR, s.get('full_frame_file', ''))) else None
        result.append(s_copy)
    return jsonify({"snapshots": result, "total": len(snaps)})

@app.route('/api/find_person', methods=['POST'])
def find_person():
    """Find a person by uploaded photo using visual similarity matching."""
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400
    
    photo = request.files['photo']
    img_data = photo.read()
    np_arr = np.frombuffer(img_data, np.uint8)
    query_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if query_img is None:
        return jsonify({'error': 'Could not decode image'}), 400
    
    # Compute color histogram for query image
    query_hist = cv2.calcHist([query_img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    cv2.normalize(query_hist, query_hist)
    query_flat = query_hist.flatten()
    
    matches = []
    with snapshots_lock:
        snap_items = list(snapshots_db.items())
    
    for uid, snap in snap_items:
        snap_path = os.path.join(SNAPSHOTS_DIR, snap.get('snapshot_file', ''))
        if not os.path.exists(snap_path):
            continue
        snap_img = cv2.imread(snap_path)
        if snap_img is None:
            continue
        try:
            snap_hist = cv2.calcHist([snap_img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
            cv2.normalize(snap_hist, snap_hist)
            snap_flat = snap_hist.flatten()
            score = float(cv2.compareHist(query_flat.reshape(query_hist.shape),
                                          snap_flat.reshape(snap_hist.shape),
                                          cv2.HISTCMP_CORREL))
        except Exception:
            score = 0.0
        
        matches.append({
            'uid': uid,
            'score': round(score, 4),
            'zone': snap['zone'],
            'timestamp': snap['timestamp'],
            'camera_id': snap['camera_id'],
            'person_index': snap['person_index'],
            'total_in_frame': snap['total_in_frame'],
            'snapshot_url': f"/snapshots/{snap['snapshot_file']}",
            'full_frame_url': f"/snapshots/{snap['full_frame_file']}",
        })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    top = matches[:10]
    
    if not top:
        return jsonify({'matches': [], 'message': 'No snapshots in database yet. Snapshots are captured automatically when people are detected.'})
    
    best = top[0]
    message = f"Best match found in {best['zone']} at {best['timestamp']} (similarity: {best['score']:.1%})"
    if best['score'] < 0.3:
        message = "Low confidence match - person may not be in the system yet."
    
    return jsonify({
        'matches': top,
        'best_match': best,
        'message': message
    })

# ────────────────────────────────────────────────────────────────────────────
# Video Upload (original functionality preserved)
# ────────────────────────────────────────────────────────────────────────────

ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}

def allowed_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_VIDEO_EXTS

@app.route('/downloads/<path:filename>')
@login_required
def downloads(filename):
    return send_from_directory(DOWNLOADS_DIR, filename, as_attachment=False)

@app.route('/upload', methods=['GET', 'POST'])
def video_upload():
    if request.method == 'GET':
        token = request.args.get('session')
        if token:
            with upload_lock:
                sess = upload_sessions.get(token)
            if not sess:
                flash('Upload session not found or finished.', 'error')
                return render_template('upload.html')
            return render_template('upload.html', session_token=token, feed_url=url_for('uploaded_feed', token=token))
        return render_template('upload.html')
    file = request.files.get('video')
    if not file or file.filename == '':
        flash('Please choose a video file.', 'error')
        return redirect(url_for('video_upload'))
    if not allowed_video(file.filename):
        flash('Unsupported file type.', 'error')
        return redirect(url_for('video_upload'))
    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    upload_path = os.path.join(UPLOADS_DIR, f"{timestamp}_{safe_name}")
    file.save(upload_path)
    token = uuid.uuid4().hex
    with upload_lock:
        upload_sessions[token] = {
            'path': upload_path,
            'metrics': {'fps': 0.0, 'frame_width': 0, 'frame_height': 0, 'total_frames': 0,
                        'latest_people': 0, 'avg_people': 0.0, 'max_people': 0,
                        'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'done': False, 'error': None},
            'active': True,
            'zone_data': make_default_zone_data(),
        }
    return redirect(url_for('video_upload', session=token))

def _generate_uploaded_frames(token: str):
    with upload_lock:
        sess = upload_sessions.get(token)
    if not sess:
        return
    video_path = sess['path']
    cap_u = cv2.VideoCapture(video_path)
    if not cap_u.isOpened():
        with upload_lock:
            sess['metrics']['error'] = 'Failed to open the uploaded video.'
            sess['active'] = False
        return
    fps = cap_u.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if fps > 0 else 25.0
    delay = 1.0 / float(fps)
    frame_w = int(cap_u.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_h = int(cap_u.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    people_counts = []
    total_frames = 0
    with upload_lock:
        sess['metrics']['fps'] = fps
        sess['metrics']['frame_width'] = frame_w
        sess['metrics']['frame_height'] = frame_h
    try:
        while True:
            ok, frame = cap_u.read()
            if not ok or frame is None:
                with upload_lock:
                    sess['metrics']['done'] = True
                    sess['active'] = False
                break
            try:
                results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes=[0],
                                       device=DEVICE, half=USE_HALF, verbose=False)
            except Exception as e:
                results = []
            raw_boxes = []
            for r in results:
                for box in getattr(r, 'boxes', []) or []:
                    try:
                        if int(box.cls[0]) != 0:
                            continue
                        conf = float(box.conf[0]) if getattr(box, 'conf', None) is not None else 0.0
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        raw_boxes.append((x1, y1, x2, y2, conf))
                    except Exception:
                        pass
            unique_boxes = _deduplicate_boxes(raw_boxes)
            count_people = len(unique_boxes)
            for (x1, y1, x2, y2, conf) in unique_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"People: {count_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            people_counts.append(int(count_people))
            total_frames += 1
            avg_people = (sum(people_counts) / total_frames) if total_frames else 0
            max_people = max(people_counts) if people_counts else 0
            with upload_lock:
                if token in upload_sessions:
                    upload_sessions[token]['metrics'].update({
                        'total_frames': total_frames, 'latest_people': int(count_people),
                        'avg_people': round(avg_people, 2), 'max_people': int(max_people)
                    })
            ok_j, buffer = cv2.imencode('.jpg', frame)
            if ok_j:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(delay)
    finally:
        try:
            cap_u.release()
        except Exception:
            pass

@app.route('/uploaded_feed/<token>')
def uploaded_feed(token: str):
    return Response(_generate_uploaded_frames(token), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload_metrics/<token>')
@login_required
def upload_metrics(token: str):
    with upload_lock:
        sess = upload_sessions.get(token)
        data = sess['metrics'] if sess else {'error': 'not_found'}
    return jsonify(data)

@app.route('/video-upload')
def upload_alias():
    return redirect(url_for('video_upload'))

@app.route('/api/upload-video', methods=['POST'])
def api_upload_video():
    file = request.files.get('video')
    if not file or file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    if not allowed_video(file.filename):
        return jsonify({'status': 'error', 'message': 'Unsupported file type'}), 400
    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    upload_path = os.path.join(UPLOADS_DIR, f"{timestamp}_{safe_name}")
    try:
        file.save(upload_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to save file: {e}'}), 500
    token = uuid.uuid4().hex
    with upload_lock:
        upload_sessions[token] = {
            'path': upload_path,
            'metrics': {'fps': 0.0, 'frame_width': 0, 'frame_height': 0, 'total_frames': 0,
                        'latest_people': 0, 'avg_people': 0.0, 'max_people': 0,
                        'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'done': False, 'error': None},
            'active': True,
            'zone_data': make_default_zone_data(),
        }
    return jsonify({'status': 'success', 'filepath': upload_path, 'filename': safe_name, 'session_token': token})

def make_default_zone_data():
    try:
        grid_size = config["detection_settings"]["grid_size"]
        rows, cols = grid_size.get("rows", 3), grid_size.get("cols", 3)
    except Exception:
        rows, cols = 3, 3
    data = {"total": 0, "zones": []}
    for r in range(rows):
        for c in range(cols):
            data["zones"].append({"id": f"Z{r * cols + c + 1}", "count": 0, "level": "Low"})
    return data

try:
    current_zone_data = make_default_zone_data()
except Exception:
    pass

if __name__ == '__main__':
    init_db()
    srv = config.get('server_settings', {})
    host = srv.get('host', '0.0.0.0')
    port = int(srv.get('port', 5000))
    debug = bool(srv.get('debug', False))
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
