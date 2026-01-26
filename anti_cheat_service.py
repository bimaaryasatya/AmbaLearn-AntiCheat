import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, g, request
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import base64
import sqlite3
import time
from datetime import datetime, timezone

# -----------------------
# CONFIG
# -----------------------
DB_PATH = "cheat_logs.db"
HEAD_THRESHOLD = 4     # Number of frames to trigger head turn alert (approx 2s)
GAZE_THRESHOLD = 4     # Number of frames to trigger gaze alert (approx 2s)
MAX_VIOLATIONS = 3     # Max violations before auto-submit
# -----------------------

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_interval=10, ping_timeout=30)

# -----------------------
# MOCK / FALLBACK MODE SETUP
# -----------------------
MOCK_MODE = False
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe loaded successfully.")
    MOCK_MODE = False

except Exception as e:
    print(f"⚠️ MediaPipe Error: {e}")
    print("⚠️ Switching to OpenCV Haar Cascade (Fallback Mode).")
    MOCK_MODE = True
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
# -----------------------
# SIMPLE SQLITE HELPERS
# -----------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        cur = db.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            socket_id TEXT,
            event_type TEXT,
            detail TEXT,
            timestamp TEXT
        )
        """)
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def log_event(socket_id, event_type, detail=""):
    try:
        # Use a new connection for logging to avoid thread issues if outside request context
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            ts = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "INSERT INTO events (socket_id, event_type, detail, timestamp) VALUES (?, ?, ?, ?)",
                (socket_id, event_type, detail, ts)
            )
            conn.commit()
    except Exception as e:
        print(f"Log Error: {e}")

# Init DB on server start
if not os.path.exists(DB_PATH):
    # Just a trick to ensure DB is created since we need app context usually
    pass 

# -----------------------
# Per-socket counters storage
# -----------------------
# Structure: { sid: { 'head_counter': int, 'gaze_counter': int, 'violation_count': int, 'disqualified': bool, 'last_seen': ts } }
clients = {}

# -----------------------
# PROCESS FRAME
# -----------------------
def analyze_face(frame):
    """
    Returns processed_frame, status_info dict.
    """
    h, w, _ = frame.shape
    
    status_info = {
        'num_faces': 0,
        'head_alert': False,
        'gaze_dir': 'Unknown'
    }
    
    status = "Normal"
    color = (0, 255, 0)

    if not MOCK_MODE:
        # --- MEDIAPIPE LOGIC ---
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            status_info['num_faces'] = num_faces
            cv2.putText(frame, f"Faces: {num_faces} (MP)", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if num_faces > 1:
                status = "⚠️ Multiple Faces Detected"
                color = (0, 0, 255)
            else:
                # Single face logic (Head/Gaze)
                face_landmarks = results.multi_face_landmarks[0]
                
                # Head Pose
                nose = face_landmarks.landmark[1]
                left_eye_outer = face_landmarks.landmark[33]
                right_eye_outer = face_landmarks.landmark[263]
                nose_x, left_x, right_x = int(nose.x * w), int(left_eye_outer.x * w), int(right_eye_outer.x * w)
                ratio_head = (right_x - nose_x) / (nose_x - left_x + 1e-6)
                
                if ratio_head < 0.6 or ratio_head > 1.6:
                    status = "⚠️ Menoleh - Potensi Cheating"
                    color = (0, 0, 255)
                    status_info['head_alert'] = True

                # Gaze
                try:
                    left_eye_left = face_landmarks.landmark[33]
                    left_eye_right = face_landmarks.landmark[133]
                    left_pupil = face_landmarks.landmark[468]

                    right_eye_left = face_landmarks.landmark[362]
                    right_eye_right = face_landmarks.landmark[263]
                    right_pupil = face_landmarks.landmark[473]

                    left_ratio = (left_pupil.x - left_eye_left.x) / (left_eye_right.x - left_eye_left.x + 1e-6)
                    right_ratio = (right_pupil.x - right_eye_left.x) / (right_eye_right.x - right_eye_left.x + 1e-6)
                    gaze_ratio = (left_ratio + right_ratio) / 2

                    if gaze_ratio < 0.40:
                        gaze_dir = "Kiri"
                    elif gaze_ratio > 0.60:
                        gaze_dir = "Kanan"
                    else:
                        gaze_dir = "Tengah"
                    status_info['gaze_dir'] = gaze_dir

                    # draw small circles for these landmarks
                    for i in [33, 133, 362, 263, 468, 473]:
                        lm = face_landmarks.landmark[i]
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                                   2, (0, 255, 255), -1)
                    cv2.putText(frame, f"Gaze: {gaze_dir}",
                                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 0), 2)
                except Exception:
                    status_info['gaze_dir'] = 'Unknown'
        else:
             status = "Wajah tidak terdeteksi"
             color = (0, 165, 255)
             
    else:
        # --- FALLBACK (OPENCV) LOGIC ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        num_faces = len(faces)
        status_info['num_faces'] = num_faces
        
        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 0, 0), 2)
            
        cv2.putText(frame, f"Faces: {num_faces} (CV2)", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if num_faces > 1:
            status = "⚠️ Multiple Faces Detected"
            color = (0, 0, 255)
        elif num_faces == 0:
             status = "Wajah tidak terdeteksi"
             color = (0, 165, 255)
        else:
            # Fake Gaze/Head check for Fallback? 
            # We can't easily detect head turn with just bounding box.
            # We will just report Normal.
            status = "Normal (Fallback)"
            
    # Draw Status
    if status != "Normal" and status != "Normal (Fallback)":
         cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return frame, status_info

# -----------------------
# SOCKET.IO HANDLERS
# -----------------------
@socketio.on("connect")
def on_connect():
    sid = request.sid
    # Initialize client state including violation_count and disqualified flag
    client_type = request.args.get('client_type', 'unknown')
    clients[sid] = {
        'head_counter': 0, 
        'gaze_counter': 0, 
        'violation_count': 0,
        'disqualified': False,
        'exam_active': False,
        'last_seen': time.time(),
        'client_type': client_type
    }
    print(f"[CONNECT] {sid} (Type: {client_type})")
    log_event(sid, "connect", f"client connected ({client_type})")
    emit("connected", {"message": "connected", "sid": sid})

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    print(f"[DISCONNECT] {sid}")
    log_event(sid, "disconnect", "client disconnected")
    clients.pop(sid, None)

@socketio.on("start_exam")
def on_start_exam(data):
    sid = request.sid
    if sid in clients:
        clients[sid]['exam_active'] = True
        # Reset counters just in case
        clients[sid]['head_counter'] = 0
        clients[sid]['gaze_counter'] = 0
        print(f"[START EXAM] {sid} - Violation tracking active. Data: {data}")
        log_event(sid, "start_exam", "User started exam")

@socketio.on("send_frame")
def receive_frame(base64_data):
    sid = request.sid
    if sid not in clients:
        clients[sid] = {
            'head_counter': 0, 
            'gaze_counter': 0, 
            'violation_count': 0,
            'disqualified': False,
            'exam_active': False,
            'last_seen': time.time(),
            'client_type': 'unknown'
        }
    
    clients[sid]['last_seen'] = time.time()

    # If disqualified, do not process further
    if clients[sid]['disqualified']:
        return

    try:
        # decode frame
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
        jpg = base64.b64decode(base64_data)
        np_arr = np.frombuffer(jpg, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # analyze
        cv2.imwrite("debug_frame.jpg", frame) # DEBUG: Save frame to check orientation/quality
        processed_frame, info = analyze_face(frame)

        # update counters based on analysis
        head_alert = info.get('head_alert', False)
        gaze_dir = info.get('gaze_dir', 'Unknown')
        num_faces = info.get('num_faces', 0)
        
        # Only track violations if exam is active
        exam_active = clients[sid].get('exam_active', False)
        reasons = []

        if exam_active:
            # reset internal counters if everything is normal to avoid noise
            if num_faces == 1 and not head_alert and gaze_dir == "Tengah":
                 clients[sid]['head_counter'] = max(0, clients[sid]['head_counter'] - 1)
                 clients[sid]['gaze_counter'] = max(0, clients[sid]['gaze_counter'] - 1)
            else:
                # If no face, treat as head violation (participant left or turned away completely)
                if num_faces == 0:
                    clients[sid]['head_counter'] += 1
                    clients[sid]['gaze_counter'] = max(0, clients[sid]['gaze_counter'] - 1)
                else:
                    if head_alert:
                        clients[sid]['head_counter'] += 1
                    else:
                        clients[sid]['head_counter'] = max(0, clients[sid]['head_counter'] - 1)

                    if gaze_dir in ("Kiri", "Kanan"):
                        clients[sid]['gaze_counter'] += 1
                    else:
                        clients[sid]['gaze_counter'] = max(0, clients[sid]['gaze_counter'] - 1)

            # If counters cross thresholds -> emit cheating_alert + log
            if clients[sid]['head_counter'] > HEAD_THRESHOLD:
                reasons.append("Menoleh terlalu lama")
                clients[sid]['head_counter'] = 0 # Reset to require new sustained violation
                
            if clients[sid]['gaze_counter'] > GAZE_THRESHOLD:
                reasons.append("Melihat keluar layar terlalu lama")
                clients[sid]['gaze_counter'] = 0 # Reset to require new sustained violation

            if num_faces > 1:
                reasons.append("Multiple faces detected")
                
            if reasons:
                detail = "; ".join(reasons)
                clients[sid]['violation_count'] += 1
                current_violations = clients[sid]['violation_count']
                
                print(f"[ALERT] {sid} -> {detail} (Violation {current_violations}/{MAX_VIOLATIONS})")
                log_event(sid, "cheating_alert", f"{detail} ({current_violations})")
                
                emit("cheating_alert", {
                    "sid": sid, 
                    "detail": detail, 
                    "violation_count": current_violations,
                    "max_violations": MAX_VIOLATIONS
                })

                # Check for 3-strike rule
                if current_violations >= MAX_VIOLATIONS:
                    print(f"[DISQUALIFIED] {sid} reached limit.")
                    clients[sid]['disqualified'] = True
                    emit("auto_submit", {"sid": sid, "detail": "Limit pelanggaran tercapai."})
                    log_event(sid, "auto_submit", "Violation limit exceeded")
        
        # Emit continuous status for calibration/tracking
        # This helps the frontend know if face is detected or if there are warnings
        status_payload = {
            "num_faces": num_faces,
            "head_alert": head_alert,
            "gaze_dir": gaze_dir,
            "status": "Cheating Detected" if reasons else "Normal",
            "exam_active": exam_active
        }
        emit("status", status_payload)

        # encode processed_frame and send back
        _, buf = cv2.imencode(".jpg", processed_frame)
        processed_b64 = base64.b64encode(buf).decode("utf-8")
        
        if clients[sid].get('client_type') == 'web':
            emit("processed_frame", {
                "image": processed_b64,
                "num_faces": num_faces
            })
        else:
            emit("processed_frame", processed_b64)

    except Exception as e:
        print("ERROR processing frame:", e)
        log_event(sid, "error", str(e))

@app.route("/")
def index():
    return "Anti-Cheat Service Running"

if __name__ == "__main__":
    with app.app_context():
        init_db()
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
